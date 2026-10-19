import Foundation
import Metal

// Reading one expert into a slot, raw or compressed.
//
// Split out of `PreadExpertStreamer.swift` under the 500-line-per-file rule when
// the compressed path landed. Pure code motion: the declarations and their
// access are unchanged, and the class has no `private` storage these touch.
extension PreadExpertStreamer {
    public func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64)
    {
        // K12: slot selection and fill share one critical section so the
        // round-robin path never lands on a slot a concurrent plan reserved
        // (`loading`) and no fill can interleave with another pread.
        cacheLock.lock()
        defer { cacheLock.unlock() }
        var candidate = nextSlot
        var scanned = 0
        while (slotState[candidate] == .loading || slotPinCount[candidate] > 0)
            && scanned < slotCount
        {
            candidate = (candidate + 1) % slotCount
            scanned += 1
        }
        guard scanned < slotCount else {
            throw ModelError.expertCacheUnplaceable(
                detail: "all \(slotCount) expert-cache slots are loading or pinned")
        }
        nextSlot = (candidate + 1) % slotCount
        return try loadExpertUnlocked(layer: layer, expert: expert, slot: candidate)
    }

    public func loadExpert(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64)
    {
        guard slot >= 0 && slot < slotCount else {
            throw StreamerError.slotOutOfRange(slot)
        }
        // K12: the pread fill and the slot bookkeeping share one critical
        // section so a concurrent plan/execute or another load cannot write
        // into this slot while the pread is in flight.
        cacheLock.lock()
        defer { cacheLock.unlock() }
        guard slotState[slot] != .loading, slotPinCount[slot] == 0 else {
            throw ModelError.expertCacheUnplaceable(
                detail: "expert-cache slot \(slot) is loading or pinned")
        }
        return try loadExpertUnlocked(layer: layer, expert: expert, slot: slot)
    }

    /// Fill `slot` with `expert` and update bookkeeping. Callers hold
    /// `cacheLock`.
    /// Reads one compressed expert: a read per frame into the staging buffer,
    /// then a decode into the slot at that frame's offset.
    ///
    /// The frame count is the layer's shape, not the expert's: a split layer has
    /// two frames (gate|up, then the tail) and a whole layer has one, so the
    /// decoded size comes from `phase1Bytes` and the stride rather than from the
    /// blob.
    private func readCompressedExpert(
        frames: [ExpertBlobIndex.Blob], slot: Int
    ) throws {
        guard let scratch = compressedScratch else {
            throw StreamerError.sizeMismatch(expected: 1, actual: 0)
        }
        let stride = Int(layout.expertStride)
        let split = layout.phase1Bytes.map(Int.init)
        var written = 0
        for (index, blob) in frames.enumerated() {
            let packedBytes = Int(blob.length)
            guard packedBytes <= compressedScratchBytes else {
                throw StreamerError.sizeMismatch(
                    expected: UInt64(compressedScratchBytes), actual: UInt64(blob.length))
            }
            try readFull(
                into: scratch,
                fileOffset: layout.streamOffset + blob.offset,
                count: packedBytes)
            let expected: Int
            if frames.count == 2, index == 0 {
                expected = split ?? 0
            } else {
                expected = stride - written
            }
            guard expected > 0 else {
                throw StreamerError.sizeMismatch(expected: 1, actual: 0)
            }
            try ExpertBlobCodec.decompress(
                source: scratch, count: packedBytes,
                into: slotPointers[slot].advanced(by: written),
                expected: expected)
            written += expected
        }
        guard written == stride else {
            throw StreamerError.sizeMismatch(
                expected: UInt64(stride), actual: UInt64(written))
        }
    }

    func loadExpertUnlocked(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64)
    {
        // A compressed layer is addressed through its index, not by
        // `n * stride`, so the stride-shaped range check does not describe it:
        // the frames are laid out back to back and are shorter than the stride.
        // The guard for that path is each frame's own end, checked below.
        let regionOffset: UInt64
        if let frames = compressedFrames {
            let expertFrames = frames[expert]
            let end = expertFrames.map { $0.offset + UInt64($0.length) }.max() ?? 0
            guard end <= layout.streamSize else {
                throw StreamerError.offsetOutOfRange(end)
            }
            regionOffset = 0
        } else {
            regionOffset = layout.expertOffset(layer: layer, expert: expert)
            // Checked for the same reason: a wrapped sum here would read as "in
            // range" and the pread below would take an offset past the end of the
            // mapped file. With this and the open-time check, every sum in this
            // type that mixes a layout offset with a region is known not to wrap.
            let (regionEnd, regionOverflow) =
                regionOffset
                .addingReportingOverflow(layout.expertStride)
            guard !regionOverflow, regionEnd <= layout.streamSize else {
                throw StreamerError.offsetOutOfRange(regionOffset)
            }
        }
        slotGeneration[slot] &+= 1
        let previousExpert = slotExpert[slot]
        if previousExpert >= 0 {
            publishResidencyUnlocked(
                expert: previousExpert,
                slot: slot,
                state: ExpertResidencyEntry.empty,
                generation: slotGeneration[slot])
        }
        slotExpert[slot] = expert
        slotState[slot] = .loading
        publishResidencyUnlocked(
            expert: expert,
            slot: slot,
            state: ExpertResidencyEntry.loading,
            generation: slotGeneration[slot])
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        do {
            if let frames = compressedFrames {
                try readCompressedExpert(
                    frames: frames[expert], slot: slot)
            } else {
                try readFull(
                    into: slotPointers[slot],
                    fileOffset: layout.streamOffset + regionOffset,
                    count: Int(layout.expertStride))
            }
            slotState[slot] = .resident
            publishResidencyUnlocked(
                expert: expert,
                slot: slot,
                state: ExpertResidencyEntry.resident,
                generation: slotGeneration[slot])
            slotLastUse[slot] = useClock
            recordSuccessfulLoadsUnlocked(
                experts: [expert], elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
        } catch {
            slotState[slot] = .empty
            slotExpert[slot] = -1
            publishResidencyUnlocked(
                expert: expert,
                slot: slot,
                state: ExpertResidencyEntry.empty,
                generation: slotGeneration[slot])
            throw error
        }
        return (slotBuffers[slot], slotBufferOffsets[slot], layout.expertStride)
    }

}
