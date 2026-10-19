import Foundation
import Synchronization
import TinyTitanFormat
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
    /// Reads and decodes one compressed expert into its slot.
    ///
    /// The decode sizes come from the layer's geometry, not from the blobs: a
    /// split layer's phase 0 is `phase1Bytes` and its tail is the rest, and a v3
    /// index further splits each phase into `compressedBlockBytes` blocks.
    func readCompressedExpert(
        frames: [ExpertBlobIndex.Blob], slot: Int
    ) throws {
        try decodeCompressedEntry(frames, into: slotPointers[slot])
    }

    /// One expert: read every block, then decode the phases into `destination`.
    func decodeCompressedEntry(
        _ pack: [ExpertBlobIndex.Blob], into destination: UnsafeMutableRawPointer
    ) throws {
        let bytes = pack.reduce(0) { $0 + Int($1.length) }
        guard bytes > 0, bytes <= compressedScratchBytes else {
            throw StreamerError.sizeMismatch(
                expected: UInt64(compressedScratchBytes), actual: UInt64(bytes))
        }
        let scratch = CompressedScratchPool.shared.borrow(bytes: bytes)
        defer { CompressedScratchPool.shared.giveBack(scratch, bytes: bytes) }
        var offsets: [UInt64] = []
        var counts: [Int] = []
        var destinations: [UnsafeMutableRawPointer] = []
        var cursor = 0
        for frame in pack {
            offsets.append(layout.streamOffset + frame.offset)
            counts.append(Int(frame.length))
            destinations.append(scratch.advanced(by: cursor))
            cursor += Int(frame.length)
        }
        try readRanges(offsets: offsets, counts: counts, into: destinations)
        let work = try blockWork(
            pack, scratch: scratch, destination: destination)
        try decodeBlocks(work, phase: 0)
        try decodeBlocks(work, phase: 1)
    }

    /// A whole set of experts: one storage call for all of it, then every block
    /// of every expert decodes in parallel.
    ///
    /// This is the demand wave and the prefetch ring's arm in one place. Both
    /// need the same two properties: one read (the reader serialises concurrent
    /// batches, so N independent fetches would queue) and independent blocks
    /// (a frame is one serial LZFSE stream, so the latency floor is a block, not
    /// an expert). `firstStage` runs after the phase-0 blocks land and before
    /// the tails decode, which is the overlap the GPU needs.
    func decodeCompressedSet(
        experts: [Int],
        frames: [[ExpertBlobIndex.Blob]],
        into destinations: [UnsafeMutableRawPointer],
        firstStage: (() -> Void)? = nil,
        reader: ParallelExpertReader? = nil
    ) throws {
        precondition(
            experts.count == destinations.count,
            "each compressed expert needs one destination")
        var borrowed: [(pointer: UnsafeMutableRawPointer, bytes: Int)] = []
        defer {
            for item in borrowed {
                CompressedScratchPool.shared.giveBack(item.pointer, bytes: item.bytes)
            }
        }
        var offsets: [UInt64] = []
        var counts: [Int] = []
        var readDestinations: [UnsafeMutableRawPointer] = []
        var scratchByExpert: [UnsafeMutableRawPointer] = []
        var packed: [[ExpertBlobIndex.Blob]] = []
        for expert in experts {
            let pack = frames[expert]
            let bytes = pack.reduce(0) { $0 + Int($1.length) }
            guard bytes > 0, bytes <= compressedScratchBytes else {
                throw StreamerError.sizeMismatch(
                    expected: UInt64(compressedScratchBytes), actual: UInt64(bytes))
            }
            let scratch = CompressedScratchPool.shared.borrow(bytes: bytes)
            borrowed.append((scratch, bytes))
            // One read per phase, not per block: the blocks of a phase are
            // contiguous, and 256 KiB reads pay the disk's per-read latency --
            // measured 2.5-3.4 ms for a 2-expert wave split by block against
            // ~1.2 ms for the same bytes in phase-sized ranges. The blocks
            // still decode independently; only the I/O granularity is coarser.
            var cursor = 0
            for range in packedPhaseRanges(pack) {
                offsets.append(layout.streamOffset + range.offset)
                counts.append(range.count)
                readDestinations.append(scratch.advanced(by: cursor))
                cursor += range.count
            }
            scratchByExpert.append(scratch)
            packed.append(pack)
        }
        let tRead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try readRanges(
            offsets: offsets, counts: counts, into: readDestinations, reader: reader)
        let tFront = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var work: [CompressedBlockWork] = []
        for (position, pack) in packed.enumerated() {
            work.append(
                contentsOf: try blockWork(
                    pack, scratch: scratchByExpert[position],
                    destination: destinations[position]))
        }
        try decodeBlocks(work, phase: 0)
        let tPublished = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        firstStage?()
        try decodeBlocks(work, phase: 1)
        let tTail = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        if ProcessInfo.processInfo.environment["TINYTITAN_COMPRESSED_TRACE"] == "1" {
            FileHandle.standardError.write(
                Data(
                    String(
                        format:
                            "[comp] experts=%d blocks=%d read=%.2f front=%.2f tail=%.2f ms\n",
                        experts.count, work.count,
                        Double(tFront - tRead) / 1e6,
                        Double(tPublished - tFront) / 1e6,
                        Double(tTail - tPublished) / 1e6
                    ).utf8))
        }
    }

    /// The packed byte runs to read for one expert: one contiguous range per
    /// phase, in the same order as its blocks.
    func packedPhaseRanges(
        _ pack: [ExpertBlobIndex.Blob]
    ) -> [(offset: UInt64, count: Int)] {
        let phase0 = layout.phase1Bytes ?? layout.expertStride
        let phase1 = layout.phase1Bytes.map { layout.expertStride - $0 }
        let phase0Blocks: Int
        let phase1Blocks: Int
        if compressedBlockBytes == 0 {
            phase0Blocks = pack.isEmpty ? 0 : 1
            phase1Blocks = pack.count > 1 ? 1 : 0
        } else {
            phase0Blocks = ExpertBlobIndex.blockCount(
                phaseBytes: phase0, blockBytes: compressedBlockBytes)
            phase1Blocks =
                phase1.map {
                    ExpertBlobIndex.blockCount(
                        phaseBytes: $0, blockBytes: compressedBlockBytes)
                } ?? 0
        }
        var ranges: [(UInt64, Int)] = []
        var index = 0
        for blocks in [phase0Blocks, phase1Blocks] where blocks > 0 {
            guard index + blocks <= pack.count else { return ranges }
            let offset = pack[index].offset
            let count = pack[index..<(index + blocks)]
                .reduce(0) { $0 + Int($1.length) }
            ranges.append((offset, count))
            index += blocks
        }
        return ranges
    }

    /// One block's read/decode geometry inside an expert.
    struct CompressedBlockWork {
        let scratch: UnsafeMutableRawPointer
        let packedOffset: Int
        let packedCount: Int
        let destination: UnsafeMutableRawPointer
        let destinationOffset: Int
        let expected: Int
        let phase: Int
    }

    /// Expands one expert's blocks into the work the decoder runs.
    func blockWork(
        _ pack: [ExpertBlobIndex.Blob],
        scratch: UnsafeMutableRawPointer,
        destination: UnsafeMutableRawPointer
    ) throws -> [CompressedBlockWork] {
        let phase0 = layout.phase1Bytes ?? layout.expertStride
        let phase1 = layout.phase1Bytes.map { layout.expertStride - $0 }
        var work: [CompressedBlockWork] = []
        work.reserveCapacity(pack.count)
        var packedOffset = 0
        for (index, blob) in pack.enumerated() {
            guard
                let target = ExpertBlobIndex.blockDestination(
                    index: index, phase0Bytes: phase0, phase1Bytes: phase1,
                    blockBytes: compressedBlockBytes)
            else {
                throw StreamerError.sizeMismatch(
                    expected: UInt64(pack.count), actual: UInt64(index))
            }
            work.append(
                CompressedBlockWork(
                    scratch: scratch,
                    packedOffset: packedOffset,
                    packedCount: Int(blob.length),
                    destination: destination,
                    destinationOffset: Int(target.offset),
                    expected: Int(target.expected),
                    phase: target.phase))
            packedOffset += Int(blob.length)
        }
        return work
    }

    /// Decodes one phase's blocks in parallel; an empty phase returns at once.
    func decodeBlocks(
        _ work: [CompressedBlockWork], phase: Int
    ) throws {
        let phaseWork = work.filter { $0.phase == phase }
        guard !phaseWork.isEmpty else { return }
        let firstError = Mutex<Error?>(nil)
        DispatchQueue.concurrentPerform(iterations: phaseWork.count) { index in
            let item = phaseWork[index]
            do {
                try ExpertBlobCodec.decompress(
                    source: item.scratch.advanced(by: item.packedOffset),
                    count: item.packedCount,
                    into: item.destination.advanced(by: item.destinationOffset),
                    expected: item.expected)
            } catch {
                firstError.withLock { if $0 == nil { $0 = error } }
            }
        }
        if let error = firstError.withLock({ $0 }) { throw error }
    }

    /// Reads ranges through the bounded reader, or in parallel on the plain fd.
    func readRanges(
        offsets: [UInt64], counts: [Int], into destinations: [UnsafeMutableRawPointer],
        reader: ParallelExpertReader? = nil
    ) throws {
        if let reader {
            try reader.fetch(offsets: offsets, byteCounts: counts, into: destinations)
            return
        }
        if let boundedReader {
            try boundedReader.fetch(
                offsets: offsets, byteCounts: counts, into: destinations)
            return
        }
        let firstError = Mutex<Error?>(nil)
        DispatchQueue.concurrentPerform(iterations: offsets.count) { index in
            do {
                try readFull(
                    into: destinations[index], fileOffset: offsets[index],
                    count: counts[index])
            } catch {
                firstError.withLock { if $0 == nil { $0 = error } }
            }
        }
        if let error = firstError.withLock({ $0 }) { throw error }
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
