import Darwin
import Foundation
import Metal
import TinyTitanFormat

// Lazy routed-expert and per-layer streamer management: opening a layer's
// backend, pinning the cache, and the counters the runner reads.
//
// Moved out of the `Model` declaration in `Model.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion;
// `openLayerLocked` stayed `private` because it moved with its callers.
extension Model {

    // MARK: - Routed expert (lazy)

    /// First touch of layer L opens its backend + verifies SHA-256; subsequent
    /// touches reuse the open backend. The backend resolves the expert to an
    /// cache-slot `(MTLBuffer, offset)` pair.
    public func routedExpert(layer L: Int, expert E: Int) throws -> TensorView {
        let backend = try openStreamer(for: L)
        // The streamer is per-layer: `openLayerLocked(L)` bound it to layer
        // L's file with `expertOffsets = layers[L].experts.map(\.offset)`, and
        // `StreamLayout.expertOffset(layer: 0, ...)` is the branch that
        // consults that per-layer offset table. Passing the actual layer here
        // would select the dense cross-layer formula and mis-offset every
        // expert on layers above 0 — layer 0 is intentional.
        let r = try backend.loadExpert(layer: 0, expert: E)
        return TensorView(
            buffer: r.buffer,
            offset: r.offset,
            length: r.size,
            scaleOffset: 0, scaleLength: 0,
            biasOffset: 0, biasLength: 0,
            shape: (UInt32(L), UInt32(E), 0, 0),
            dtype: 0)
    }

    /// Open layer L's file + verify SHA, idempotent.
    func ensureLayerOpened(_ L: Int) throws {
        try streamersQueue.sync {
            try openLayerLocked(L)
        }
    }

    /// The open streamer for a layer, after making sure the layer is open.
    ///
    /// `ensureLayerOpened` either leaves a streamer in the box or throws, so
    /// the lookup below cannot be nil in practice. It is a thrown
    /// `internalInconsistency` rather than a force unwrap so a broken
    /// invariant fails loudly without crashing the process.
    func openStreamer(for layer: Int) throws -> PreadExpertStreamer {
        try ensureLayerOpened(layer)
        guard let streamer = streamersQueue.sync(execute: { streamersBox.streamers[layer] }) else {
            throw ModelError.internalInconsistency(
                detail: "routed-expert streamer for layer \(layer) missing after ensureLayerOpened")
        }
        return streamer
    }

    /// Best-effort overlap hook for prefill: starts the same lazy layer open on
    /// the model's streamer queue without waiting for the first expert fetch.
    /// The open is retried synchronously by `ensureLayerOpened(_:)` before any
    /// expert fetch on the layer, which rethrows the identical error — so a
    /// failure here is never dropped end-to-end.
    ///
    /// `nonisolated(unsafe)` is required because `Model` is not formally
    /// `Sendable`; the capture is safe because every mutable member
    /// (`streamersBox`) is confined behind the serial `streamersQueue` and the
    /// remaining members are immutable values.
    public func beginOpeningRoutedExpertStreamer(layer L: Int) {
        nonisolated(unsafe) let model = self
        streamersQueue.async {
            do {
                try model.openLayerLocked(L)
            } catch {
                // Deferred: the synchronous `ensureLayerOpened(L)` that
                // precedes every expert fetch on this layer performs the same
                // idempotent open and rethrows this error to the prefill loop.
                // Nothing is silently lost; the async path only overlaps the
                // SHA-256 verification with the chunk's GPU work.
            }
        }
    }

    /// Drop a layer's expert cache. Safe once that layer's work is finished:
    /// the next use reopens it lazily, which is how it was created.
    public func releaseLayerStreamer(_ L: Int) {
        streamersQueue.sync {
            guard L >= 0, L < streamersBox.streamers.count else { return }
            streamersBox.streamers[L] = nil
            streamersBox.layerVerified[L] = false
        }
    }

    private func openLayerLocked(_ L: Int) throws {
        if streamersBox.streamers[L] != nil {
            return
        }
        let basename = packedExpertsLayout.layers[L].file
        let manifestRel = "packed_experts/\(basename)"
        let url =
            directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent(basename)
        let layerFD = try modelDirectory.openFile(manifestRel)
        defer { close(layerFD) }
        if !streamersBox.layerVerified[L] {
            guard let entry = manifest.files[manifestRel] else {
                throw ModelError.missingFile(name: manifestRel)
            }
            let actualSize = try modelDirectory.fileSize(
                fileDescriptor: layerFD, relativePath: manifestRel)
            guard actualSize == entry.size else {
                throw ModelError.tensorSizeMismatch(
                    name: manifestRel, expected: entry.size, actual: actualSize)
            }
            switch integrityPolicy {
            case .fullSha256:
                try Sha256Verifier.verifyFile(
                    fileDescriptor: layerFD,
                    named: manifestRel,
                    expectedHex: entry.sha256)
            case .sizeCheckTrustedReceipt:
                break
            }
            streamersBox.layerVerified[L] = true
        }
        // Checked at the one place this product is formed. `StreamLayout.
        // expertOffset` multiplies `perLayer` again on every expert read
        // (`@inline(always)`, recomputed per call, on the decode path), so the
        // product is validated here rather than there -- a wrapped one would make
        // `streamSize` small, pass the streamer's own file-size check, and leave
        // every derived offset pointing outside the layer file.
        let (streamSize, streamSizeOverflow) = UInt64(packedExpertsLayout.expertsPerLayer)
            .multipliedReportingOverflow(by: packedExpertsLayout.expertStride)
        guard !streamSizeOverflow else {
            throw ModelError.internalInconsistency(
                detail: "packed expert layer \(L) declares \(packedExpertsLayout.expertsPerLayer) "
                    + "experts of \(packedExpertsLayout.expertStride) bytes, which overflows "
                    + "the stream size")
        }
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: streamSize,
            expertsPerLayer: packedExpertsLayout.expertsPerLayer,
            expertStride: packedExpertsLayout.expertStride,
            expertOffsets: packedExpertsLayout.layers[L].experts.map(\.offset))
        let slotCount: Int
        switch streamingMode {
        case .pread(let configuredSlotCount):
            slotCount = configuredSlotCount
        }
        let effectiveSlotCount = slotCount
        let metalStagingPool: MetalExpertStagingPool?
        let metalIOService: MetalExpertIOService?
        if try ExpertIOBackend.environmentValue() == .metal {
            if streamersBox.metalStagingPool == nil {
                streamersBox.metalStagingPool = try MetalExpertStagingPool(
                    device: device,
                    byteCount: Int(packedExpertsLayout.expertStride),
                    // One staging slot per routed expert: a layer can miss all
                    // of them, and tryAcquire fails the whole request if the
                    // ring is short. This was hardcoded to 8 for the top-8
                    // families, which made the MTLIO + event path unreachable
                    // on Qwen3.8-Flash-Next -- it routes top-10, so any layer
                    // missing nine or more failed with "staging ring is
                    // unavailable" and the combination could never be measured.
                    slotCapacity: config.topKExperts)
            }
            if streamersBox.metalIOService == nil {
                streamersBox.metalIOService = try MetalExpertIOService(
                    device: device, maximumCommandsInFlight: 4)
            }
            metalStagingPool = streamersBox.metalStagingPool
            metalIOService = streamersBox.metalIOService
        } else {
            metalStagingPool = nil
            metalIOService = nil
        }
        streamersBox.streamers[L] = try PreadExpertStreamer(
            layout: layout,
            device: device,
            slotCount: effectiveSlotCount,
            cachePolicy: expertCachePolicy,
            eventCoordinator: expertIOEventCoordinator,
            metalStagingPool: metalStagingPool,
            metalIOService: metalIOService)
        // A newly opened layer is not wired yet; the next pin walks again.
        // When this run holds the cache wired (`profile.keepExpertCacheWired`,
        // set through `setKeepExpertCacheWired`) it is wired here, so a cache
        // that is never unpinned is never swapped out and the first decode
        // token does not pay to fault it back (measured 1.6-4.7 s per request
        // on Qwen3.8).
        streamersBox.pinnedComplete = false
        if streamersBox.keepWired {
            streamersBox.streamers[L]?.setSlotsPinned(true)
        }
    }

    /// Test hook: how many layer files have been opened so far.
    public func openLayerFileCount() -> Int {
        streamersQueue.sync { streamersBox.streamers.compactMap { $0 }.count }
    }

    /// Wire the routed-expert slot cache for decode, or release it for
    /// prefill.
    ///
    /// Decode is the phase where a reclaimed slot page costs an SSD read on
    /// the critical path, so that is the phase worth wiring. Prefill streams
    /// experts in bulk and instead needs the headroom -- holding the cache
    /// wired throughout measurably slowed ANE prefill, which has to place
    /// Core ML arenas alongside it. Called at the phase boundaries; cheap and
    /// idempotent, since each streamer skips a state it is already in.
    /// Wire layers as they open from now on (see `ModelProfile.keepExpertCacheWired`).
    public func setKeepExpertCacheWired(_ keep: Bool) {
        streamersQueue.sync { streamersBox.keepWired = keep }
    }

    public var expertCachePinQueueWaitNanos: UInt64 {
        streamersQueue.sync { streamersBox.pinQueueWaitNanos }
    }

    /// Returns true when every opened streamer is in the requested state.
    @discardableResult
    public func setExpertCachePinned(_ pinned: Bool) -> Bool {
        // Every decode step asks for the cache to be wired. Once every
        // streamer reports it is, there is nothing to do, and walking all of
        // them under the serial queue per token measured 73-91 ms on a
        // 48-layer model (TINYTITAN_RUNNER_STATS pre_pin_ms). The walk only runs
        // again after an unpin or a partial wire.
        let tEnter = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var walked = 0
        var unpinnedAfter = 0
        var earlyReturn = false
        var walkNanos: UInt64 = 0
        streamersQueue.sync {
            streamersBox.pinQueueWaitNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tEnter
            if pinned, streamersBox.pinnedComplete {
                earlyReturn = true
                return
            }
            let tWalk = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            var complete = pinned
            for streamer in streamersBox.streamers {
                guard let streamer else { continue }
                walked += 1
                streamer.setSlotsPinned(pinned)
                if pinned, !streamer.isPinned {
                    complete = false
                    unpinnedAfter += 1
                }
            }
            walkNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWalk
            streamersBox.pinnedComplete = complete
        }
        let total = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tEnter
        if PreadExpertStreamer.wireTraceEnabled, total > 2_000_000 {
            FileHandle.standardError.write(
                Data(
                    "[wire] setExpertCachePinned(\(pinned)) \(Double(total) / 1e6) ms early=\(earlyReturn) walked=\(walked) walk_ms=\(Double(walkNanos) / 1e6) unpinned_after=\(unpinnedAfter)\n"
                        .utf8))
        }
        return streamersQueue.sync { streamersBox.pinnedComplete } == pinned
    }
}
