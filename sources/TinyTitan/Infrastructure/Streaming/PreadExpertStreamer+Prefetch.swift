import Foundation
import TinyTitanFormat
import Metal
import Synchronization

// Speculative expert reads: the ring, and the compressed arm that decodes into
// its destinations.
//
// Split out of `PreadExpertStreamer+Residency.swift` under the 500-line-per-file
// rule when compressed prefetch landed. Pure code motion plus the compressed
// branch, which decodes into the ring buffer so adoption stays the `memcpy` the
// raw path uses.
extension PreadExpertStreamer {
    /// Starts a bounded, raw speculative read. The destination buffers are not
    /// cache slots, so an incorrect prediction cannot evict an authoritative
    /// expert. Demand work is always scheduled at higher priority.
    public func beginPrefetch(
        experts: [Int],
        destinations: [UnsafeMutableRawPointer]
    ) throws
        -> ExpertLoadOperation
    {
        guard experts.count == destinations.count else {
            throw ModelError.internalInconsistency(
                detail: "prefetch experts and destinations differ in count")
        }
        // A compressed layer prefetches by decoding into the destination
        // buffers, so adoption stays the same memcpy the raw path uses. The
        // decode happens on this scheduler thread, off the decode critical path,
        // which is where the cost has to be if it is to be free -- and it runs
        // in parallel across the predicted experts. Serially, one top-k set is
        // ~15 MB at ~0.9 GB/s = ~17 ms, longer than the ~6 ms a layer takes, so
        // the ring never finished in time and adoption measured 0.06 per token
        // against the raw path's 21.75.
        if let frames = compressedFrames {
            let operation = ExpertLoadOperation()
            // The same wrapper the raw path uses: raw pointers are not `Sendable`
            // and the scheduler's closure is.
            let safeDestinations = PrefetchDestinations(destinations)
            ExpertIOScheduler.shared.submit(
                priority: .speculative
            ) { [self, operation, safeDestinations] in
                operation.markInFlight()
                do {
                    try decodePrefetchedExperts(
                        experts, frames: frames, into: safeDestinations.values)
                    operation.finish(.success(()))
                } catch {
                    operation.finish(.failure(error))
                }
            }
            return operation
        }
        let offsets = try experts.map { expert -> UInt64 in
            guard expert >= 0 && expert < layout.expertsPerLayer else {
                throw ModelError.internalInconsistency(detail: "invalid prefetched expert")
            }
            return layout.streamOffset + layout.expertOffset(layer: 0, expert: expert)
        }
        let safeDestinations = PrefetchDestinations(destinations)
        let operation = ExpertLoadOperation()
        ExpertIOScheduler.shared.submit(priority: .speculative) {
            [self, operation, safeDestinations] in
            operation.markInFlight()
            do {
                if let boundedReader {
                    try boundedReader.fetch(offsets: offsets, into: safeDestinations.values)
                } else {
                    for (offset, destination) in zip(offsets, safeDestinations.values) {
                        try readFull(
                            into: destination, fileOffset: offset,
                            count: Int(layout.expertStride))
                    }
                }
                operation.finish(.success(()))
            } catch {
                operation.finish(.failure(error))
            }
        }
        return operation
    }

    /// Decode a predicted expert set into the ring's buffers, in parallel.
    ///
    /// The ring is the only place where more than one expert's decode has to
    /// finish by a deadline (the next layer), so it is the place where one core
    /// is not enough.
    private func decodePrefetchedExperts(
        _ experts: [Int],
        frames: [[ExpertBlobIndex.Blob]],
        into destinations: [UnsafeMutableRawPointer]
    ) throws {
        if experts.count <= 1 {
            if let expert = experts.first, let destination = destinations.first {
                try decodeCompressedExpert(frames: frames[expert], into: destination)
            }
            return
        }
        let firstError = Mutex<Error?>(nil)
        DispatchQueue.concurrentPerform(iterations: experts.count) { index in
            do {
                try decodeCompressedExpert(
                    frames: frames[experts[index]], into: destinations[index])
            } catch {
                firstError.withLock { if $0 == nil { $0 = error } }
            }
        }
        if let error = firstError.withLock({ $0 }) { throw error }
    }

}
