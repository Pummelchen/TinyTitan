import Foundation
import Metal

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
        // which is where the cost has to be if it is to be free.
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
                    for (expert, destination) in zip(
                        experts, safeDestinations.values)
                    {
                        try decodeCompressedExpert(
                            frames: frames[expert], into: destination)
                    }
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

}
