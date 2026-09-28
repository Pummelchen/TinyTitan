import Darwin
import Foundation
import Synchronization
import Metal

// The Metal, bounded and cached read paths.
//
// Split out of `PreadExpertStreamer.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
// Members whose callers stay behind widened from `private` to internal.
extension PreadExpertStreamer {

    func beginEventDrivenMetalReads(
        _ plan: ExpertCachePlan,
        reader: MetalExpertReader,
        token: ExpertIOCompletionToken?
    ) throws -> ExpertLoadOperation {
        guard let token,
            let stagingLease = metalStagingPool?.tryAcquire(count: plan.misses.count)
        else {
            throw ModelError.internalInconsistency(
                detail: "event-driven Metal I/O staging ring is unavailable")
        }
        let transfer = try makeMetalStagingTransfer(plan: plan, stagingLease: stagingLease)
        let operation = ExpertLoadOperation(
            completionToken: token,
            eventCoordinator: eventCoordinator,
            // MTLIO writes the status word and signals the event in command
            // order. Its handler records terminal state but never wakes the
            // decode task to encode a fixup.
            backendSignalsEvent: true,
            metalStagingTransfer: transfer,
            requiresGPUFinalization: true)
        operation.markInFlight()
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        do {
            try beginMetalReads(
                plan,
                reader: reader,
                destinations: stagingLease.buffers,
                destinationOffsets: [Int](repeating: 0, count: stagingLease.buffers.count),
                completionToken: token
            ) { [self, operation] result in
                switch result {
                case .success:
                    // Cache slots remain LOADING. The runner publishes
                    // RESIDENT only after its event-gated blit completes.
                    finishPlanExecution(
                        plan,
                        succeeded: true,
                        elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
                    operation.finish(.success(()))
                case .failure(let error):
                    finishPlanExecution(
                        plan,
                        succeeded: false,
                        elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
                    operation.finish(.failure(error))
                }
            }
        } catch {
            finishPlanExecution(plan, succeeded: false, elapsedNanos: 0)
            operation.releaseStagingTransfer()
            operation.finish(.failure(error))
        }
        return operation
    }

    func executeMetalReads(
        _ plan: ExpertCachePlan,
        reader: MetalExpertReader
    ) throws {
        var offsets: [UInt64] = []
        var destinations: [MTLBuffer] = []
        offsets.reserveCapacity(plan.misses.count)
        destinations.reserveCapacity(plan.misses.count)
        for index in plan.misses {
            offsets.append(try fileOffset(plan: plan, index: index))
            destinations.append(slotBuffers[plan.assignedSlots[index]])
        }
        try reader.fetch(
            offsets: offsets,
            into: destinations,
            byteCount: Int(layout.expertStride),
            destinationOffsets: plan.misses.map {
                Int(slotBufferOffsets[plan.assignedSlots[$0]])
            })
    }

    func beginMetalReads(
        _ plan: ExpertCachePlan,
        reader: MetalExpertReader,
        destinations explicitDestinations: [MTLBuffer]? = nil,
        destinationOffsets explicitDestinationOffsets: [Int]? = nil,
        completionToken: ExpertIOCompletionToken?,
        completion: @escaping @Sendable (Result<Void, any Error>) -> Void
    ) throws {
        var offsets: [UInt64] = []
        var destinations: [MTLBuffer] = []
        offsets.reserveCapacity(plan.misses.count)
        destinations.reserveCapacity(plan.misses.count)
        for index in plan.misses {
            offsets.append(try fileOffset(plan: plan, index: index))
            destinations.append(slotBuffers[plan.assignedSlots[index]])
        }
        let finalDestinations = explicitDestinations ?? destinations
        let finalDestinationOffsets =
            explicitDestinationOffsets
            ?? plan.misses.map {
                Int(slotBufferOffsets[plan.assignedSlots[$0]])
            }
        try reader.beginFetch(
            offsets: offsets,
            into: finalDestinations,
            byteCount: Int(layout.expertStride),
            destinationOffsets: finalDestinationOffsets,
            completionToken: completionToken,
            completion: completion)
    }

    func makeMetalStagingTransfer(
        plan: ExpertCachePlan,
        stagingLease: MetalExpertStagingLease
    ) throws -> MetalExpertStagingTransfer {
        var destinations: [MTLBuffer] = []
        var destinationOffsets: [Int] = []
        destinations.reserveCapacity(plan.misses.count)
        destinationOffsets.reserveCapacity(plan.misses.count)
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            guard slot >= 0, slot < slotBuffers.count else {
                stagingLease.release()
                throw ModelError.internalInconsistency(
                    detail: "Metal I/O staging transfer references an invalid cache slot")
            }
            destinations.append(slotBuffers[slot])
            destinationOffsets.append(Int(slotBufferOffsets[slot]))
        }
        return MetalExpertStagingTransfer(
            lease: stagingLease,
            destinations: destinations,
            destinationOffsets: destinationOffsets,
            byteCount: Int(layout.expertStride))
    }

    func executeBoundedReads(
        _ plan: ExpertCachePlan,
        reader: ParallelExpertReader
    ) throws {
        var offsets: [UInt64] = []
        var destinations: [UnsafeMutableRawPointer] = []
        offsets.reserveCapacity(plan.misses.count)
        destinations.reserveCapacity(plan.misses.count)
        for index in plan.misses {
            offsets.append(try fileOffset(plan: plan, index: index))
            destinations.append(slotPointers[plan.assignedSlots[index]])
        }
        try reader.fetch(offsets: offsets, into: destinations)
    }

    func executeCachedPreads(
        _ plan: ExpertCachePlan,
        parallel: Bool
    ) throws {
        if parallel {
            let firstError = Mutex<Error?>(nil)
            DispatchQueue.concurrentPerform(iterations: plan.misses.count) { offset in
                do {
                    try readPlanMiss(plan, index: plan.misses[offset])
                } catch {
                    firstError.withLock { if $0 == nil { $0 = error } }
                }
            }
            if let error = firstError.withLock({ $0 }) { throw error }
            return
        }
        for index in plan.misses { try readPlanMiss(plan, index: index) }
    }

    func readPlanMiss(_ plan: ExpertCachePlan, index: Int) throws {
        try readFull(
            into: slotPointers[plan.assignedSlots[index]],
            fileOffset: try fileOffset(plan: plan, index: index),
            count: Int(layout.expertStride))
    }

    func fileOffset(plan: ExpertCachePlan, index: Int) throws -> UInt64 {
        let regionOffset = layout.expertOffset(
            layer: plan.layer,
            expert: plan.experts[index])
        // Checked for the same reason: a wrapped sum here would read as "in
        // range" and the pread below would take an offset past the end of the
        // mapped file. With this and the open-time check, every sum in this type
        // that mixes a layout offset with a region is known not to wrap.
        let (regionEnd, regionOverflow) =
            regionOffset
            .addingReportingOverflow(layout.expertStride)
        guard !regionOverflow, regionEnd <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        return layout.streamOffset + regionOffset
    }

}
