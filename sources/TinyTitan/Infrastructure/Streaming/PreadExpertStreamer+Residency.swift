import Darwin
import Foundation
import Synchronization
import Metal

// Residency publication, statistics, eviction and prefetch.
//
// Split out of `PreadExpertStreamer.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
// Members whose callers stay behind widened from `private` to internal.
extension PreadExpertStreamer {

    public func expertCachePlanBuffers(_ plan: ExpertCachePlan)
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)]
    {
        precondition(
            plan.assignedSlots.count == plan.experts.count,
            "expert cache plan slot count mismatch")
        return plan.assignedSlots.map { slot in
            (slotBuffers[slot], slotBufferOffsets[slot], layout.expertStride)
        }
    }

    public func expertResidencyResources() -> ExpertResidencyResources {
        ExpertResidencyResources(
            table: residencyTable,
            poolSlotStride: UInt64(poolSlotStride),
            expertStride: layout.expertStride,
            expertCount: layout.expertsPerLayer)
    }

    public func residencyEntry(expert: Int) -> ExpertResidencyEntry {
        precondition(expert >= 0 && expert < layout.expertsPerLayer)
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return residencyTable.contents()
            .bindMemory(
                to: ExpertResidencyEntry.self,
                capacity: layout.expertsPerLayer)[expert]
    }

    public func adviseExpertCachePlanMisses(_ plan: ExpertCachePlan) -> ExpertIOAdviceResult {
        let experts = plan.misses.map { plan.experts[$0] }
        return adviseRanges(
            expertAdviceRanges(experts: experts, layer: plan.layer),
            requested: experts.count)
    }

    public func adviseExperts(experts: [Int]) -> ExpertIOAdviceResult {
        adviseRanges(expertAdviceRanges(experts: experts, layer: 0), requested: experts.count)
    }

    public func adviseExpertMisses(experts: [Int]) -> ExpertIOAdviceResult {
        cacheLock.lock()
        let misses = experts.filter { expert in
            !slotExpert.indices.contains { slot in
                slotState[slot] == .resident && slotExpert[slot] == expert
            }
        }
        cacheLock.unlock()
        return adviseRanges(expertAdviceRanges(experts: misses, layer: 0), requested: misses.count)
    }

    static func coalescedAdjacentAdviceRanges(_ ranges: [(offset: UInt64, count: UInt64)])
        -> [(offset: UInt64, count: UInt64)]
    {
        let sorted = ranges.filter { $0.count > 0 }.sorted {
            $0.offset == $1.offset ? $0.count < $1.count : $0.offset < $1.offset
        }
        var result: [(offset: UInt64, count: UInt64)] = []
        for range in sorted {
            guard var last = result.popLast() else {
                result.append(range)
                continue
            }
            // K28: checked arithmetic — a wrapping `&+` could merge two
            // huge ranges into a nonsense span. On overflow keep the ranges
            // separate (the merge is an optimization, never a correctness
            // requirement).
            let (lastEnd, lastOverflow) = last.offset.addingReportingOverflow(last.count)
            let (rangeEnd, rangeOverflow) = range.offset.addingReportingOverflow(range.count)
            if lastOverflow || rangeOverflow {
                result.append(last)
                result.append(range)
                continue
            }
            if range.offset <= lastEnd {
                last.count = max(lastEnd, rangeEnd) - last.offset
                result.append(last)
            } else {
                result.append(last)
                result.append(range)
            }
        }
        return result
    }

    /// Zero the LFU use counts, keeping the slots and their contents. Called
    /// at the prefill-to-decode transition: a prefill chunk plans every
    /// expert it touches hundreds of times, so the leftovers outrank any
    /// expert decode has used once or twice and decode cannot evict them.
    /// With the counts zeroed, ties fall to LRU order and decode's own
    /// working set takes the slots within a few tokens.
    public func resetExpertUseCounts() {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        for i in expertUseCount.indices { expertUseCount[i] = 0 }
    }

    func shouldEvictSlot(_ lhs: Int, before rhs: Int) -> Bool {
        if cachePolicy == .lru {
            return slotLastUse[lhs] < slotLastUse[rhs]
        }
        let lhsExpert = slotExpert[lhs]
        let rhsExpert = slotExpert[rhs]
        if lhsExpert < 0 || rhsExpert < 0 {
            return lhsExpert < rhsExpert
        }
        let lhsCount = lhsExpert < expertUseCount.count ? expertUseCount[lhsExpert] : 0
        let rhsCount = rhsExpert < expertUseCount.count ? expertUseCount[rhsExpert] : 0
        if lhsCount != rhsCount { return lhsCount < rhsCount }
        return slotLastUse[lhs] < slotLastUse[rhs]
    }

    func pin(_ plan: ExpertCachePlan) throws -> ExpertCacheLease {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        guard plan.assignedSlots.count == plan.experts.count,
            plan.assignedGenerations.count == plan.experts.count
        else {
            throw ModelError.internalInconsistency(
                detail: "cannot pin an incomplete expert-cache plan")
        }
        for index in plan.experts.indices {
            let slot = plan.assignedSlots[index]
            guard slot >= 0, slot < slotCount,
                slotGeneration[slot] == plan.assignedGenerations[index],
                slotExpert[slot] == plan.experts[index],
                slotState[slot] != .empty
            else {
                throw ModelError.internalInconsistency(
                    detail: "expert-cache plan became stale before GPU pin")
            }
        }
        for slot in plan.assignedSlots { slotPinCount[slot] &+= 1 }
        return ExpertCacheLease(
            streamer: self,
            slots: plan.assignedSlots,
            generations: plan.assignedGenerations)
    }

    func unpin(slots: [Int], generations: [UInt64]) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        for (slot, generation) in zip(slots, generations)
        where slot >= 0 && slot < slotCount && slotGeneration[slot] == generation {
            precondition(slotPinCount[slot] > 0, "expert-cache slot pin underflow")
            slotPinCount[slot] -= 1
        }
    }

    public func statistics() -> ExpertStreamingStatistics {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return ExpertStreamingStatistics(
            plans: statisticsPlans,
            requestedExperts: statisticsRequestedExperts,
            hits: statisticsHits,
            misses: statisticsMisses,
            bytesRead: statisticsBytesRead,
            readOperations: statisticsReadOperations,
            evictions: statisticsEvictions,
            reloads: statisticsReloads,
            loadBatches: statisticsLoadBatches,
            totalLoadNanos: statisticsTotalLoadNanos,
            maximumLoadNanos: statisticsMaximumLoadNanos,
            latencyHistogram: statisticsLatencyHistogram,
            residentSlots: slotState.count(where: { $0 == .resident }),
            loadingSlots: slotState.count(where: { $0 == .loading }),
            pinnedSlots: slotPinCount.count(where: { $0 > 0 }),
            peakLoadingSlots: statisticsPeakLoadingSlots)
    }

    /// A stable snapshot of authoritative cache entries. Loading slots are
    /// intentionally omitted: their bytes must not be consumed or treated as
    /// available by a predictor until a successful demand load publishes them.
    /// Diagnostic and policy code use this before a cache plan reserves slots.
    public func residentExperts() -> [Int] {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return zip(slotExpert, slotState).compactMap { expert, state in
            state == .resident && expert >= 0 ? expert : nil
        }.sorted()
    }

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

    func finishPlanExecution(
        _ plan: ExpertCachePlan,
        succeeded: Bool,
        elapsedNanos: UInt64
    ) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if succeeded {
            recordSuccessfulLoadsUnlocked(
                experts: plan.misses.map { plan.experts[$0] },
                elapsedNanos: elapsedNanos)
            return
        }
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            if slotGeneration[slot] == plan.assignedGenerations[index],
                slotState[slot] == .loading
            {
                slotState[slot] = .empty
                slotExpert[slot] = -1
                publishResidencyUnlocked(
                    expert: plan.experts[index],
                    slot: slot,
                    state: ExpertResidencyEntry.empty,
                    generation: plan.assignedGenerations[index])
            }
        }
    }

    func markPlanMissesResident(_ plan: ExpertCachePlan) throws {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            guard slotGeneration[slot] == plan.assignedGenerations[index] else {
                throw ModelError.internalInconsistency(
                    detail: "expert-cache slot generation changed during expert load")
            }
        }
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            slotState[slot] = .resident
            slotExpert[slot] = plan.experts[index]
            slotLastUse[slot] = useClock
            publishResidencyUnlocked(
                expert: plan.experts[index],
                slot: slot,
                state: ExpertResidencyEntry.resident,
                generation: plan.assignedGenerations[index])
        }
    }

    /// The Metal staging route copies into cache slots on the GPU after the
    /// MTLIO event. Its slots cannot become resident until that command buffer
    /// has completed, otherwise a later layer could read bytes still owned by
    /// the blit engine.
    func markStagedMetalPlanResident(_ plan: ExpertCachePlan) throws {
        try markPlanMissesResident(plan)
    }

    /// Clears a staged load if its event-gated transfer command fails. This is
    /// intentionally separate from `finishPlanExecution`: I/O may have
    /// succeeded and been accounted for, while the GPU copy did not complete.
    func failStagedMetalPlan(_ plan: ExpertCachePlan) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            guard slot >= 0, slot < slotCount,
                slotGeneration[slot] == plan.assignedGenerations[index],
                slotState[slot] == .loading
            else { continue }
            slotState[slot] = .empty
            slotExpert[slot] = -1
            publishResidencyUnlocked(
                expert: plan.experts[index],
                slot: slot,
                state: ExpertResidencyEntry.empty,
                generation: plan.assignedGenerations[index])
        }
    }

    /// CPU publication occurs under the cache lock. A loading entry is visible
    /// immediately after reservation; resident is written only after every
    /// byte lands. Event-driven consumers additionally wait on the batch's
    /// shared-event value, which is the CPU/GPU release/acquire boundary.
    func publishResidencyUnlocked(
        expert: Int,
        slot: Int,
        state: UInt32,
        generation: UInt64
    ) {
        guard expert >= 0 && expert < layout.expertsPerLayer else { return }
        let entries = residencyTable.contents()
            .bindMemory(
                to: ExpertResidencyEntry.self,
                capacity: layout.expertsPerLayer)
        entries[expert] = ExpertResidencyEntry(
            slot: state == ExpertResidencyEntry.empty
                ? ExpertResidencyEntry.notResidentSlot : UInt32(slot),
            state: state,
            generation: generation)
    }

    func recordSuccessfulLoadsUnlocked(experts: [Int], elapsedNanos: UInt64) {
        guard !experts.isEmpty else { return }
        statisticsBytesRead &+= UInt64(experts.count) * layout.expertStride
        statisticsReadOperations &+= UInt64(experts.count)
        statisticsLoadBatches &+= 1
        statisticsTotalLoadNanos &+= elapsedNanos
        statisticsMaximumLoadNanos = max(statisticsMaximumLoadNanos, elapsedNanos)
        let bucket = Self.latencyBucketIndex(nanos: elapsedNanos)
        statisticsLatencyHistogram[bucket] &+= 1
        for expert in experts where expert >= 0 && expert < expertLoadCount.count {
            if expertLoadCount[expert] > 0 { statisticsReloads &+= 1 }
            expertLoadCount[expert] &+= 1
        }
    }

    func recordPrefetchAdoptionsUnlocked(_ experts: [Int]) {
        for expert in experts where expert >= 0 && expert < expertLoadCount.count {
            if expertLoadCount[expert] > 0 { statisticsReloads &+= 1 }
            expertLoadCount[expert] &+= 1
        }
    }

    static func latencyBucketIndex(nanos: UInt64) -> Int {
        var bound: UInt64 = 125_000
        for index in 0..<16 {
            if nanos <= bound { return index }
            bound &*= 2
        }
        return 16
    }

    static func latencyBucketUpperBound(index: Int) -> UInt64 {
        guard index < 16 else { return UInt64.max }
        return 125_000 << UInt64(index)
    }

    func expertAdviceRanges(
        experts: [Int],
        layer: Int
    ) -> [(offset: UInt64, count: UInt64)] {
        experts.compactMap { expert in
            let regionOffset = layout.expertOffset(layer: layer, expert: expert)
            let (regionEnd, regionOverflow) =
                regionOffset
                .addingReportingOverflow(layout.expertStride)
            guard !regionOverflow, regionEnd <= layout.streamSize else { return nil }
            return (layout.streamOffset + regionOffset, layout.expertStride)
        }
    }

    func adviseRanges(
        _ ranges: [(offset: UInt64, count: UInt64)],
        requested: Int
    ) -> ExpertIOAdviceResult {
        let coalesced = Self.coalescedAdjacentAdviceRanges(ranges)
        var failed = 0
        var bytes: UInt64 = 0
        var maxCallNanos: UInt64 = 0
        for range in coalesced {
            let result = RDAdvice.call(fd: fd, offset: range.offset, byteCount: range.count)
            if !result.succeeded { failed += 1 }
            bytes &+= result.requestedBytes
            maxCallNanos = max(maxCallNanos, result.elapsedNanos)
        }
        return ExpertIOAdviceResult(
            requested: requested,
            failed: failed,
            calls: coalesced.count,
            bytes: bytes,
            maxCallNanos: maxCallNanos)
    }

    func readFull(
        into destination: UnsafeMutableRawPointer,
        fileOffset: UInt64,
        count: Int
    ) throws {
        var filled = 0
        while filled < count {
            let readCount = pread(
                fd,
                destination.advanced(by: filled),
                count - filled,
                off_t(fileOffset) + off_t(filled))
            if readCount < 0 {
                // A signal interrupted the read: nothing was transferred, and
                // the call has to be retried. Every other read loop in this
                // module and in the model-IO layer does that; treating it as a
                // failure turned a delivered signal into a streamer error, and
                // the callers cannot tell the two apart.
                if errno == EINTR { continue }
                throw StreamerError.preadFailed(errno: errno)
            }
            if readCount == 0 {
                throw StreamerError.sizeMismatch(expected: UInt64(count), actual: UInt64(filled))
            }
            filled += readCount
        }
    }
}
