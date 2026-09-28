import Foundation
import Metal

/// Decode-time routed MoE: the shared expert, the hit/fixup expert dispatch,
/// and the slot bookkeeping the stage publishes.
///
/// Split out of `RealForwardRunner+Decode.swift` (2026-09-28) under the
/// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion:
/// the same bodies, moved so the token layer loop and the expert stage read
/// apart. No signature or behaviour change; every declaration they touch was
/// already internal (`private` is file-scoped in Swift, so a cross-file reach
/// would have forced a widening here -- none was needed).
extension RealForwardRunner {
    func finishPendingRoutedCommand(
        _ pending: PendingRoutedCommand,
        waitIfNeeded: Bool
    ) throws {
        defer { pending.expertLease?.release() }
        // A staged Metal-I/O batch owns its source buffers until the compute
        // command has completed. If any command/error path exits early, leave
        // the cache entries empty rather than retaining a LOADING slot.
        var finalizedStagingTransfer = false
        defer {
            if let operation = pending.storageOperation {
                if operation.storage.requiresGPUFinalization,
                    !finalizedStagingTransfer
                {
                    model.failRoutedExpertStagingTransfer(plan: operation.plan)
                }
                operation.storage.releaseStagingTransfer()
            }
        }
        if waitIfNeeded {
            if let sharedCB = pending.sharedCB {
                try waitForCompletion(sharedCB)
            }
            if let phase1HitCB = pending.phase1HitCB {
                try waitForCompletion(phase1HitCB)
            }
            try waitForCompletion(pending.cb)
        } else if let err = pending.cb.error {
            throw ModelError.commandBufferFailed(
                detail: "routed layer command buffer: \(err)")
        }
        if let operation = pending.storageOperation {
            // Event-gated commands cannot complete before this operation is
            // terminal, so this is an error check, not a successful-I/O host
            // wait. A failed read is surfaced after safe no-op kernels have
            // prevented incomplete slot bytes from being dereferenced.
            try operation.storage.wait()
            if operation.storage.requiresGPUFinalization {
                try model.finalizeRoutedExpertStagingTransfer(plan: operation.plan)
                finalizedStagingTransfer = true
            }
            totalIOQueueNanos &+= operation.storage.submissionToStartNanos
            totalIoNanos &+= operation.storage.loadNanos
            totalMissIoNanos &+= operation.storage.loadNanos
            if let latest = pending.overlapCompletionClock?.latest(
                expected: pending.expectedOverlapCompletions)
            {
                let completed = operation.storage.completedNanos
                if completed > latest {
                    totalExposedIoNanos &+= completed - latest
                }
            }
        }
        if let sharedCB = pending.sharedCB, let err = sharedCB.error {
            throw ModelError.commandBufferFailed(
                detail: "shared-expert command buffer: \(err)")
        }
        if let phase1HitCB = pending.phase1HitCB, let err = phase1HitCB.error {
            throw ModelError.commandBufferFailed(
                detail: "routed phase-1 hit command buffer: \(err)")
        }
        if let sharedCB = pending.sharedCB {
            recordKernelGPU(role: "shared_expert", sharedCB)
        }
        if let phase1HitCB = pending.phase1HitCB {
            recordKernelGPU(role: "moe_phase1_hit", phase1HitCB)
        }
        recordKernelGPU(role: pending.kernelRole, pending.cb)
        totalCb2Nanos &+= pending.encodeAndCommitNanos
    }

    func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
        let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<slots.count { ptr[i] = slots[i] }
    }

    /// Encodes the shared dense MLP and commits it immediately.
    ///
    /// It depends only on `routedX`, which `tailCB` produces, so it can be
    /// queued the moment `tailCB` is committed -- before the router readback,
    /// not after it. Both sit on the same queue, so the GPU runs this while the
    /// CPU is blocked waiting for `tailCB` to report the routing.
    ///
    /// That ordering is the whole point. Encoding it after the readback left a
    /// measured 7.88 ms/token of GPU idle in the
    /// `attn_tail_router -> shared_expert` transition -- 0.197 ms per layer of
    /// command-buffer round trip during which the GPU had nothing queued, and
    /// the largest single component of decode's idle time.
    func encodeAndCommitSharedExpert(
        layer L: Int,
        completionClock: CommandCompletionClock?
    ) throws -> MTLCommandBuffer {
        let sharedProj = sharedExpertProjections[L]
        let D = UInt32(cfg.hiddenSize)
        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try shared.encode(
            commandBuffer: sharedCB,
            x: routedX,
            gate: sharedProj.gate,
            up: sharedProj.up,
            down: sharedProj.down,
            y: h1Buf,
            scratchGate: denseScratchGate,
            scratchUp: denseScratchUp,
            scratchAct: denseScratchAct)
        if cfg.sharedExpertGated {
            // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX)
            let gateView = try requireTensorView(sharedProj.scalarGate, "shared-expert scalar gate")
            try encodeScalarGate(
                commandBuffer: sharedCB,
                view: gateView,
                x: routedX,
                y: try requireBuffer(sharedScalarGateBuf, "shared-expert gate buffer"),
                n: D)
            try requireElementwise().encodeSigmoidScalarMul(
                commandBuffer: sharedCB,
                y: h1Buf,
                gate: try requireBuffer(sharedScalarGateBuf, "shared-expert gate buffer"),
                count: cfg.hiddenSize)
        }
        completionClock?.track(sharedCB)
        sharedCB.commit()
        return sharedCB
    }

    /// Routed-expert stage of one decode layer: top-k readback, expert fetch,
    /// phase-1/phase-2 encode, and the deferred completion hand-off.
    ///
    /// lint:allow-long one pipeline whose phases share the fetch plan, the
    /// argument buffer and the slot scratch; the layer trace at the end reports
    /// timings from every phase, so splitting it would mean threading those
    /// back out purely to shorten a function.
    func encodeDecodeRoutedMoE(
        layer L: Int,
        position: Int,
        sharedProj: LayerSharedExpertProjections,
        attnCB: MTLCommandBuffer,
        tailCB: MTLCommandBuffer,
        sharedCB: MTLCommandBuffer,
        overlapCompletionClock: CommandCompletionClock?,
        pending pendingRoutedCommand: inout PendingRoutedCommand?,
        bodyStart tBodyStart: UInt64,
        cb1Start tCb1Start: UInt64,
        waitMark tWait: UInt64,
        waitNanos: UInt64,
        previousRoutedMicros prevRoutedUs: Double,
        predictedNextLayer: [Int],
        predictedNextLayerWeights: [Float] = []
    ) async throws {
        let D = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let readbackStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let idxPtr = outIndices.contents().bindMemory(
            to: UInt32.self,
            capacity: cfg.topKExperts)
        decodeExpertsScratch.removeAll(keepingCapacity: true)
        decodeExpertsScratch.reserveCapacity(cfg.topKExperts)
        for i in 0..<cfg.topKExperts {
            decodeExpertsScratch.append(min(Int(idxPtr[i]), cfg.numExperts - 1))
        }
        totalRouterReadbackNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - readbackStarted
        let experts = decodeExpertsScratch
        recordRouteTrace(layer: L, position: position, experts: experts)

        let routedOffsets = try model.routedExpertOffsets(layer: L)
        let topK = UInt32(cfg.topKExperts)
        let canUsePlannedFetch = cfg.topKExperts <= moe.maxStreamedExperts
        let residentBeforePlan =
            prefetchTraceFD >= 0
            ? try model.routedExpertResidentIDs(layer: L) : []
        let readyPrefetches = predictivePrefetch?.readyBuffers(layer: L, experts: experts) ?? [:]
        let cachePlanStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let plannedFetch =
            canUsePlannedFetch
            ? try model.planRoutedExperts(
                layer: L, experts: experts, prefetched: readyPrefetches)
            : nil
        totalCachePlanNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cachePlanStarted
        if !readyPrefetches.isEmpty {
            predictivePrefetch?.consume(layer: L, experts: Set(readyPrefetches.keys))
            totalPrefetchAdopted &+= UInt64(readyPrefetches.count)
        }
        let missesForTrace =
            plannedFetch.map { plan in
                plan.misses.map { experts[$0] }
            } ?? experts
        recordPrefetchTrace(
            layer: L, position: position, experts: experts,
            misses: missesForTrace, resident: residentBeforePlan,
            nextLayerPrediction: predictedNextLayer,
            next2LayerPrediction: lastPredictedNext2Layer,
            nextLayerWeights: predictedNextLayerWeights)
        let expertLease = try plannedFetch.map { try model.pinRoutedExperts(for: $0) }
        // v4.2 Phase B: once slots and generations are reserved and pinned,
        // submit real storage immediately. Hit partitioning, argument binding,
        // and command encoding below now overlap the reader queue.
        let shouldSubmitImmediately =
            expertIOSubmission == .immediate
            || expertIOSynchronization == .event
        let plannedLoad =
            shouldSubmitImmediately
            ? try plannedFetch.map {
                try model.beginFetchRoutedExperts(
                    plan: $0,
                    eventDriven: expertIOSynchronization == .event && !$0.misses.isEmpty)
            }
            : nil
        var transferredExpertLease = false
        var phase1HitCB: MTLCommandBuffer?
        defer {
            if !transferredExpertLease {
                // A thrown fetch/encode must not make a hit slot evictable
                // while its already-committed phase-1 command is still reading.
                if let phase1HitCB, let expertLease {
                    try? waitForCompletion(phase1HitCB)
                    expertLease.release()
                } else {
                    expertLease?.release()
                }
            }
        }
        var phase1HitSplitArgBuf: MTLBuffer?
        decodeHitSplitRoutedBufsScratch.removeAll(keepingCapacity: true)
        decodeHitSplitRoutedOffsetsScratch.removeAll(keepingCapacity: true)
        decodeHitSlotsScratch.removeAll(keepingCapacity: true)
        decodeMissSlotsScratch.removeAll(keepingCapacity: true)

        if let plan = plannedFetch,
            decodeExpertExecution == .hitFixup
                || decodeExpertExecution == .gpuResidency
        {
            if decodeExpertExecution == .gpuResidency {
                let hitCount = min(
                    Int(residencyHitCount.contents().load(as: UInt32.self)),
                    cfg.topKExperts)
                let missCount = min(
                    Int(residencyMissCount.contents().load(as: UInt32.self)),
                    cfg.topKExperts)
                let hitPointer = residencyHitPositions.contents()
                    .bindMemory(to: UInt32.self, capacity: cfg.topKExperts)
                let missPointer = residencyMissPositions.contents()
                    .bindMemory(to: UInt32.self, capacity: cfg.topKExperts)
                for index in 0..<hitCount {
                    decodeHitSlotsScratch.append(hitPointer[index])
                }
                for index in 0..<missCount {
                    decodeMissSlotsScratch.append(missPointer[index])
                }
                // The GPU appends through an atomic counter, so its ordering
                // is the order threads finished -- not the ascending order
                // `DecodeExpertPartition.populate` produces on the CPU. Sort
                // before both the comparison and the use: the encoders below
                // consume these in the CPU path's order, so an unsorted GPU
                // partition would execute a different assignment even when it
                // classified every expert correctly.
                decodeHitSlotsScratch.sort()
                decodeMissSlotsScratch.sort()
                // The GPU classifies against the residency table as it stood
                // when the kernel ran, which is *before* the cache plan adopts
                // prefetched experts. Adoption memcpys a prefetched expert into
                // its slot and drops it from `plan.misses`, so the GPU's list is
                // legitimately the CPU's plus the adoptions -- one per layer at
                // the shipped prefetch depth of 1. Requiring equality made the
                // mode unusable on any build with prefetch enabled.
                //
                // A miss the CPU sees and the GPU does not is the real error:
                // that direction means the table claimed residency for
                // something the eviction authority had already reclaimed.
                let gpuMisses = decodeMissSlotsScratch.map(Int.init)
                let planMisses = plan.misses.sorted()
                guard Set(gpuMisses).isSuperset(of: planMisses) else {
                    throw ModelError.internalInconsistency(
                        detail: "GPU residency classification disagrees with cache "
                            + "plan: gpu=\(gpuMisses) plan=\(planMisses)")
                }
                totalGPUClassifiedHits &+= UInt64(hitCount)
                totalGPUClassifiedMisses &+= UInt64(missCount)
                if missCount == 0 { totalGPUResidencyAllHitLayers &+= 1 }
                // Execute the CPU partition regardless. It is the authority,
                // and it is the only one that reflects adoption; running the
                // GPU's partition would re-fetch an expert already resident.
                DecodeExpertPartition.populate(
                    topK: cfg.topKExperts,
                    missIndices: plan.misses,
                    hits: &decodeHitSlotsScratch,
                    misses: &decodeMissSlotsScratch)
            } else {
                DecodeExpertPartition.populate(
                    topK: cfg.topKExperts,
                    missIndices: plan.misses,
                    hits: &decodeHitSlotsScratch,
                    misses: &decodeMissSlotsScratch)
            }
        }
        // Capture the populated arrays. Capturing them before `populate` made
        // empty value-semantic snapshots and silently disabled hit/fixup.
        let phase1HitSlots = decodeHitSlotsScratch
        let phase1MissSlots = decodeMissSlotsScratch
        func encodeRoutedPhase1Full(
            _ cb: MTLCommandBuffer,
            argBuf: MTLBuffer,
            routedBufs: [MTLBuffer],
            ioStatus: MTLBuffer? = nil,
            ioStatusOffset: Int = 0
        ) throws {
            try moe.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: cb,
                routedArgBuffer: argBuf,
                routedBlobs: routedBufs,
                routedOffsets: routedOffsets,
                x: routedX,
                acts: moeActs,
                d: D,
                f: FmoE,
                topK: topK,
                ioStatus: ioStatus,
                ioStatusOffset: ioStatusOffset)
        }

        func encodeRoutedPhase1Subset(
            _ cb: MTLCommandBuffer,
            argBuf: MTLBuffer,
            routedBufs: [MTLBuffer],
            activeSlots: MTLBuffer,
            activeSlotIndices: [UInt32],
            activeCount: UInt32,
            ioStatus: MTLBuffer? = nil,
            ioStatusOffset: Int = 0
        ) throws {
            try moe.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: cb,
                routedArgBuffer: argBuf,
                routedBlobs: routedBufs,
                routedOffsets: routedOffsets,
                x: routedX,
                acts: moeActs,
                activeSlots: activeSlots,
                activeSlotIndices: activeSlotIndices,
                activeCount: activeCount,
                d: D,
                f: FmoE,
                topK: topK,
                ioStatus: ioStatus,
                ioStatusOffset: ioStatusOffset)
        }

        if let plan = plannedFetch,
            plan.hits > 0,
            !plan.misses.isEmpty,
            !phase1HitSlots.isEmpty
        {
            let plannedBlobs = try model.routedExpertBuffers(for: plan)
            for blob in plannedBlobs {
                decodeHitSplitRoutedBufsScratch.append(blob.buffer)
                decodeHitSplitRoutedOffsetsScratch.append(Int(blob.offset))
            }
            phase1HitSplitArgBuf = moe.makeRoutedArgumentBuffer(
                routedBlobs: decodeHitSplitRoutedBufsScratch,
                topK: topK,
                routedBufferOffsets: decodeHitSplitRoutedOffsetsScratch)
            if let argBuf = phase1HitSplitArgBuf, plan.hits > 0, !plan.misses.isEmpty {
                writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                guard let cb = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                try encodeRoutedPhase1Subset(
                    cb,
                    argBuf: argBuf,
                    routedBufs: decodeHitSplitRoutedBufsScratch,
                    activeSlots: moeHitActiveSlots,
                    activeSlotIndices: phase1HitSlots,
                    activeCount: UInt32(phase1HitSlots.count))
                phase1HitCB = cb
            }
        }

        if let cb = phase1HitCB {
            overlapCompletionClock?.track(cb)
            cb.commit()
        }
        let missCount = plannedFetch?.misses.count ?? experts.count
        let completionClock = missCount > 0 ? overlapCompletionClock : nil
        let expectedOverlapCompletions = phase1HitCB == nil ? 1 : 2
        if plannedLoad == nil && rdadviseEnabled && rdadvisePolicyMode != .off {
            let requestedMisses = missCount
            let estimatedAdviceBytes = try model.routedExpertAdviceByteEstimate(
                layer: L,
                missCount: requestedMisses)
            if let skipped = shouldSkipRDAdvice(
                position: position,
                requestedMisses: requestedMisses,
                estimatedBytes: estimatedAdviceBytes,
                canOverlapUsefulGPUWork: true)
            {
                recordRDAdvice(skipped, wallNanos: 0)
            } else {
                let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                let result: ExpertIOAdviceResult
                if let plannedFetch {
                    result = try model.adviseRoutedExperts(plan: plannedFetch)
                } else {
                    result = try model.adviseRoutedExperts(layer: L, experts: experts)
                }
                let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice
                recordRDAdvice(result, wallNanos: wallNanos)
                updateRDAdvicePolicy(after: result, position: position)
            }
        }

        // Routed-expert pread — overlaps the shared MLP GPU work above.
        let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let blobs: [TensorView]
        var completedStorageNanos: UInt64?
        let eventLoad = plannedLoad.flatMap { operation -> RoutedExpertLoadOperation? in
            operation.storage.completionToken == nil ? nil : operation
        }
        if let eventLoad {
            // Slot resources and offsets are known from the reservation. Their
            // bytes are consumed only after the shared-event wait encoded
            // below, so no successful completion has to resume this task.
            blobs = try model.routedExpertBuffers(for: eventLoad.plan)
            totalExpertIOHostWaitsAvoided &+= 1
        } else if let plannedFetch, plannedFetch.misses.isEmpty {
            // An all-hit layer has already pinned its current slot generations.
            // Do not manufacture a completed storage operation and an async
            // continuation only to retrieve the same cache views.
            blobs = try model.routedExpertBuffers(for: plannedFetch)
            totalExpertIOHostWaitsAvoided &+= 1
        } else if let plannedLoad {
            totalExpertIOHostWaits &+= plannedLoad.plan.misses.isEmpty ? 0 : 1
            blobs = try await plannedLoad.completion()
            totalIOQueueNanos &+= plannedLoad.storage.submissionToStartNanos
            completedStorageNanos = plannedLoad.storage.completedNanos
        } else if let plannedFetch {
            // The production deferred schedule still uses the split operation
            // so queueing and completion remain observable. It deliberately
            // begins here, after the independent hit work is committed.
            let deferredLoad = try model.beginFetchRoutedExperts(plan: plannedFetch)
            totalExpertIOHostWaits &+= plannedFetch.misses.isEmpty ? 0 : 1
            blobs = try await deferredLoad.completion()
            totalIOQueueNanos &+= deferredLoad.storage.submissionToStartNanos
            completedStorageNanos = deferredLoad.storage.completedNanos
        } else {
            blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
        }
        let layerIo =
            eventLoad == nil
            ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart : 0
        if eventLoad == nil { totalIoNanos &+= layerIo }
        if missCount > 0 && eventLoad == nil {
            totalMissIoNanos &+= layerIo
            if let latest = completionClock?.latest(expected: expectedOverlapCompletions) {
                let overlapEnd = max(tIoStart, latest)
                if overlapEnd < tIoStart + layerIo {
                    totalExposedIoNanos &+= tIoStart + layerIo - overlapEnd
                }
            }
        }
        if let predictivePrefetch, L + 1 < cfg.numLayers {
            // The ring is one read a layer, aimed at the next layer, from the
            // profile's depth. Every alternative allocation measured here lost:
            // a second rank of the same layer (per-expert operations, -3.3%),
            // the same read aimed two layers ahead (-2.8%), a second horizon
            // beside it (-1.4%), the probe-weight margin gate (-4.2%) and a
            // lower disk tier (-0.6%). Each is recorded in
            // docs/qwen38-prefetch-predictor-study.md and Lever 5/8 of
            // benchmark/internal-speeds/v2-qwen38-4bit-telemetry.txt.
            let target = L + 1
            let prediction = predictedNextLayer
            let resident = Set(try model.routedExpertResidentIDs(layer: target))
            // The whole ranked prediction goes in; `begin` drops the experts
            // that are already resident or already in flight and then fills
            // whatever ring slots are free, in rank order.
            //
            // Truncating to top-M here first (v4.3) spent the budget before
            // the residency filter ran. About 85% of the top-M predictions are
            // already cached, so the filter starved the ring instead of aiming
            // it: on a 191-position qwen38 trace, top-4 issued 0.94 reads per
            // layer and covered 23% of the demand misses, where filtering
            // first issues 2.59 and covers 46%. The ring size stays the cap on
            // reads in flight; only the order of cap and filter changed.
            try predictivePrefetch.begin(
                model: model, layer: target,
                experts: prediction,
                resident: resident, currentLayer: L)
        }
        decodeRoutedBufsScratch.removeAll(keepingCapacity: true)
        decodeRoutedOffsetsScratch.removeAll(keepingCapacity: true)
        for blob in blobs {
            decodeRoutedBufsScratch.append(blob.buffer)
            decodeRoutedOffsetsScratch.append(Int(blob.offset))
        }
        let routedBufs = decodeRoutedBufsScratch
        let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // The phase-2 reduce already folded the shared branch (h1Buf
        // as its residual); the tail is a plain residual add.
        let gTail: (MTLCommandBuffer) throws -> Void = { [self] cb in
            try encodeResidualExitDecode(
                commandBuffer: cb,
                hidden: hidden, delta: h2Buf,
                sublayer: .mlp, layer: L)
        }
        guard let routedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let ioToken = eventLoad?.storage.completionToken
        let ioStatus = ioToken.map { ($0.status, $0.statusOffset) }
        if let token = ioToken {
            routedCB.encodeWaitForEvent(token.event, value: token.value)
        }
        if let stagingTransfer = eventLoad?.storage.metalStagingTransfer {
            // The compute command references cache slots only after it has
            // waited for the MTLIO staging event. This is deliberately a GPU
            // blit, not a CPU memcpy or a completion-handler submission.
            try stagingTransfer.encodeCopy(commandBuffer: routedCB)
        }
        let splitArgBuf =
            phase1HitCB != nil && !phase1MissSlots.isEmpty
            ? phase1HitSplitArgBuf
            : nil
        let argBuf =
            splitArgBuf
            ?? moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBufs,
                topK: topK,
                routedBufferOffsets: decodeRoutedOffsetsScratch)
        if splitArgBuf != nil {
            totalHitFixupLayers &+= 1
            writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
            try encodeRoutedPhase1Subset(
                routedCB,
                argBuf: argBuf,
                routedBufs: routedBufs,
                activeSlots: moeMissActiveSlots,
                activeSlotIndices: phase1MissSlots,
                activeCount: UInt32(phase1MissSlots.count),
                ioStatus: ioStatus?.0,
                ioStatusOffset: ioStatus?.1 ?? 0)
        } else {
            try encodeRoutedPhase1Full(
                routedCB,
                argBuf: argBuf,
                routedBufs: routedBufs,
                ioStatus: ioStatus?.0,
                ioStatusOffset: ioStatus?.1 ?? 0)
        }
        try moe.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: routedCB,
            routedArgBuffer: argBuf,
            routedBlobs: routedBufs,
            routedOffsets: routedOffsets,
            acts: moeActs,
            routingWeights: outWeights,
            residual: h1Buf,
            y: h2Buf,
            d: D,
            f: FmoE,
            topK: topK,
            ioStatus: ioStatus?.0,
            ioStatusOffset: ioStatus?.1 ?? 0)
        try gTail(routedCB)
        routedCB.commit()
        if missCount > 0, let completed = completedStorageNanos, completed > 0 {
            let submitted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if submitted >= completed {
                totalIOCompletionToFixupSubmitNanos &+= submitted - completed
            }
        }
        guard pendingRoutedCommand == nil else {
            // The pipeline drains the previous layer's routed CB before
            // queuing the next, so this is a logic error, not a user
            // condition — but it must fail the generation, not trap.
            throw ModelError.internalInconsistency(
                detail: "routed command-buffer pipeline not drained before queuing the next layer")
        }
        pendingRoutedCommand = PendingRoutedCommand(
            cb: routedCB,
            sharedCB: sharedCB,
            phase1HitCB: phase1HitCB,
            expertLease: expertLease,
            storageOperation: eventLoad,
            overlapCompletionClock: eventLoad == nil ? nil : overlapCompletionClock,
            expectedOverlapCompletions: expectedOverlapCompletions,
            kernelRole: splitArgBuf == nil
                ? "moe_phase1_2_routed"
                : "moe_phase1_miss_fixup_phase2",
            encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
        transferredExpertLease = true
        totalBodyNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tBodyStart
        if ProcessInfo.processInfo.environment["TINYTITAN_LAYER_TRACE"] != nil,
            position < 3 || position % 16 == 0
        {
            let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let attnUs = (attnCB.gpuEndTime - attnCB.gpuStartTime) * 1_000_000
            let tailUs = (tailCB.gpuEndTime - tailCB.gpuStartTime) * 1_000_000
            print(
                "TinyTitan layer pos=\(position) L=\(L) "
                    + "body_us=\((now - tBodyStart) / 1000) "
                    + "wait_us=\(waitNanos / 1000) io_us=\(layerIo / 1000) "
                    + "cb1_us=\((tWait - tCb1Start) / 1000) "
                    + "cb2_us=\((now - tCb2Start) / 1000) "
                    + "gpu_attn_us=\(Int(attnUs)) gpu_tail_us=\(Int(tailUs)) "
                    + "gpu_routed_us=\(Int(prevRoutedUs))")
        }
    }
}
