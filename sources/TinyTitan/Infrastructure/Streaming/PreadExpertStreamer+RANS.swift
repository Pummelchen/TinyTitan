import Foundation
import Metal
import Synchronization
import TinyTitanFormat

// The rANS decode path: read a wave's payload runs, then run the kernel into
// the slots, phase 0 before the publish and phase 1 behind the GPU's phase 1.
extension PreadExpertStreamer {

    /// Decodes every miss of a plan, one wave at a time.
    func executeRANSPlan(
        _ plan: ExpertCachePlan,
        firstStage: (() -> Void)? = nil
    ) throws {
        guard let resources = ransResources else { return }
        var start = 0
        while start < plan.misses.count {
            let end = min(start + resources.maximumWave, plan.misses.count)
            let misses = Array(plan.misses[start..<end])
            try decodeRANSExperts(
                experts: misses.map { plan.experts[$0] },
                slots: misses.map { plan.assignedSlots[$0] },
                resources: resources,
                firstStage: start == 0 ? firstStage : nil)
            start = end
        }
    }

    /// One expert, for the single-expert API.
    func decodeRANSExpert(expert: Int, slot: Int) throws {
        guard let resources = ransResources else { return }
        try decodeRANSExperts(
            experts: [expert], slots: [slot], resources: resources, firstStage: nil)
    }

    /// Reads the wave's payload runs and decodes them into the slots.
    ///
    /// The payload is read with the same bounded reader as a raw layer, into a
    /// shared MTLBuffer so the kernel needs no upload copy. Descriptors are
    /// written per expert and phase; the kernel runs one encoder per (expert,
    /// phase), because each slot is its own MTLBuffer.
    func decodeRANSExperts(
        experts: [Int],
        slots: [Int],
        resources: RANSLayerResources,
        firstStage: (() -> Void)?
    ) throws {
        precondition(
            experts.count == slots.count,
            "each rANS expert needs one slot")
        guard !experts.isEmpty else { return }
        guard experts.count <= resources.maximumWave else {
            throw ModelError.internalInconsistency(
                detail: "rANS wave of \(experts.count) exceeds \(resources.maximumWave)")
        }

        var offsets: [UInt64] = []
        var counts: [Int] = []
        var destinations: [UnsafeMutableRawPointer] = []
        let payloadBase = resources.payload.contents()
        for (position, expert) in experts.enumerated() {
            guard let range = resources.payloadRange(expert: expert) else {
                throw ModelError.internalInconsistency(
                    detail: "rANS sidecar has no payload for expert \(expert)")
            }
            offsets.append(range.offset)
            counts.append(range.count)
            destinations.append(
                payloadBase.advanced(by: position * resources.payloadStride))
        }
        let tRead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try readRanges(offsets: offsets, counts: counts, into: destinations)
        let tReadDone = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // One expert's descriptors live in their own aligned region.
        let descriptorStride = resources.layout.models.count * 256 * 32
        let descriptorPointer = resources.descriptors.contents()

        for phase in 0...1 {
            guard let command = resources.queue.makeCommandBuffer() else {
                throw ModelError.internalInconsistency(detail: "rANS command buffer unavailable")
            }
            for (position, expert) in experts.enumerated() {
                let plans = resources.plans[expert].filter { $0.phase == phase }
                guard !plans.isEmpty else { continue }
                let region = descriptorPointer.advanced(
                    by: position * descriptorStride
                ).bindMemory(to: UInt32.self, capacity: plans.count * 8)
                for (index, plan) in plans.enumerated() {
                    var payloadOffset =
                        plan.payloadOffset
                        + UInt32(position * resources.payloadStride)
                    region[index * 8] = payloadOffset
                    region[index * 8 + 1] = plan.tableOffset
                    region[index * 8 + 2] =
                        plan.outputOffset
                        + UInt32(slotBufferOffsets[slots[position]])
                    region[index * 8 + 3] = plan.stride
                    region[index * 8 + 4] = plan.symbolCount
                    region[index * 8 + 5] = plan.streamIndex
                    region[index * 8 + 6] = plan.streamCount
                    region[index * 8 + 7] = plan.slotOffset
                    payloadOffset = 0
                }
                guard let encoder = command.makeComputeCommandEncoder() else {
                    throw ModelError.internalInconsistency(
                        detail: "rANS compute encoder unavailable")
                }
                encoder.setComputePipelineState(resources.pipeline)
                encoder.setBuffer(resources.payload, offset: 0, index: 0)
                encoder.setBuffer(resources.tables, offset: 0, index: 1)
                encoder.setBuffer(
                    resources.descriptors, offset: position * descriptorStride, index: 2)
                encoder.setBuffer(slotBuffers[slots[position]], offset: 0, index: 3)
                encoder.setBuffer(resources.slotTables, offset: 0, index: 4)
                encoder.dispatchThreads(
                    MTLSize(width: plans.count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                encoder.endEncoding()
            }
            command.commit()
            command.waitUntilCompleted()
            let tPhase = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if ProcessInfo.processInfo.environment["TINYTITAN_RANS_TRACE"] == "1" {
                FileHandle.standardError.write(
                    Data(
                        String(
                            format: "[rans] experts=%d phase=%d streams=%d read=%.2f gpu=%.2f ms\n",
                            experts.count, phase,
                            resources.plans[experts[0]].filter { $0.phase == phase }.count,
                            Double(tReadDone - tRead) / 1e6,
                            Double(tPhase - tReadDone) / 1e6
                        ).utf8))
            }
            if let error = command.error {
                throw ModelError.internalInconsistency(
                    detail: "rANS decode failed: \(error)")
            }
            if phase == 0 { firstStage?() }
        }
    }
}
