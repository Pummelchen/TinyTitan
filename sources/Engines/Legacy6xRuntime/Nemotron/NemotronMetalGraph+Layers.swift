import Foundation
import Metal

/// The layer encoders, split out of the graph so both files stay under the
/// repository's 500-line production limit.
extension NemotronMetalGraph {
    // MARK: - Layer encoders

    func encodeEmbedding(_ command: MTLCommandBuffer, token: Int) throws {
        let ref = try install.tensor("language_model.backbone.embeddings.weight")
        let encoder = computeEncoder(command)
        encoder.setComputePipelineState(embedding)
        encoder.setBuffer(weights, offset: ref.fileOffset, index: 0)
        encoder.setBuffer(weights, offset: ref.scaleOffset, index: 1)
        encoder.setBuffer(weights, offset: ref.biasOffset, index: 2)
        encoder.setBuffer(hidden, offset: 0, index: 3)
        var columns = UInt32(ref.columns)
        var bits = UInt32(ref.bits ?? 8)
        var row = UInt32(token)
        encoder.setBytes(&columns, length: 4, index: 4)
        encoder.setBytes(&bits, length: 4, index: 5)
        encoder.setBytes(&row, length: 4, index: 6)
        dispatch(encoder, count: ref.columns)
        encoder.endEncoding()
    }

    func encodeMambaLayer(_ command: MTLCommandBuffer, layer: Int) throws {
        let g = geometry
        let prefix = "language_model.backbone.layers.\(layer).mixer"
        try encodeNorm(command, layer: layer, from: hidden, into: normalized)
        let inProj = try install.tensor("\(prefix).in_proj.weight")
        encodeGEMV(
            command, buffer: weights, weight: inProj.fileOffset,
            scales: inProj.scaleOffset, biases: inProj.biasOffset, bits: inProj.bits ?? 4,
            rows: inProj.rows, columns: inProj.columns,
            x: normalized, xOffset: 0, y: projection, yOffset: 0)

        let convWeight = try install.tensor("\(prefix).conv1d.weight")
        let convBias = try install.tensor("\(prefix).conv1d.bias")
        var channels = UInt32(convDimension)
        var kernel = UInt32(g.convKernel)
        let enc = computeEncoder(command)
        enc.setComputePipelineState(convSilu)
        enc.setBuffer(projection, offset: g.mambaIntermediate * 2, index: 0)
        enc.setBuffer(weights, offset: convWeight.fileOffset, index: 1)
        enc.setBuffer(weights, offset: convBias.fileOffset, index: 2)
        enc.setBuffer(
            convStates,
            offset: mambaIndex[layer] * convDimension * g.convKernel * 4, index: 3)
        enc.setBuffer(convOutput, offset: 0, index: 4)
        enc.setBytes(&channels, length: 4, index: 5)
        enc.setBytes(&kernel, length: 4, index: 6)
        dispatch(enc, count: convDimension)
        enc.endEncoding()

        let aLog = try install.tensor("\(prefix).A_log")
        let skipD = try install.tensor("\(prefix).D")
        let dtBias = try install.tensor("\(prefix).dt_bias")
        var heads = UInt32(g.mambaHeads)
        var headDim = UInt32(g.mambaHeadDim)
        var stateSize = UInt32(g.ssmStateSize)
        var groups = UInt32(g.mambaGroups)
        var timeStepMin = g.timeStepMin
        let ssmEncoder = computeEncoder(command)
        ssmEncoder.setComputePipelineState(ssm)
        ssmEncoder.setBuffer(convOutput, offset: 0, index: 0)
        ssmEncoder.setBuffer(
            convOutput, offset: g.mambaIntermediate * 2, index: 1)
        ssmEncoder.setBuffer(
            convOutput,
            offset: (g.mambaIntermediate + g.mambaGroups * g.ssmStateSize) * 2, index: 2)
        ssmEncoder.setBuffer(
            projection, offset: (g.mambaIntermediate + convDimension) * 2, index: 3)
        ssmEncoder.setBuffer(weights, offset: aLog.fileOffset, index: 4)
        ssmEncoder.setBuffer(weights, offset: skipD.fileOffset, index: 5)
        ssmEncoder.setBuffer(weights, offset: dtBias.fileOffset, index: 6)
        ssmEncoder.setBuffer(
            mambaStates,
            offset: mambaIndex[layer] * g.mambaHeads * g.mambaHeadDim * g.ssmStateSize * 4,
            index: 7)
        ssmEncoder.setBuffer(ssmOutput, offset: 0, index: 8)
        ssmEncoder.setBytes(&heads, length: 4, index: 9)
        ssmEncoder.setBytes(&headDim, length: 4, index: 10)
        ssmEncoder.setBytes(&stateSize, length: 4, index: 11)
        ssmEncoder.setBytes(&groups, length: 4, index: 12)
        ssmEncoder.setBytes(&timeStepMin, length: 4, index: 13)
        dispatch(ssmEncoder, count: g.mambaHeads * g.mambaHeadDim)
        ssmEncoder.endEncoding()

        try encodeGatedNorm(command, layer: layer)
        let outProj = try install.tensor("\(prefix).out_proj.weight")
        encodeGEMV(
            command, buffer: weights, weight: outProj.fileOffset,
            scales: outProj.scaleOffset, biases: outProj.biasOffset, bits: outProj.bits ?? 4,
            rows: g.hiddenSize, columns: g.mambaIntermediate,
            x: mambaNormalized, xOffset: 0, y: layerOutput, yOffset: 0)
        encodeAdd(command, a: hidden, b: layerOutput, out: hidden, count: g.hiddenSize)
    }

    func encodeAttentionLayer(_ command: MTLCommandBuffer, layer: Int) throws {
        let g = geometry
        let prefix = "language_model.backbone.layers.\(layer).mixer"
        try encodeNorm(command, layer: layer, from: hidden, into: normalized)
        let q = try install.tensor("\(prefix).q_proj.weight")
        let k = try install.tensor("\(prefix).k_proj.weight")
        let v = try install.tensor("\(prefix).v_proj.weight")
        let o = try install.tensor("\(prefix).o_proj.weight")
        encodeGEMV(
            command, buffer: weights, weight: q.fileOffset,
            scales: q.scaleOffset, biases: q.biasOffset, bits: q.bits ?? 4,
            rows: q.rows, columns: q.columns, x: normalized, xOffset: 0,
            y: query, yOffset: 0)
        encodeGEMV(
            command, buffer: weights, weight: k.fileOffset,
            scales: k.scaleOffset, biases: k.biasOffset, bits: k.bits ?? 4,
            rows: k.rows, columns: k.columns, x: normalized, xOffset: 0,
            y: key, yOffset: 0)
        encodeGEMV(
            command, buffer: weights, weight: v.fileOffset,
            scales: v.scaleOffset, biases: v.biasOffset, bits: v.bits ?? 4,
            rows: v.rows, columns: v.columns, x: normalized, xOffset: 0,
            y: value, yOffset: 0)

        let slot = g.attentionLayers.firstIndex(of: layer) ?? 0
        let kvStride = maxSequence * g.attentionKVHeads * g.attentionHeadDim
        let base = slot * 2 * kvStride
        let keyBase = base
        let valueBase = base + kvStride
        var kvHeads = UInt32(g.attentionKVHeads)
        var headDim = UInt32(g.attentionHeadDim)
        var maxSeq = UInt32(maxSequence)
        var positionValue = UInt32(position)
        let kvEncoder = computeEncoder(command)
        kvEncoder.setComputePipelineState(kvAppend)
        kvEncoder.setBuffer(key, offset: 0, index: 0)
        kvEncoder.setBuffer(value, offset: 0, index: 1)
        kvEncoder.setBuffer(keyValues, offset: keyBase * 2, index: 2)
        kvEncoder.setBuffer(keyValues, offset: valueBase * 2, index: 3)
        kvEncoder.setBytes(&kvHeads, length: 4, index: 4)
        kvEncoder.setBytes(&headDim, length: 4, index: 5)
        kvEncoder.setBytes(&maxSeq, length: 4, index: 6)
        kvEncoder.setBytes(&positionValue, length: 4, index: 7)
        dispatch(kvEncoder, count: g.attentionKVHeads * g.attentionHeadDim)
        kvEncoder.endEncoding()

        var heads = UInt32(g.attentionHeads)
        var scale = g.attentionScale
        let attnEncoder = computeEncoder(command)
        attnEncoder.setComputePipelineState(attention)
        attnEncoder.setBuffer(query, offset: 0, index: 0)
        attnEncoder.setBuffer(keyValues, offset: keyBase * 2, index: 1)
        attnEncoder.setBuffer(keyValues, offset: valueBase * 2, index: 2)
        attnEncoder.setBuffer(scoreScratch, offset: 0, index: 3)
        attnEncoder.setBuffer(attentionOutput, offset: 0, index: 4)
        attnEncoder.setBytes(&heads, length: 4, index: 5)
        attnEncoder.setBytes(&kvHeads, length: 4, index: 6)
        attnEncoder.setBytes(&headDim, length: 4, index: 7)
        attnEncoder.setBytes(&maxSeq, length: 4, index: 8)
        attnEncoder.setBytes(&positionValue, length: 4, index: 9)
        attnEncoder.setBytes(&scale, length: 4, index: 10)
        dispatch(attnEncoder, count: g.attentionHeads)
        attnEncoder.endEncoding()

        encodeGEMV(
            command, buffer: weights, weight: o.fileOffset,
            scales: o.scaleOffset, biases: o.biasOffset, bits: o.bits ?? 4,
            rows: o.rows, columns: o.columns, x: attentionOutput, xOffset: 0,
            y: layerOutput, yOffset: 0)
        encodeAdd(command, a: hidden, b: layerOutput, out: hidden, count: g.hiddenSize)
    }

    func encodeMoERouter(_ command: MTLCommandBuffer, layer: Int) throws {
        let g = geometry
        let prefix = "language_model.backbone.layers.\(layer).mixer"
        try encodeNorm(command, layer: layer, from: hidden, into: normalized)
        let gate = try install.tensor("\(prefix).gate.weight")
        var rows = UInt32(g.expertCounts[layer])
        var columns = UInt32(g.hiddenSize)
        let encoder = computeEncoder(command)
        encoder.setComputePipelineState(bf16Gemv)
        encoder.setBuffer(weights, offset: gate.fileOffset, index: 0)
        encoder.setBuffer(normalized, offset: 0, index: 1)
        encoder.setBuffer(routerLogits, offset: 0, index: 2)
        encoder.setBytes(&rows, length: 4, index: 3)
        encoder.setBytes(&columns, length: 4, index: 4)
        dispatch(encoder, count: g.expertCounts[layer])
        encoder.endEncoding()
    }

    struct Selection {
        let experts: [Int]
        let weights: [Float]
    }

    func selectExperts(layer: Int) throws -> Selection {
        let g = geometry
        let count = g.expertCounts[layer]
        let pointer = routerLogits.contents().assumingMemoryBound(to: Float.self)
        let bias = try install.floatVector(
            "language_model.backbone.layers.\(layer).mixer.gate.e_score_correction_bias")
        var scores = [Float](repeating: 0, count: count)
        var choice = [Float](repeating: 0, count: count)
        for index in 0..<count {
            let score = 1.0 / (1.0 + expf(-pointer[index]))
            scores[index] = score
            choice[index] = score + (index < bias.count ? bias[index] : 0)
        }
        var order = Array(0..<count).sorted { choice[$0] > choice[$1] }
        order = Array(order.prefix(g.topKExperts))
        var total: Float = 0
        for index in order { total += scores[index] }
        var weights = order.map { scores[$0] }
        if g.routerNormTopK, total > 0 {
            weights = weights.map { $0 / total * g.routedScalingFactor }
        } else {
            weights = weights.map { $0 * g.routedScalingFactor }
        }
        let destination = expertWeights.contents()
            .assumingMemoryBound(to: Float16.self)
        for (index, weight) in weights.enumerated() { destination[index] = Float16(weight) }
        return Selection(experts: order, weights: weights)
    }

    func encodeMoEExperts(
        _ command: MTLCommandBuffer, layer: Int, experts: [Int], weights: [Float]
    ) throws {
        let g = geometry
        let prefix = "language_model.backbone.layers.\(layer).mixer"
        let entry = try install.expertsFor(layer: layer)
        guard let up = entry.roles["up"], let down = entry.roles["down"],
            let upScales = entry.roles["up_scales"],
            let upBiases = entry.roles["up_biases"],
            let downScales = entry.roles["down_scales"],
            let downBiases = entry.roles["down_biases"]
        else {
            throw NemotronInstallError.layout("layer \(layer) expert roles are incomplete")
        }
        let expertWidth = up.shape[0]
        let latentWidth = down.shape[0]
        // Copy the nine selected blobs into wired memory once, so the GPU does
        // not fault 26-50 MB of file-backed pages inside the command buffer.
        if let mapping = install.layerMapping(layer) {
            let destination = expertScratch.contents()
            for (index, expert) in experts.enumerated() {
                memcpy(
                    destination.advanced(by: index * entry.stride),
                    mapping.base.advanced(by: expert * entry.stride),
                    entry.stride)
            }
        }
        let fc1 = try install.tensor("\(prefix).fc1_latent_proj.weight")
        encodeGEMV(
            command, buffer: self.weights, weight: fc1.fileOffset,
            scales: fc1.scaleOffset, biases: fc1.biasOffset, bits: fc1.bits ?? 4,
            rows: g.moeLatentSize, columns: g.hiddenSize,
            x: normalized, xOffset: 0, y: latent, yOffset: 0)
        for index in experts.indices {
            let base = index * entry.stride
            encodeGEMV(
                command, buffer: expertScratch, weight: base + up.offset,
                scales: base + upScales.offset, biases: base + upBiases.offset,
                bits: up.bits ?? 4, rows: up.shape[0], columns: up.shape[1],
                x: latent, xOffset: 0,
                y: expertUp, yOffset: index * expertWidth * 2)
        }
        encodeRelu2(
            command, buffer: expertUp, count: experts.count * expertWidth)
        for index in experts.indices {
            let base = index * entry.stride
            encodeGEMV(
                command, buffer: expertScratch, weight: base + down.offset,
                scales: base + downScales.offset, biases: base + downBiases.offset,
                bits: down.bits ?? 4, rows: down.shape[0], columns: down.shape[1],
                x: expertUp, xOffset: index * expertWidth * 2,
                y: expertDown, yOffset: index * latentWidth * 2)
        }
        var width = UInt32(latentWidth)
        var members = UInt32(experts.count)
        let combine = computeEncoder(command)
        combine.setComputePipelineState(moeCombine)
        combine.setBuffer(expertDown, offset: 0, index: 0)
        combine.setBuffer(expertWeights, offset: 0, index: 1)
        combine.setBuffer(latentAccumulator, offset: 0, index: 2)
        combine.setBytes(&width, length: 4, index: 3)
        combine.setBytes(&members, length: 4, index: 4)
        dispatch(combine, count: latentWidth)
        combine.endEncoding()

        let fc2 = try install.tensor("\(prefix).fc2_latent_proj.weight")
        encodeGEMV(
            command, buffer: self.weights, weight: fc2.fileOffset,
            scales: fc2.scaleOffset, biases: fc2.biasOffset, bits: fc2.bits ?? 4,
            rows: g.hiddenSize, columns: g.moeLatentSize,
            x: latentAccumulator, xOffset: 0, y: moeOutput, yOffset: 0)

        let sharedUp = try install.tensor("\(prefix).shared_experts.up_proj.weight")
        let sharedDown = try install.tensor("\(prefix).shared_experts.down_proj.weight")
        encodeGEMV(
            command, buffer: self.weights, weight: sharedUp.fileOffset,
            scales: sharedUp.scaleOffset, biases: sharedUp.biasOffset,
            bits: sharedUp.bits ?? 4,
            rows: g.moeSharedIntermediate, columns: g.hiddenSize,
            x: normalized, xOffset: 0, y: sharedHidden, yOffset: 0)
        encodeRelu2(command, buffer: sharedHidden, count: g.moeSharedIntermediate)
        encodeGEMV(
            command, buffer: self.weights, weight: sharedDown.fileOffset,
            scales: sharedDown.scaleOffset, biases: sharedDown.biasOffset,
            bits: sharedDown.bits ?? 4,
            rows: g.hiddenSize, columns: g.moeSharedIntermediate,
            x: sharedHidden, xOffset: 0, y: sharedOutput, yOffset: 0)
        encodeAdd(
            command, a: moeOutput, b: sharedOutput, out: layerOutput,
            count: g.hiddenSize)
        encodeAdd(
            command, a: hidden, b: layerOutput, out: hidden, count: g.hiddenSize)
    }

    func encodeHead(_ command: MTLCommandBuffer) throws {
        let g = geometry
        try encodeNorm(
            command, weight: finalNormBuffer, from: hidden, into: normalized,
            count: g.hiddenSize)
        let head = try install.tensor("language_model.lm_head.weight")
        encodeGEMV(
            command, buffer: weights, weight: head.fileOffset,
            scales: head.scaleOffset, biases: head.biasOffset, bits: head.bits ?? 8,
            rows: g.vocabSize, columns: g.hiddenSize,
            x: normalized, xOffset: 0, y: logitsBuffer, yOffset: 0)
    }

}
