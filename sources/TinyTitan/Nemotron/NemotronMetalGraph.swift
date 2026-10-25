import Foundation
import Metal

/// The Nemotron graph with the hidden state, Mamba state and KV cache resident
/// on the GPU.
///
/// This is the measurement path. `NemotronMetal` (the matvec backend) is
/// correct but launch-latency bound: this model needs a CPU-side norm, conv or
/// scan between a layer's GEMVs, and an M3's GPU downclocks in the few
/// milliseconds it is idle, so each of its ~350 waits costs ~10 ms even though
/// the same GEMV takes 0.8 ms back to back. Here a whole mamba or attention
/// layer is encoded into one command buffer, and the graph stops only at a MoE
/// router, where 9 of 240-496 experts are chosen on the CPU. That is 40 waits a
/// token instead of ~350.
public final class NemotronMetalGraph {
    public let install: NemotronInstall
    public let geometry: NemotronGeometry

    private let context: MetalContext
    private let int4: MTLComputePipelineState
    private let int8: MTLComputePipelineState
    private let rmsnorm: MTLComputePipelineState
    private let rmsnormGated: MTLComputePipelineState
    private let convSilu: MTLComputePipelineState
    private let ssm: MTLComputePipelineState
    private let relu2: MTLComputePipelineState
    private let add: MTLComputePipelineState
    private let moeCombine: MTLComputePipelineState
    private let embedding: MTLComputePipelineState
    private let attention: MTLComputePipelineState
    private let bf16Gemv: MTLComputePipelineState
    private let kvAppend: MTLComputePipelineState

    private let weights: MTLBuffer
    private var layers: [Int: MTLBuffer] = [:]

    private let hidden: MTLBuffer
    private let normalized: MTLBuffer
    private let projection: MTLBuffer
    private let convOutput: MTLBuffer
    private let ssmOutput: MTLBuffer
    private let mambaNormalized: MTLBuffer
    private let layerOutput: MTLBuffer
    private let query: MTLBuffer
    private let key: MTLBuffer
    private let value: MTLBuffer
    private let attentionOutput: MTLBuffer
    private let latent: MTLBuffer
    private let latentAccumulator: MTLBuffer
    private let expertUp: MTLBuffer
    private let expertDown: MTLBuffer
    private let expertWeights: MTLBuffer
    private let expertScratch: MTLBuffer
    private let routerLogits: MTLBuffer
    private let sharedHidden: MTLBuffer
    private let sharedOutput: MTLBuffer
    private let moeOutput: MTLBuffer
    private let logitsBuffer: MTLBuffer
    private let mambaStates: MTLBuffer
    private let convStates: MTLBuffer
    private let keyValues: MTLBuffer
    private let scoreScratch: MTLBuffer

    private let mambaIndex: [Int]
    private let convDimension: Int
    private let maxSequence: Int
    private var position = 0
    private var normBuffers: [Int: MTLBuffer] = [:]
    private var mambaNormBuffers: [Int: MTLBuffer] = [:]
    private var finalNormBuffer: MTLBuffer

    public init(
        install: NemotronInstall, context: MetalContext, maxSequence: Int
    ) throws {
        self.install = install
        self.geometry = install.geometry
        self.context = context
        self.maxSequence = maxSequence
        let g = install.geometry
        convDimension = g.mambaIntermediate + 2 * g.mambaGroups * g.ssmStateSize
        var index: [Int] = []
        var counter = 0
        for kind in g.layerKinds {
            if kind == .mamba {
                index.append(counter)
                counter += 1
            } else {
                index.append(-1)
            }
        }
        mambaIndex = index

        func constant(_ index: Int) -> MetalFunctionConstant {
            MetalFunctionConstant(index: index, value: .bool(false))
        }
        int4 = try context.pipeline(
            "dequant_int4_gemv_simd",
            constants: [constant(22), constant(26)])
        int8 = try context.pipeline(
            "dequant_int8_gemv_simd", constants: [constant(72)])
        rmsnorm = try context.pipeline("nemotron_rmsnorm_f16")
        rmsnormGated = try context.pipeline("nemotron_rmsnorm_gated_f16")
        convSilu = try context.pipeline("nemotron_conv_silu_f16")
        ssm = try context.pipeline("nemotron_ssm_f16")
        relu2 = try context.pipeline("nemotron_relu2_f16")
        add = try context.pipeline("nemotron_add_f16")
        moeCombine = try context.pipeline("nemotron_moe_combine_f16")
        embedding = try context.pipeline("nemotron_embedding_f16")
        attention = try context.pipeline("nemotron_attention_f16")
        bf16Gemv = try context.pipeline("nemotron_bf16_gemv")
        kvAppend = try context.pipeline("nemotron_kv_append_f16")

        let page = Int(getpagesize())
        func rounded(_ value: Int) -> Int { ((value + page - 1) / page) * page }
        func wrap(_ mapping: (base: UnsafeRawPointer, size: Int), name: String) throws
            -> MTLBuffer
        {
            guard let buffer = context.device.makeBuffer(
                bytesNoCopy: UnsafeMutableRawPointer(mutating: mapping.base),
                length: rounded(mapping.size),
                options: .storageModeShared, deallocator: nil)
            else {
                throw NemotronInstallError.unsupported("cannot wrap \(name) in Metal")
            }
            return buffer
        }
        let residentMapping = install.weightsMapping
        if ProcessInfo.processInfo.environment["TINYTITAN_NEMOTRON_NOCOPY"] == "1",
            let buffer = context.device.makeBuffer(
                bytesNoCopy: UnsafeMutableRawPointer(mutating: residentMapping.base),
                length: rounded(residentMapping.size),
                options: .storageModeShared, deallocator: nil)
        {
            weights = buffer
        } else if let buffer = context.device.makeBuffer(
            length: residentMapping.size, options: .storageModeShared)
        {
            // A GPU read of a file-backed mmap faults a page at a time on the
            // GPU's path, and that is what made the first graph version slower
            // than the CPU. The resident tensors are 5-9 GB, so they are
            // copied once into wired shared memory at load.
            memcpy(buffer.contents(), residentMapping.base, residentMapping.size)
            weights = buffer
        } else {
            throw NemotronInstallError.unsupported("cannot allocate the resident weight buffer")
        }
        for (layer, _) in install.experts {
            guard let mapping = install.layerMapping(layer) else { continue }
            layers[layer] = try wrap(mapping, name: "layer \(layer) experts")
        }

        func half(_ count: Int) -> MTLBuffer {
            context.device.makeBuffer(
                length: max(count, 1) * MemoryLayout<Float16>.size,
                options: .storageModeShared)!
        }
        func float(_ count: Int) -> MTLBuffer {
            context.device.makeBuffer(
                length: max(count, 1) * MemoryLayout<Float>.size,
                options: .storageModeShared)!
        }
        hidden = half(g.hiddenSize)
        normalized = half(g.hiddenSize)
        projection = half(
            2 * g.mambaIntermediate + convDimension + g.mambaHeads + g.hiddenSize)
        convOutput = half(convDimension)
        ssmOutput = half(g.mambaIntermediate)
        mambaNormalized = half(g.mambaIntermediate)
        layerOutput = half(g.hiddenSize)
        query = half(g.attentionHeads * g.attentionHeadDim)
        key = half(g.attentionKVHeads * g.attentionHeadDim)
        value = half(g.attentionKVHeads * g.attentionHeadDim)
        attentionOutput = half(g.attentionHeads * g.attentionHeadDim)
        latent = half(g.moeLatentSize)
        latentAccumulator = half(g.moeLatentSize)
        expertUp = half(g.topKExperts * g.maxMoeIntermediate)
        expertDown = half(g.topKExperts * g.moeLatentSize)
        expertWeights = half(g.topKExperts)
        let expertStride = install.experts.values.map(\.stride).max() ?? 0
        expertScratch = half(max(1, g.topKExperts * expertStride / 2))
        routerLogits = float(g.maxExperts)
        sharedHidden = half(g.moeSharedIntermediate)
        sharedOutput = half(g.hiddenSize)
        moeOutput = half(g.hiddenSize)
        logitsBuffer = half(g.vocabSize)
        mambaStates = float(
            max(1, counter * g.mambaHeads * g.mambaHeadDim * g.ssmStateSize))
        convStates = float(max(1, counter * convDimension * g.convKernel))
        keyValues = half(
            max(1, g.attentionLayers.count * maxSequence * 2 * g.attentionKVHeads
                * g.attentionHeadDim))
        scoreScratch = float(g.attentionHeads * maxSequence)
        func bf16Buffer(_ values: [Float]) -> MTLBuffer {
            let buffer = context.device.makeBuffer(
                length: values.count * 2, options: .storageModeShared)!
            let pointer = buffer.contents().assumingMemoryBound(to: UInt16.self)
            for index in values.indices {
                pointer[index] = UInt16(truncatingIfNeeded: values[index].bitPattern >> 16)
            }
            return buffer
        }
        for layer in 0..<g.numLayers {
            normBuffers[layer] = bf16Buffer(
                try install.floatVector(
                    "language_model.backbone.layers.\(layer).norm.weight"))
            if g.layerKinds[layer] == .mamba {
                mambaNormBuffers[layer] = bf16Buffer(
                    try install.floatVector(
                        "language_model.backbone.layers.\(layer).mixer.norm.weight"))
            }
        }
        finalNormBuffer = bf16Buffer(
            try install.floatVector("language_model.backbone.norm_f.weight"))
    }

    public func reset() {
        position = 0
        memset(hidden.contents(), 0, hidden.length)
        memset(mambaStates.contents(), 0, mambaStates.length)
        memset(convStates.contents(), 0, convStates.length)
        memset(keyValues.contents(), 0, keyValues.length)
    }

    /// Run one token. Returns FP32 logits.
    public func step(token: Int) throws -> [Float] {
        guard position < maxSequence else {
            throw NemotronInstallError.unsupported("sequence exceeds the KV cache")
        }
        let g = geometry
        var command = context.queue.makeCommandBuffer()!
        try encodeEmbedding(command, token: token)
        let requested = ProcessInfo.processInfo.environment["TINYTITAN_NEMOTRON_LAYERS"]
            .flatMap { Int($0) } ?? g.numLayers
        for layer in 0..<min(g.numLayers, max(0, requested)) {
            switch g.layerKinds[layer] {
            case .mamba:
                try encodeMambaLayer(command, layer: layer)
            case .attention:
                try encodeAttentionLayer(command, layer: layer)
            case .moe:
                // The router is the one point the CPU has to see: 9 of
                // 240-496 experts are chosen from 496 logits.
                try encodeMoERouter(command, layer: layer)
                command.commit()
                command.waitUntilCompleted()
                let selected = try selectExperts(layer: layer)
                command = context.queue.makeCommandBuffer()!
                try encodeMoEExperts(
                    command, layer: layer, experts: selected.experts,
                    weights: selected.weights)
            }
        }
        try encodeHead(command)
        command.commit()
        command.waitUntilCompleted()
        position += 1
        return try readLogits()
    }

    // MARK: - Layer encoders

    private func encodeEmbedding(_ command: MTLCommandBuffer, token: Int) throws {
        let ref = try install.tensor("language_model.backbone.embeddings.weight")
        let encoder = command.makeComputeCommandEncoder()!
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

    private func encodeMambaLayer(_ command: MTLCommandBuffer, layer: Int) throws {
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
        let enc = command.makeComputeCommandEncoder()!
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
        let ssmEncoder = command.makeComputeCommandEncoder()!
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

    private func encodeAttentionLayer(_ command: MTLCommandBuffer, layer: Int) throws {
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
        let kvEncoder = command.makeComputeCommandEncoder()!
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
        let attnEncoder = command.makeComputeCommandEncoder()!
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

    private func encodeMoERouter(_ command: MTLCommandBuffer, layer: Int) throws {
        let g = geometry
        let prefix = "language_model.backbone.layers.\(layer).mixer"
        try encodeNorm(command, layer: layer, from: hidden, into: normalized)
        let gate = try install.tensor("\(prefix).gate.weight")
        var rows = UInt32(g.expertCounts[layer])
        var columns = UInt32(g.hiddenSize)
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(bf16Gemv)
        encoder.setBuffer(weights, offset: gate.fileOffset, index: 0)
        encoder.setBuffer(normalized, offset: 0, index: 1)
        encoder.setBuffer(routerLogits, offset: 0, index: 2)
        encoder.setBytes(&rows, length: 4, index: 3)
        encoder.setBytes(&columns, length: 4, index: 4)
        dispatch(encoder, count: g.expertCounts[layer])
        encoder.endEncoding()
    }

    private struct Selection {
        let experts: [Int]
        let weights: [Float]
    }

    private func selectExperts(layer: Int) throws -> Selection {
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

    private func encodeMoEExperts(
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
        let combine = command.makeComputeCommandEncoder()!
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

    private func encodeHead(_ command: MTLCommandBuffer) throws {
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

    // MARK: - Kernel helpers

    private func encodeNorm(
        _ command: MTLCommandBuffer, layer: Int, from source: MTLBuffer,
        into destination: MTLBuffer
    ) throws {
        guard let weight = normBuffers[layer] else { return }
        var n = UInt32(geometry.hiddenSize)
        var eps = geometry.normEpsilon
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(rmsnorm)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(destination, offset: 0, index: 2)
        encoder.setBytes(&n, length: 4, index: 3)
        encoder.setBytes(&eps, length: 4, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeNorm(
        _ command: MTLCommandBuffer, weight: MTLBuffer, from source: MTLBuffer,
        into destination: MTLBuffer, count: Int
    ) throws {
        var n = UInt32(count)
        var eps = geometry.normEpsilon
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(rmsnorm)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(destination, offset: 0, index: 2)
        encoder.setBytes(&n, length: 4, index: 3)
        encoder.setBytes(&eps, length: 4, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeGatedNorm(
        _ command: MTLCommandBuffer, layer: Int
    ) throws {
        guard let buffer = mambaNormBuffers[layer] else { return }
        let g = geometry
        var group = UInt32(g.mambaIntermediate / g.mambaGroups)
        var eps = g.mambaEpsilon
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(rmsnormGated)
        encoder.setBuffer(ssmOutput, offset: 0, index: 0)
        // `proj` starts with `gate`, the same layout the reference splits.
        encoder.setBuffer(projection, offset: 0, index: 1)
        encoder.setBuffer(buffer, offset: 0, index: 2)
        encoder.setBuffer(mambaNormalized, offset: 0, index: 3)
        encoder.setBytes(&group, length: 4, index: 4)
        encoder.setBytes(&eps, length: 4, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: g.mambaGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeGEMV(
        _ command: MTLCommandBuffer,
        buffer: MTLBuffer, weight: Int, scales: Int, biases: Int, bits: Int,
        rows: Int, columns: Int,
        x: MTLBuffer, xOffset: Int, y: MTLBuffer, yOffset: Int
    ) {
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(bits == 4 ? int4 : int8)
        encoder.setBuffer(buffer, offset: weight, index: 0)
        encoder.setBuffer(buffer, offset: scales, index: 1)
        encoder.setBuffer(buffer, offset: biases, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var m = UInt32(rows)
        var n = UInt32(columns)
        encoder.setBytes(&m, length: 4, index: 5)
        encoder.setBytes(&n, length: 4, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * 8, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeRelu2(_ command: MTLCommandBuffer, buffer: MTLBuffer, count: Int) {
        var n = UInt32(count)
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(relu2)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBuffer(buffer, offset: 0, index: 1)
        encoder.setBytes(&n, length: 4, index: 2)
        dispatch(encoder, count: count)
        encoder.endEncoding()
    }

    private func encodeAdd(
        _ command: MTLCommandBuffer, a: MTLBuffer, b: MTLBuffer, out: MTLBuffer,
        count: Int
    ) {
        var n = UInt32(count)
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(add)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(b, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        encoder.setBytes(&n, length: 4, index: 3)
        dispatch(encoder, count: count)
        encoder.endEncoding()
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder, count: Int) {
        let threads = 256
        let groups = (count + threads - 1) / threads
        encoder.dispatchThreadgroups(
            MTLSize(width: max(groups, 1), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    private func readLogits() throws -> [Float] {
        let count = geometry.vocabSize
        let source = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        var out = [Float](repeating: 0, count: count)
        for index in 0..<count { out[index] = Float(source[index]) }
        return out
    }
}
