import Foundation
import TinyTitanKernelsC

/// The text-only Nemotron 3.5 Bolt graph: embedding, 88 pre-norm mixer blocks
/// (40 Mamba-2, 40 LatentMoE, 8 full attention), final norm, untied head.
///
/// Matvecs run through the affine GEMV kernels in `TinyTitanKernelsC`; the
/// per-token elementwise work (norms, conv, SSM update, activations) is the
/// Nemotron kernel file beside them. The graph is deliberately a straight
/// transcription of `modeling_nemotron_h.py` so a NumPy reference can check it
/// layer for layer.
public final class NemotronModel {
    public let install: NemotronInstall
    public let geometry: NemotronGeometry
    public let maxSequence: Int

    private let workers: Int
    private var position = 0

    private let hidden: UnsafeMutablePointer<Float>
    private let normalized: UnsafeMutablePointer<Float>
    private let projection: UnsafeMutablePointer<Float>
    private let convInput: UnsafeMutablePointer<Float>
    private let convOutput: UnsafeMutablePointer<Float>
    private let ssmY: UnsafeMutablePointer<Float>
    private let mambaNormed: UnsafeMutablePointer<Float>
    private let matrixOut: UnsafeMutablePointer<Float>
    private let sequenceBuffer: UnsafeMutablePointer<Float>
    private let latent: UnsafeMutablePointer<Float>
    private let latentAccumulator: UnsafeMutablePointer<Float>
    private let expertHidden: UnsafeMutablePointer<Float>
    private let expertOutput: UnsafeMutablePointer<Float>
    private let sharedOutput: UnsafeMutablePointer<Float>
    private let routerScores: UnsafeMutablePointer<Float>
    private let routerChoice: UnsafeMutablePointer<Float>
    private let logits: UnsafeMutablePointer<Float>

    private let mambaStates: UnsafeMutablePointer<Float>
    private let convStates: UnsafeMutablePointer<Float>
    private let keyValues: UnsafeMutablePointer<Float>

    private let convDimension: Int
    private let mambaStateCount: Int
    private let routerCapacity: Int
    private let mambaIndex: [Int]

    private var layerNormWeights: [Int: [Float]] = [:]
    private var mambaNormWeights: [Int: [Float]] = [:]

    public init(install: NemotronInstall, maxSequence: Int) throws {
        self.install = install
        self.geometry = install.geometry
        self.maxSequence = maxSequence
        self.workers = max(1, min(8, ProcessInfo.processInfo.activeProcessorCount))
        let g = install.geometry
        convDimension = g.mambaIntermediate + 2 * g.mambaGroups * g.ssmStateSize
        mambaStateCount = g.layerKinds.filter { $0 == .mamba }.count
        routerCapacity = g.maxExperts
        var index: [Int] = []
        var mamba = 0
        for kind in g.layerKinds {
            if kind == .mamba {
                index.append(mamba)
                mamba += 1
            } else {
                index.append(-1)
            }
        }
        mambaIndex = index

        func allocate(_ count: Int) -> UnsafeMutablePointer<Float> {
            let pointer = UnsafeMutablePointer<Float>.allocate(capacity: count)
            pointer.initialize(repeating: 0, count: count)
            return pointer
        }
        hidden = allocate(g.hiddenSize)
        normalized = allocate(g.hiddenSize)
        projection = allocate(2 * g.mambaIntermediate + convDimension + g.mambaHeads)
        convInput = allocate(convDimension)
        convOutput = allocate(convDimension)
        ssmY = allocate(g.mambaIntermediate)
        mambaNormed = allocate(g.mambaIntermediate)
        matrixOut = allocate(g.hiddenSize)
        sequenceBuffer = allocate(max(g.hiddenSize, 2 * maxSequence))
        latent = allocate(g.moeLatentSize)
        latentAccumulator = allocate(g.moeLatentSize)
        expertHidden = allocate(max(g.maxMoeIntermediate, g.moeSharedIntermediate))
        expertOutput = allocate(g.moeLatentSize)
        sharedOutput = allocate(g.hiddenSize)
        routerScores = allocate(max(g.maxExperts, 1))
        routerChoice = allocate(max(g.maxExperts, 1))
        logits = allocate(g.vocabSize)
        mambaStates = allocate(
            max(1, mambaStateCount * g.mambaHeads * g.mambaHeadDim * g.ssmStateSize))
        convStates = allocate(max(1, mambaStateCount * convDimension * g.convKernel))
        keyValues = allocate(
            max(
                1,
                g.attentionLayers.count * maxSequence * g.attentionKVHeads
                    * g.attentionHeadDim))

        for layer in 0..<g.numLayers {
            layerNormWeights[layer] = try install.floatVector(
                "language_model.backbone.layers.\(layer).norm.weight")
            if g.layerKinds[layer] == .mamba {
                mambaNormWeights[layer] = try install.floatVector(
                    "language_model.backbone.layers.\(layer).mixer.norm.weight")
            }
        }
        try validateShapes()
    }

    deinit {
        for pointer in [
            hidden, normalized, projection, convInput, convOutput, ssmY,
            mambaNormed, matrixOut, sequenceBuffer, latent, latentAccumulator,
            expertHidden, expertOutput, sharedOutput, routerScores,
            routerChoice, logits, mambaStates, convStates, keyValues,
        ] {
            pointer.deallocate()
        }
    }

    public func reset() {
        let g = geometry
        position = 0
        hidden.update(repeating: 0, count: g.hiddenSize)
        keyValues.update(
            repeating: 0,
            count: g.attentionLayers.count * maxSequence * g.attentionKVHeads
                * g.attentionHeadDim)
        mambaStates.update(
            repeating: 0,
            count: mambaStateCount * g.mambaHeads * g.mambaHeadDim * g.ssmStateSize)
        convStates.update(
            repeating: 0,
            count: mambaStateCount * convDimension * g.convKernel)
    }

    /// Run one token and return the logits. `position` advances.
    public func step(token: Int) throws -> [Float] {
        guard position < maxSequence else {
            throw NemotronInstallError.unsupported(
                "sequence \(position) exceeds the \(maxSequence)-token cache")
        }
        let g = geometry

        // Embedding row, affine 8-bit in both installs.
        let embedding = try install.tensor("language_model.backbone.embeddings.weight")
        try withTensorPointers(embedding) { weights, scales, biases in
            tinytitan_affine_embedding_row(
                weights, scales, biases, g.hiddenSize, embedding.bits ?? 8,
                token, hidden)
        }

        for layer in 0..<g.numLayers {
            switch g.layerKinds[layer] {
            case .mamba:
                try runMambaLayer(layer)
            case .moe:
                try runMoELayer(layer)
            case .attention:
                try runAttentionLayer(layer)
            }
        }

        let finalNorm = try install.floatVector("language_model.backbone.norm_f.weight")
        finalNorm.withUnsafeBufferPointer { weight in
            tinytitan_rmsnorm_f32(hidden, weight.baseAddress, g.hiddenSize,
                                  g.normEpsilon, normalized)
        }
        let head = try install.tensor("language_model.lm_head.weight")
        try withTensorPointers(head) { weights, scales, biases in
            gemv(weights, scales, biases, head.bits ?? 8,
                 rows: g.vocabSize, columns: g.hiddenSize,
                 x: normalized, y: logits)
        }
        position += 1
        return Array(UnsafeBufferPointer(start: logits, count: g.vocabSize))
    }

    // MARK: - Layers

    private func runMambaLayer(_ layer: Int) throws {
        let g = geometry
        let prefix = "language_model.backbone.layers.\(layer).mixer"
        if let weight = layerNormWeights[layer] {
            weight.withUnsafeBufferPointer {
                tinytitan_rmsnorm_f32(hidden, $0.baseAddress, g.hiddenSize,
                                      g.normEpsilon, normalized)
            }
        }
        let inProj = try install.tensor("\(prefix).in_proj.weight")
        try withTensorPointers(inProj) { weights, scales, biases in
            gemv(weights, scales, biases, inProj.bits ?? 4,
                 rows: inProj.rows, columns: inProj.columns,
                 x: normalized, y: projection)
        }
        // Reference split: gate (z), the conv input, then dt.
        let gate = projection
        let convSource = projection + g.mambaIntermediate
        let dt = projection + g.mambaIntermediate + convDimension
        let stateIndex = mambaIndex[layer]
        let state = convStates + stateIndex * convDimension * g.convKernel
        let convWeight = try install.bf16Tensor("\(prefix).conv1d.weight")
        let convBias = g.mambaConvBias
            ? try install.bf16Tensor("\(prefix).conv1d.bias") : nil
        tinytitan_mamba_conv_step(
            convSource, convWeight, convBias, convDimension, g.convKernel,
            state, convOutput)

        let x = convOutput
        let b = convOutput + g.mambaIntermediate
        let c = convOutput + g.mambaIntermediate + g.mambaGroups * g.ssmStateSize
        let aLog = try install.bf16Tensor("\(prefix).A_log")
        let skipD = try install.bf16Tensor("\(prefix).D")
        let dtBias = try install.bf16Tensor("\(prefix).dt_bias")
        let ssmState = mambaStates + stateIndex * g.mambaHeads * g.mambaHeadDim
            * g.ssmStateSize
        parallel(heads: g.mambaHeads) { first, count in
            tinytitan_mamba_ssm_range(
                x, b, c, dt, aLog, skipD, dtBias,
                g.mambaHeads, g.mambaHeadDim, g.ssmStateSize, g.mambaGroups,
                g.timeStepMin, first, count, ssmState, ssmY)
        }
        if let weight = mambaNormWeights[layer] {
            weight.withUnsafeBufferPointer {
                tinytitan_rmsnorm_gated_f32(
                    ssmY, gate, $0.baseAddress, g.mambaIntermediate,
                    g.mambaIntermediate / g.mambaGroups, g.mambaEpsilon,
                    mambaNormed)
            }
        } else {
            tinytitan_rmsnorm_gated_f32(
                ssmY, gate, nil, g.mambaIntermediate,
                g.mambaIntermediate / g.mambaGroups, g.mambaEpsilon,
                mambaNormed)
        }
        let outProj = try install.tensor("\(prefix).out_proj.weight")
        try withTensorPointers(outProj) { weights, scales, biases in
            gemv(weights, scales, biases, outProj.bits ?? 4,
                 rows: g.hiddenSize, columns: g.mambaIntermediate,
                 x: mambaNormed, y: matrixOut)
        }
        add(matrixOut, to: hidden, count: g.hiddenSize)
    }

    private func runMoELayer(_ layer: Int) throws {
        let g = geometry
        let prefix = "language_model.backbone.layers.\(layer).mixer"
        if let weight = layerNormWeights[layer] {
            weight.withUnsafeBufferPointer {
                tinytitan_rmsnorm_f32(hidden, $0.baseAddress, g.hiddenSize,
                                      g.normEpsilon, normalized)
            }
        }
        // Router: BF16 gate, sigmoid, correction bias, top-9 (n_group is 1).
        let gate = try install.tensor("\(prefix).gate.weight")
        let experts = try install.expertsFor(layer: layer)
        let count = experts.count
        let gatePointer = install.pointer(of: gate)
            .assumingMemoryBound(to: UInt16.self)
        tinytitan_bf16_gemv(gatePointer, normalized, count, g.hiddenSize, routerScores)
        let bias = try install.floatVector("\(prefix).gate.e_score_correction_bias")
        for index in 0..<count {
            let score = 1.0 / (1.0 + expf(-routerScores[index]))
            routerScores[index] = score
            routerChoice[index] = score + (index < bias.count ? bias[index] : 0)
        }
        let selected = topIndices(routerChoice, k: g.topKExperts, count: count)
        var total: Float = 0
        for index in selected { total += routerScores[index] }
        if g.routerNormTopK, total > 0 {
            for index in selected {
                routerScores[index] = routerScores[index] / total * g.routedScalingFactor
            }
        } else {
            for index in selected {
                routerScores[index] *= g.routedScalingFactor
            }
        }

        let fc1 = try install.tensor("\(prefix).fc1_latent_proj.weight")
        try withTensorPointers(fc1) { weights, scales, biases in
            gemv(weights, scales, biases, fc1.bits ?? 4,
                 rows: g.moeLatentSize, columns: g.hiddenSize,
                 x: normalized, y: latent)
        }
        latentAccumulator.update(repeating: 0, count: g.moeLatentSize)
        let sharedDown = try install.tensor("\(prefix).shared_experts.down_proj.weight")
        let sharedUp = try install.tensor("\(prefix).shared_experts.up_proj.weight")
        let latentWidth = g.moeLatentSize
        for index in selected {
            guard let up = experts.roles["up"], let down = experts.roles["down"],
                let upScales = experts.roles["up_scales"],
                let upBiases = experts.roles["up_biases"],
                let downScales = experts.roles["down_scales"],
                let downBiases = experts.roles["down_biases"]
            else {
                throw NemotronInstallError.layout("layer \(layer) expert roles are incomplete")
            }
            let base = try install.expertPointer(layer: layer, expert: index)
            let weight = routerScores[index]
            let upWeights = base.advanced(by: up.offset)
                .assumingMemoryBound(to: UInt8.self)
            let upScalePointer = base.advanced(by: upScales.offset)
                .assumingMemoryBound(to: UInt16.self)
            let upBiasPointer = base.advanced(by: upBiases.offset)
                .assumingMemoryBound(to: UInt16.self)
            let downWeights = base.advanced(by: down.offset)
                .assumingMemoryBound(to: UInt8.self)
            let downScalePointer = base.advanced(by: downScales.offset)
                .assumingMemoryBound(to: UInt16.self)
            let downBiasPointer = base.advanced(by: downBiases.offset)
                .assumingMemoryBound(to: UInt16.self)
            gemv(upWeights, upScalePointer, upBiasPointer, up.bits ?? 4,
                 rows: up.shape[0], columns: up.shape[1],
                 x: latent, y: expertHidden)
            tinytitan_relu2_f32(expertHidden, up.shape[0], expertHidden)
            gemv(downWeights, downScalePointer, downBiasPointer, down.bits ?? 4,
                 rows: down.shape[0], columns: down.shape[1],
                 x: expertHidden, y: expertOutput)
            for element in 0..<latentWidth {
                latentAccumulator[element] += weight * expertOutput[element]
            }
        }
        let fc2 = try install.tensor("\(prefix).fc2_latent_proj.weight")
        try withTensorPointers(fc2) { weights, scales, biases in
            gemv(weights, scales, biases, fc2.bits ?? 4,
                 rows: g.hiddenSize, columns: g.moeLatentSize,
                 x: latentAccumulator, y: matrixOut)
        }
        try withTensorPointers(sharedUp) { weights, scales, biases in
            gemv(weights, scales, biases, sharedUp.bits ?? 4,
                 rows: g.moeSharedIntermediate, columns: g.hiddenSize,
                 x: normalized, y: expertHidden)
        }
        tinytitan_relu2_f32(expertHidden, g.moeSharedIntermediate, expertHidden)
        try withTensorPointers(sharedDown) { weights, scales, biases in
            gemv(weights, scales, biases, sharedDown.bits ?? 4,
                 rows: g.hiddenSize, columns: g.moeSharedIntermediate,
                 x: expertHidden, y: sharedOutput)
        }
        for element in 0..<g.hiddenSize {
            hidden[element] += matrixOut[element] + sharedOutput[element]
        }
    }

    private func runAttentionLayer(_ layer: Int) throws {
        let g = geometry
        let prefix = "language_model.backbone.layers.\(layer).mixer"
        if let weight = layerNormWeights[layer] {
            weight.withUnsafeBufferPointer {
                tinytitan_rmsnorm_f32(hidden, $0.baseAddress, g.hiddenSize,
                                      g.normEpsilon, normalized)
            }
        }
        let q = try install.tensor("\(prefix).q_proj.weight")
        let k = try install.tensor("\(prefix).k_proj.weight")
        let v = try install.tensor("\(prefix).v_proj.weight")
        let o = try install.tensor("\(prefix).o_proj.weight")
        let query = projection
        let key = projection + g.attentionHeads * g.attentionHeadDim
        let value = key + g.attentionKVHeads * g.attentionHeadDim
        try withTensorPointers(q) { weights, scales, biases in
            gemv(weights, scales, biases, q.bits ?? 4,
                 rows: q.rows, columns: q.columns, x: normalized, y: query)
        }
        try withTensorPointers(k) { weights, scales, biases in
            gemv(weights, scales, biases, k.bits ?? 4,
                 rows: k.rows, columns: k.columns, x: normalized, y: key)
        }
        try withTensorPointers(v) { weights, scales, biases in
            gemv(weights, scales, biases, v.bits ?? 4,
                 rows: v.rows, columns: v.columns, x: normalized, y: value)
        }
        let slot = g.attentionLayers.firstIndex(of: layer) ?? 0
        let headDim = g.attentionHeadDim
        let kvStride = maxSequence * g.attentionKVHeads * headDim
        let cache = keyValues + slot * kvStride
        for head in 0..<g.attentionKVHeads {
            for d in 0..<headDim {
                cache[position * g.attentionKVHeads * headDim + head * headDim + d] =
                    key[head * headDim + d]
                cache[maxSequence * g.attentionKVHeads * headDim
                    + position * g.attentionKVHeads * headDim
                    + head * headDim + d] = value[head * headDim + d]
            }
        }
        let keyBase = cache
        let valueBase = cache + maxSequence * g.attentionKVHeads * headDim
        let groups = g.attentionHeads / g.attentionKVHeads
        for head in 0..<g.attentionHeads {
            let kvHead = head / groups
            let qh = query + head * headDim
            var maximum: Float = -.infinity
            for token in 0...position {
                var dot: Float = 0
                let kb = keyBase + token * g.attentionKVHeads * headDim
                    + kvHead * headDim
                for d in 0..<headDim { dot += qh[d] * kb[d] }
                dot *= g.attentionScale
                sequenceBuffer[token] = dot
                if dot > maximum { maximum = dot }
            }
            var sum: Float = 0
            for token in 0...position {
                let e = expf(sequenceBuffer[token] - maximum)
                sequenceBuffer[token] = e
                sum += e
            }
            let inverse = 1.0 / sum
            for token in 0...position { sequenceBuffer[token] *= inverse }
            let outHead = sequenceBuffer + maxSequence
            for d in 0..<headDim { outHead[d] = 0 }
            for token in 0...position {
                let probability = sequenceBuffer[token]
                let vb = valueBase + token * g.attentionKVHeads * headDim
                    + kvHead * headDim
                for d in 0..<headDim { outHead[d] += probability * vb[d] }
            }
            for d in 0..<headDim {
                query[head * headDim + d] = outHead[d]
            }
        }
        try withTensorPointers(o) { weights, scales, biases in
            gemv(weights, scales, biases, o.bits ?? 4,
                 rows: o.rows, columns: o.columns, x: query, y: matrixOut)
        }
        add(matrixOut, to: hidden, count: g.hiddenSize)
    }

    // MARK: - Helpers

    private func validateShapes() throws {
        let g = geometry
        guard g.routerGroupCount == 1, g.routerTopKGroup == 1 else {
            throw NemotronInstallError.unsupported(
                "the grouped router has more than one group")
        }
        guard g.hiddenSize % 64 == 0, g.moeLatentSize % 64 == 0,
            g.moeSharedIntermediate % 64 == 0, g.mambaIntermediate % 64 == 0
        else {
            throw NemotronInstallError.unsupported(
                "a quantized width is not a multiple of 64")
        }
        // The attention softmax scratch is shared with the output half.
        guard maxSequence * 2 <= g.mambaIntermediate else {
            throw NemotronInstallError.unsupported(
                "maxSequence \(maxSequence) exceeds the shared scratch plan")
        }
    }

    private func add(
        _ source: UnsafePointer<Float>, to target: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        for index in 0..<count { target[index] += source[index] }
    }

    private func topIndices(
        _ values: UnsafeMutablePointer<Float>, k: Int, count: Int
    ) -> [Int] {
        var selected: [Int] = []
        selected.reserveCapacity(k)
        var used = [Bool](repeating: false, count: count)
        for _ in 0..<min(k, count) {
            var best = -1
            var bestValue = -Float.infinity
            for index in 0..<count where !used[index] {
                if values[index] > bestValue {
                    bestValue = values[index]
                    best = index
                }
            }
            if best >= 0 {
                used[best] = true
                selected.append(best)
            }
        }
        return selected
    }

    private func parallel(heads: Int, _ body: (Int, Int) -> Void) {
        if heads < workers || workers == 1 {
            body(0, heads)
            return
        }
        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            let first = worker * heads / workers
            let end = (worker + 1) * heads / workers
            if first < end { body(first, end - first) }
        }
    }

    private func withTensorPointers(
        _ ref: NemotronTensorRef,
        _ body: (UnsafePointer<UInt8>, UnsafePointer<UInt16>, UnsafePointer<UInt16>) throws -> Void
    ) rethrows {
        let base = install.pointer(of: ref)
        try body(
            base.assumingMemoryBound(to: UInt8.self),
            install.pointer(of: ref, scale: true).assumingMemoryBound(to: UInt16.self),
            install.pointer(of: ref, scale: false).assumingMemoryBound(to: UInt16.self))
    }

    private func gemv(
        _ weights: UnsafePointer<UInt8>,
        _ scales: UnsafePointer<UInt16>,
        _ biases: UnsafePointer<UInt16>,
        _ bits: Int,
        rows: Int,
        columns: Int,
        x: UnsafePointer<Float>,
        y: UnsafeMutablePointer<Float>
    ) {
        let groups = columns / 64
        let workers = rows >= 256 ? self.workers : 1
        if workers > 1 {
            DispatchQueue.concurrentPerform(iterations: workers) { worker in
                let first = worker * rows / workers
                let count = (worker + 1) * rows / workers - first
                if count <= 0 { return }
                let rowBytes = columns * bits / 8
                let weightPointer = weights.advanced(by: first * rowBytes)
                let scalePointer = scales.advanced(by: first * groups)
                let biasPointer = biases.advanced(by: first * groups)
                let output = y.advanced(by: first)
                if bits == 4 {
                    tinytitan_int4_affine_gemv(
                        weightPointer, scalePointer, biasPointer, x,
                        count, columns, output)
                } else {
                    tinytitan_int8_affine_gemv(
                        weightPointer, scalePointer, biasPointer, x,
                        count, columns, output)
                }
            }
        } else if bits == 4 {
            tinytitan_int4_affine_gemv(
                weights, scales, biases, x, rows, columns, y)
        } else {
            tinytitan_int8_affine_gemv(
                weights, scales, biases, x, rows, columns, y)
        }
    }
}

extension NemotronInstall {
    /// A BF16 tensor's raw bits, for the C kernels that read bf16 directly.
    func bf16Tensor(_ name: String) throws -> UnsafePointer<UInt16> {
        let ref = try tensor(name)
        guard ref.dtype == 1 else {
            throw NemotronInstallError.unsupported("\(name) is not BF16")
        }
        return pointer(of: ref).assumingMemoryBound(to: UInt16.self)
    }
}
