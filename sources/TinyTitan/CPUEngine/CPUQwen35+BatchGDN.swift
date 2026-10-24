import Foundation
import TinyTitanKernelsC

/// Batched gated-DeltaNet layers.
///
/// The input projections and the two per-head scalar projections are batched
/// GEMMs over the whole chunk. The convolution and the delta rule are not:
/// both read state from the previous token, so they run token by token in
/// order, exactly as `gatedDeltaNet` does. The delta rule itself is a C/NEON
/// kernel (`tinytitan_delta_rule`) because it was three quarters of the
/// prefill once the projections were cheap; its arithmetic is the decode
/// path's, with the reduction vectorized. The output projection is batched
/// again.
extension CPUQwen35 {

    func gatedDeltaNetBatch(layer: Int, x: [Float], tokens: Int) throws -> [Float] {
        let stem = "\(prefix)layers.\(layer).linear_attn."
        let hk = configuration.linearKeyHeads
        let hv = configuration.linearValueHeads
        let dk = configuration.linearKeyHeadDim
        let dv = configuration.linearValueHeadDim
        let kernel = configuration.convKernel
        guard hk > 0, hv > 0, dk > 0, dv > 0, kernel > 0, hv % hk == 0 else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(layer): head geometry is not usable")
        }
        guard let deltaWeightsA = deltaA[layer],
            let deltaWeightsB = deltaB[layer],
            let taps = convTaps[layer],
            let decayLog = aLog[layer],
            let decayBias = dtBias[layer],
            let normGamma = gdnNorm[layer]
        else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(layer) is missing delta-rule or norm weights")
        }

        let mixed = try CPUBatchedMatmul.multiply(
            try matrix(stem + "in_proj_qkv.weight"), x: x, tokenCount: tokens, threads: threads)
        let z = try CPUBatchedMatmul.multiply(
            try matrix(stem + "in_proj_z.weight"), x: x, tokenCount: tokens, threads: threads)
        let aRows = try denseBatch(deltaWeightsA, rows: hv, x: x, tokens: tokens)
        let bRows = try denseBatch(deltaWeightsB, rows: hv, x: x, tokens: tokens)
        let convDim = mixed.count / tokens
        let keyWidth = hk * dk
        let valueWidth = hv * dv
        guard convDim == 2 * keyWidth + valueWidth, z.count == tokens * valueWidth else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(layer): projection widths do not match the config"
            )
        }

        var readout = [Float](repeating: 0, count: tokens * valueWidth)
        var state = recurrent[layer] ?? [Float](repeating: 0, count: hv * dv * dk)
        var tail = convolution[layer] ?? [Float](repeating: 0, count: (kernel - 1) * convDim)
        var siluScratch = [Float](repeating: 0, count: convDim)
        let inverse = 1 / Float(dv).squareRoot()

        for row in 0..<tokens {
            let mixedBase = row * convDim
            var convolved = [Float](repeating: 0, count: convDim)
            for channel in 0..<convDim {
                var total: Float = 0
                for step in 0..<(kernel - 1) {
                    total += taps[channel * kernel + step] * tail[step * convDim + channel]
                }
                total += taps[channel * kernel + kernel - 1] * mixed[mixedBase + channel]
                convolved[channel] = total
            }
            convolved.withUnsafeMutableBufferPointer { values in
                siluScratch.withUnsafeMutableBufferPointer { scratch in
                    CPUOps.siluInPlace(values, scratch: scratch)
                }
            }
            tail.removeFirst(convDim)
            tail.append(contentsOf: mixed[mixedBase..<(mixedBase + convDim)])

            var query = Array(convolved[0..<keyWidth])
            var key = Array(convolved[keyWidth..<(2 * keyWidth)])
            let value = Array(convolved[(2 * keyWidth)...])
            CPUOps.l2NormalizePerSlice(&query, width: dk, epsilon: configuration.normEpsilon)
            CPUOps.l2NormalizePerSlice(&key, width: dk, epsilon: configuration.normEpsilon)
            applyDeltaRule(
                query: query, key: key, value: value,
                a: Array(aRows[row * hv..<(row + 1) * hv]),
                b: Array(bRows[row * hv..<(row + 1) * hv]),
                decayLog: decayLog, decayBias: decayBias,
                state: &state, readout: &readout, rowBase: row * valueWidth,
                heads: hv, keyHeads: hk, dk: dk, dv: dv, inverse: inverse)
        }
        recurrent[layer] = state
        convolution[layer] = tail

        for row in 0..<tokens {
            let base = row * valueWidth
            var normalized = Array(readout[base..<(base + valueWidth)])
            CPUOps.rmsNormPerSlice(
                &normalized, width: dv, gamma: normGamma,
                epsilon: configuration.normEpsilon)
            for index in 0..<valueWidth {
                normalized[index] *= CPUOps.silu(z[base + index])
            }
            for index in 0..<valueWidth {
                readout[base + index] = normalized[index]
            }
        }
        return try CPUBatchedMatmul.multiply(
            try matrix(stem + "out_proj.weight"), x: readout, tokenCount: tokens, threads: threads)
    }

    /// A dense (unquantized) matrix against T rows: the delta rule's two
    /// per-head projections are BF16 in the snapshot and tiny, so they are
    /// multiplied through the same BLAS wrapper without a dequantization step.
    func denseBatch(_ weights: [Float], rows: Int, x: [Float], tokens: Int) throws -> [Float] {
        let columns = weights.count / rows
        guard columns > 0, x.count == tokens * columns else {
            throw ModelError.internalInconsistency(
                detail: "batched dense projection: \(weights.count) weights for "
                    + "\(rows) rows and \(tokens)x\(x.count / max(tokens, 1)) activations")
        }
        var out = [Float](repeating: 0, count: tokens * rows)
        x.withUnsafeBufferPointer { xBuffer in
            weights.withUnsafeBufferPointer { wBuffer in
                out.withUnsafeMutableBufferPointer { outBuffer in
                    guard let xBase = xBuffer.baseAddress,
                        let wBase = wBuffer.baseAddress,
                        let outBase = outBuffer.baseAddress
                    else { return }
                    tinytitan_sgemm_rowmajor(
                        Int32(1), Int32(tokens), Int32(rows), Int32(columns),
                        xBase, wBase, outBase)
                }
            }
        }
        return out
    }

    /// One token's delta rule. The pointers stay inside this call; the C
    /// kernel writes the updated state and this token's readout.
    func applyDeltaRule(
        query: [Float], key: [Float], value: [Float],
        a: [Float], b: [Float],
        decayLog: [Float], decayBias: [Float],
        state: inout [Float], readout: inout [Float], rowBase: Int,
        heads: Int, keyHeads: Int, dk: Int, dv: Int, inverse: Float
    ) {
        query.withUnsafeBufferPointer { queryBuffer in
            key.withUnsafeBufferPointer { keyBuffer in
                value.withUnsafeBufferPointer { valueBuffer in
                    a.withUnsafeBufferPointer { aBuffer in
                        b.withUnsafeBufferPointer { bBuffer in
                            decayLog.withUnsafeBufferPointer { logBuffer in
                                decayBias.withUnsafeBufferPointer { biasBuffer in
                                    state.withUnsafeMutableBufferPointer { stateBuffer in
                                        readout.withUnsafeMutableBufferPointer { readoutBuffer in
                                            guard let queryBase = queryBuffer.baseAddress,
                                                let keyBase = keyBuffer.baseAddress,
                                                let valueBase = valueBuffer.baseAddress,
                                                let aBase = aBuffer.baseAddress,
                                                let bBase = bBuffer.baseAddress,
                                                let logBase = logBuffer.baseAddress,
                                                let biasBase = biasBuffer.baseAddress,
                                                let stateBase = stateBuffer.baseAddress,
                                                let readoutBase = readoutBuffer.baseAddress
                                            else { return }
                                            // Heads are independent: the state
                                            // of one is not read by another, so
                                            // a row-split is exact. One head per
                                            // worker would make the dispatch
                                            // cost larger than the work.
                                            let usable = max(1, min(threads, heads / 8))
                                            let output = readoutBase + rowBase
                                            if usable <= 1 {
                                                tinytitan_delta_rule(
                                                    queryBase, keyBase, valueBase, aBase, bBase,
                                                    logBase, biasBase,
                                                    heads, keyHeads, dk, dv, inverse,
                                                    stateBase, output)
                                            } else {
                                                let chunk = (heads + usable - 1) / usable
                                                nonisolated(unsafe) let q = queryBase
                                                nonisolated(unsafe) let k = keyBase
                                                nonisolated(unsafe) let v = valueBase
                                                nonisolated(unsafe) let a = aBase
                                                nonisolated(unsafe) let b = bBase
                                                nonisolated(unsafe) let log = logBase
                                                nonisolated(unsafe) let bias = biasBase
                                                nonisolated(unsafe) let statePointer = stateBase
                                                nonisolated(unsafe) let outPointer = output
                                                DispatchQueue.concurrentPerform(
                                                    iterations: usable
                                                ) { slice in
                                                    let first = slice * chunk
                                                    guard first < heads else { return }
                                                    let count = min(chunk, heads - first)
                                                    tinytitan_delta_rule_range(
                                                        q, k, v, a, b, log, bias,
                                                        heads, keyHeads, dk, dv, inverse,
                                                        first, count, statePointer, outPointer)
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
