import Foundation
import Metal

// The PLE n-gram block: gathering rows, rewinding accepted speculations and
// the decode/prefill encoders.
//
// Split out of `RealForwardRunner+Residual.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
// Its two `private` view helpers moved with their only callers.
// MARK: - PLE n-gram block

extension RealForwardRunner {
    /// Reads this token's n-gram rows off storage into the block's input.
    ///
    /// A no-op for families without PLE. The gather is 16 rows of 320 bytes —
    /// 5 KiB — which is three orders of magnitude under one token's routed
    /// expert traffic, so it is done inline rather than scheduled.
    func gatherPLERows(token: Int32) throws {
        guard let ple = pleBlock, let hash = pleHash, let table = ngramTable
        else { return }
        pleContext.insert(token, at: 0)
        if pleContext.count > hash.ngramSize {
            pleContext.removeLast(pleContext.count - hash.ngramSize)
        }
        let rows = hash.rows(context: pleContext)
        try table.gather(rows: rows, into: ple.embedding.contents())
    }

    /// Rewinds the n-gram block's carried state after a speculative pass whose
    /// rows were not all accepted.
    func rewindPLE(acceptedRows: Int, passRows: Int) {
        guard pleBlock != nil, acceptedRows < passRows else { return }
        pleBlock?.rewindWindow(acceptedRows: acceptedRows, passRows: passRows)
        // The hashed context must drop the same rows, or every later token's
        // n-gram ids are computed one position out.
        let discard = min(passRows - acceptedRows, pleContext.count)
        pleContext.removeFirst(discard)
    }

    /// Clears the state that spans a completion: the convolution history and
    /// the token context it hashes. Leaving either in place would let one
    /// prompt's trailing n-grams open the next one.
    func resetPLEState() {
        pleContext.removeAll(keepingCapacity: true)
        pleBlock?.resetState()
    }

    /// Encodes the n-gram block on the layers that carry one.
    func encodePLEDecode(
        commandBuffer: MTLCommandBuffer,
        layer: Int, position: Int, eps: Float
    ) throws {
        guard let ple = pleBlock,
            cfg.ple.layerIndices.contains(layer)
        else { return }
        if activationDumpActive(position: position) {
            dumpActivation("L\(layer)_ple_pre", hidden, count: residualWidth, position: position)
        }
        let weights = PLEBlock.Weights(
            keyProj: projection(try model.pleKeyProj(layer: layer)),
            valueProj: projection(try model.pleValueProj(layer: layer)),
            normKey: vector(try model.pleNormKey(layer: layer)),
            normQuery: vector(try model.pleNormQuery(layer: layer)),
            normConv: vector(try model.pleNormConv(layer: layer)),
            conv1d: vector(try model.pleConv(layer: layer)))
        try ple.encodeDecode(
            commandBuffer: commandBuffer,
            streamsBuffer: hidden,
            weights: weights, eps: eps)
    }

    /// Gathers a whole prefill chunk's n-gram rows.
    ///
    /// Unlike decode, the context for row `i` is the chunk's own tokens plus
    /// whatever preceded the chunk, so this walks the chunk in order and
    /// leaves `pleContext` positioned for the next one.
    func gatherPLERowsPrefill(tokens: ArraySlice<Int32>) throws {
        guard let ple = pleBlock, let hash = pleHash, let table = ngramTable
        else { return }
        let rowBytes = hash.headCount * table.rowBytes
        for (index, token) in tokens.enumerated() {
            pleContext.insert(token, at: 0)
            if pleContext.count > hash.ngramSize {
                pleContext.removeLast(pleContext.count - hash.ngramSize)
            }
            try table.gather(
                rows: hash.rows(context: pleContext),
                into: ple.embedding.contents()
                    .advanced(by: index * rowBytes))
        }
    }

    /// Encodes the n-gram block over a whole prefill chunk.
    func encodePLEPrefill(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer,
        layer: Int, tokens: Int, eps: Float
    ) throws {
        guard let ple = pleBlock,
            cfg.ple.layerIndices.contains(layer)
        else { return }
        let weights = PLEBlock.Weights(
            keyProj: projection(try model.pleKeyProj(layer: layer)),
            valueProj: projection(try model.pleValueProj(layer: layer)),
            normKey: vector(try model.pleNormKey(layer: layer)),
            normQuery: vector(try model.pleNormQuery(layer: layer)),
            normConv: vector(try model.pleNormConv(layer: layer)),
            conv1d: vector(try model.pleConv(layer: layer)))
        try ple.encodeRows(
            commandBuffer: commandBuffer,
            streamsBuffer: hidden, weights: weights,
            tokens: tokens, eps: eps
        ) {
            [self] cb, proj, x, y, rows, columns, count in
            // Same reason as the hyper-connection gate: a promoted projection
            // has no scales or biases for the batched QMM to read.
            if proj.isBF16 {
                let halfBytes = MemoryLayout<Float16>.stride
                for row in 0..<count {
                    try bf16Projection.encode(
                        commandBuffer: cb,
                        weights: proj.weights,
                        weightsOffset: proj.weightsOffset,
                        x: x, xOffset: row * columns * halfBytes,
                        y: y, yOffset: row * rows * halfBytes,
                        m: UInt32(rows), n: UInt32(columns))
                }
                return
            }
            try prefillQMM.encode(
                commandBuffer: cb,
                weights: proj.weights,
                weightsOffset: proj.weightsOffset,
                scales: proj.scales,
                scalesOffset: proj.scalesOffset,
                biases: proj.biases,
                biasesOffset: proj.biasesOffset,
                x: x, y: y,
                t: count, n: rows, k: columns)
        }
    }

    private func projection(_ view: TensorView) -> PLEBlock.Projection {
        PLEBlock.Projection(
            weights: view.buffer,
            weightsOffset: Int(view.offset),
            scales: view.buffer,
            scalesOffset: Int(view.scaleOffset),
            biases: view.buffer,
            biasesOffset: Int(view.biasOffset),
            isBF16: view.dtype == 1)
    }

    private func vector(_ view: TensorView) -> PLEBlock.Vector {
        PLEBlock.Vector(buffer: view.buffer, offset: Int(view.offset))
    }
}
