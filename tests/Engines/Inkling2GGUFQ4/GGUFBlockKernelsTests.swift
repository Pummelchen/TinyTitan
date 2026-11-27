import Foundation
import Metal
import Testing

@testable import Inkling2GGUFQ4
@testable import TinyTitanCore

/// The Inkling 2 plugin's own kernels, against their host references: the
/// small-rows chunk GEMM, the chunk and one-token decode attention, and the
/// controller apply.
///
/// The former suite also covered the *legacy* runtime's wider GGUF type set by
/// importing its `GGUFType`; this plugin keeps only the types the shipped Q4
/// install uses and that module is gone (TT-142). What is left is the reference
/// coverage for the kernels the shipped plugins actually run — the check a
/// performance change must keep passing.
@Suite struct GGUFBlockKernelsTests {
    static func activation(_ count: Int, seed: Int) -> [Float] {
        (0..<count).map { Float((($0 + seed) * 17) % 29) / 29.0 - 0.5 }
    }

    static func buffer<T>(_ device: MTLDevice, _ values: [T]) throws -> MTLBuffer {
        // `makeBuffer(bytes:length:)` copies `length` bytes, so asking for 16
        // from a 4-byte array read 12 bytes past it (ASan: heap-buffer-overflow
        // in `newBufferWithBytes:length:`). Pad the copy instead: a kernel that
        // reads a 16-byte minimum still sees zeros, and the copy stays inside
        // the array it was given.
        let bytes = values.withUnsafeBytes { Array($0) }
        var padded = [UInt8](repeating: 0, count: max(bytes.count, 16))
        padded.replaceSubrange(0..<bytes.count, with: bytes)
        let made = padded.withUnsafeBytes { raw -> MTLBuffer? in
            guard let base = raw.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let made else { throw MetalError.bufferAllocationFailed("test operand") }
        return made
    }

    static func run(
        _ context: MetalContext, _ body: (MTLComputeCommandEncoder) throws -> Void
    ) throws {
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        try body(encoder)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
    }

    static func floats(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
        Array(
            UnsafeBufferPointer(
                start: buffer.contents().bindMemory(to: Float.self, capacity: count), count: count))
    }

    /// y[p][row] against scale(p) * W[matrix(p)][row] . x[xRow(p)] in Double.
}

extension GGUFBlockKernelsTests {
    /// GPU ms and GFLOP/s of the small-rows f32 GEMM over a chunk at the
    /// controller (64 x 4,608) and router (128 x 2,304) shapes, through
    /// `InklingQ4Ops.gemv`, against the host result.
    @Test(arguments: [
        (64, 4_608, 512), (128, 2_304, 512), (24, 2_304, 37), (64, 4_608, 64), (128, 2_304, 64),
    ])
    func smallRowsChunkGemm(rows: Int, columns: Int, tokens: Int) throws {
        let context = try MetalContext()
        let ops = try InklingQ4Ops(context: context)
        let device = context.device
        let w = Self.activation(rows * columns, seed: 5)
        let x = Self.activation(tokens * columns, seed: 9)
        let wb = try Self.buffer(device, w)
        let tensor = InklingQ4Tensor(
            operand: .init(wb), type: .f32, rows: rows, columns: columns, byteCount: w.count * 4)
        let xb = try Self.buffer(device, x)
        let y = try #require(device.makeBuffer(length: tokens * rows * 4))
        var samples: [Double] = []
        for _ in 0..<5 {
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
            for _ in 0..<10 {
                try ops.gemv(tensor, x: .init(xb), y: .init(y), tokens: tokens, encoder: encoder)
            }
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            samples.append((commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) / 10)
        }
        let got = Self.floats(y, tokens * rows)
        for t in Swift.stride(from: 0, to: tokens, by: 7) {
            for r in 0..<rows {
                var want = 0.0
                for c in 0..<columns {
                    want += Double(w[r * columns + c]) * Double(x[t * columns + c])
                }
                #expect(abs(Double(got[t * rows + r]) - want) <= 1e-4 * max(1, abs(want)))
            }
        }
        let median = samples.sorted()[2]
        print(
            String(
                format: "small-rows GEMM %dx%d x%d tokens: %.3f ms, %.1f GFLOP/s", rows, columns,
                tokens, median * 1e3, 2.0 * Double(rows * columns * tokens) / median / 1e9))
    }
}

extension GGUFBlockKernelsTests {
    /// The tiled chunk attention against a CPU reference of the decode
    /// kernel's definition: relative bias within the extent, unscaled bias and
    /// sink, the sink's extra logit 0, sigmoid gate; a plain cache from
    /// position 0, and a ring with a window and a non-zero start.
    @Test(arguments: [false, true])
    func chunkAttentionMatchesReference(ring: Bool) throws {
        try checkAttention(tokens: 37, ring: ring)
    }

    /// The one-token path runs the decode kernel instead of the tiled chunk
    /// kernel, and it had no reference test at all — only the chunk path was
    /// pinned, which is how AUD-0091's missing head-dimension bound went
    /// unnoticed. Same reference, one query.
    @Test(arguments: [false, true])
    func oneTokenDecodeAttentionMatchesReference(ring: Bool) throws {
        try checkAttention(tokens: 1, ring: ring)
    }

    private func checkAttention(tokens: Int, ring: Bool) throws {
        let context = try MetalContext()
        let ops = try InklingQ4Ops(context: context)
        let device = context.device
        let (heads, kvHeads, dim, rank, extent) = (6, 2, 128, 16, 20)
        let capacity = ring ? 48 : 128
        let start = ring ? 30 : 0
        let window = ring ? 25 : 1 << 30
        let last = start + tokens - 1
        let slots = ring ? min(last + 1, capacity) : start + tokens
        let kv = (0..<(capacity * kvHeads * dim)).map { Float(($0 * 29) % 37) / 37 - 0.5 }
        let vv = (0..<(capacity * kvHeads * dim)).map { Float(($0 * 17) % 41) / 41 - 0.5 }
        let qv = Self.activation(tokens * heads * dim, seed: 2)
        let rv = Self.activation(tokens * heads * rank, seed: 3)
        let bv = Self.activation(extent * rank, seed: 4)
        let sv = Self.activation(tokens * heads, seed: 5)
        let gv = Self.activation(tokens * heads * dim, seed: 6)
        let kh = kv.map(Float16.init)
        let vh = vv.map(Float16.init)
        let out = try #require(device.makeBuffer(length: tokens * heads * dim * 4))
        let args = InklingQ4Ops.AttentionArgs(
            heads: UInt32(heads), kvHeads: UInt32(kvHeads), headDim: UInt32(dim),
            relative: UInt32(rank), extent: UInt32(extent), capacity: UInt32(capacity),
            ring: ring ? 1 : 0, scale: 1 / Float(dim).squareRoot(), window: UInt32(window),
            tokens: UInt32(tokens))
        try Self.run(context) { encoder in
            try ops.attention(
                q: .init(try Self.buffer(device, qv)), kCache: try Self.buffer(device, kh),
                vCache: try Self.buffer(device, vh), r: .init(try Self.buffer(device, rv)),
                relativeBias: InklingQ4Tensor(
                    operand: .init(try Self.buffer(device, bv)), type: .f32, rows: extent,
                    columns: rank, byteCount: bv.count * 4),
                sinks: .init(try Self.buffer(device, sv)),
                gate: .init(try Self.buffer(device, gv)), out: .init(out),
                step: .init(try Self.buffer(device, [UInt32(start)])), args: args,
                encoder: encoder)
        }
        let got = Self.floats(out, tokens * heads * dim)
        var worst = 0.0
        for t in 0..<tokens {
            let position = start + t
            for h in 0..<heads {
                let g = h / (heads / kvHeads)
                var logits: [(Int, Double)] = []
                for j in 0..<slots {
                    let held = ring ? last - (last + capacity - j) % capacity : j
                    guard held <= position, position - held < window else { continue }
                    let distance = position - held
                    var dot = 0.0
                    for e in 0..<dim {
                        dot +=
                            Double(qv[(t * heads + h) * dim + e])
                            * Double(Float(kh[(j * kvHeads + g) * dim + e]))
                    }
                    var bias = 0.0
                    if distance < extent {
                        for e in 0..<rank {
                            bias +=
                                Double(rv[(t * heads + h) * rank + e])
                                * Double(bv[distance * rank + e])
                        }
                    }
                    logits.append(
                        (j, dot / Double(dim).squareRoot() + bias - Double(sv[t * heads + h])))
                }
                let peak = max(0, logits.map(\.1).max() ?? 0)
                let denominator = logits.reduce(exp(-peak)) { $0 + exp($1.1 - peak) }
                for d in 0..<dim {
                    var acc = 0.0
                    for (j, l) in logits {
                        acc += exp(l - peak) * Double(Float(vh[(j * kvHeads + g) * dim + d]))
                    }
                    let at = (t * heads + h) * dim + d
                    let want = acc / denominator / (1 + exp(-Double(gv[at])))
                    worst = max(worst, abs(Double(got[at]) - want))
                }
            }
        }
        print("chunk attention ring=\(ring): max |error| \(worst)")
        #expect(worst < 1e-4)
    }
}

extension GGUFBlockKernelsTests {
    /// A chunk's controller apply (rows reused across 16-token blocks) is bit
    /// identical to the one-token kernel run token by token.
    @Test func chunkControllerApplyIsBitIdentical() throws {
        let context = try MetalContext()
        let ops = try InklingQ4Ops(context: context)
        let device = context.device
        let (width, rank, tokens) = (2_304, 64, 37)
        func tensor(_ rows: Int, _ columns: Int, _ seed: Int) throws -> InklingQ4Tensor {
            let v = Self.activation(rows * columns, seed: seed).map { $0 * 0.07 }
            return InklingQ4Tensor(
                operand: .init(try Self.buffer(device, v)), type: .f32, rows: rows,
                columns: columns, byteCount: v.count * 4)
        }
        let gain = try tensor(width, rank, 2)
        let forget = try tensor(width, rank, 3)
        let cur = try Self.buffer(device, Self.activation(tokens * width, seed: 4))
        let message = try Self.buffer(device, Self.activation(tokens * width, seed: 5))
        let h = try Self.buffer(device, Self.activation(tokens * rank, seed: 6).map { $0 * 4 })
        let chunk = try #require(device.makeBuffer(length: tokens * width * 4))
        let step = try #require(device.makeBuffer(length: tokens * width * 4))
        try Self.run(context) { encoder in
            try ops.controlApply(
                cur: .init(cur), message: .init(message), h: .init(h), gain: gain,
                forget: forget, out: .init(chunk), width: width, tokens: tokens, encoder: encoder)
            for t in 0..<tokens {
                try ops.controlApply(
                    cur: .init(cur, offset: t * width * 4),
                    message: .init(message, offset: t * width * 4),
                    h: .init(h, offset: t * rank * 4), gain: gain, forget: forget,
                    out: .init(step, offset: t * width * 4), width: width, tokens: 1,
                    encoder: encoder)
            }
        }
        #expect(Self.floats(chunk, tokens * width) == Self.floats(step, tokens * width))
    }

    /// The decode kernel indexes threadgroup memory as `simd_group * 128 + d`,
    /// so a head dimension over 128 writes past its array. The converter only
    /// requires a positive `attention.key_length`, so the refusal has to be
    /// here (AUD-0091). The encoder is ended by hand: `Self.run` would leave it
    /// open when the body throws, which Metal aborts on.
    @Test func aDecodeHeadDimensionOver128IsRefused() throws {
        let context = try MetalContext()
        let ops = try InklingQ4Ops(context: context)
        let device = context.device
        let dim = 192
        let args = InklingQ4Ops.AttentionArgs(
            heads: 6, kvHeads: 2, headDim: UInt32(dim), relative: 16, extent: 20, capacity: 8,
            ring: 0, scale: 1 / Float(dim).squareRoot(), window: 1 << 30, tokens: 1)
        let out = try #require(device.makeBuffer(length: 6 * dim * 4))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        #expect(throws: MetalError.self) {
            try ops.attention(
                q: .init(try Self.buffer(device, Self.activation(6 * dim, seed: 2))),
                kCache: try Self.buffer(device, [Float16](repeating: 0, count: 8 * 2 * dim)),
                vCache: try Self.buffer(device, [Float16](repeating: 0, count: 8 * 2 * dim)),
                r: .init(try Self.buffer(device, Self.activation(dim * 16, seed: 3))),
                relativeBias: InklingQ4Tensor(
                    operand: .init(try Self.buffer(device, Self.activation(20 * 16, seed: 4))),
                    type: .f32, rows: 20, columns: 16, byteCount: 20 * 16 * 4),
                sinks: .init(try Self.buffer(device, Self.activation(6, seed: 5))),
                gate: .init(try Self.buffer(device, Self.activation(6 * dim, seed: 6))),
                out: .init(out),
                step: .init(try Self.buffer(device, [UInt32(0)])),
                args: args,
                encoder: encoder)
        }
        encoder.endEncoding()
    }

    /// `sigmoid_topk` writes 0xFFFFFFFF when it cannot fill the top-k, which the
    /// streamed path reads as expert -1 (AUD-0094); the quantized KV row layout
    /// divides the head dimension into 32/16-channel groups (AUD-0092). Both are
    /// refused at the encoder, where the geometry is known.
    @Test func impossibleRouterAndKvGeometryAreRefused() throws {
        let context = try MetalContext()
        let ops = try InklingQ4Ops(context: context)
        let device = context.device
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        let x = try Self.buffer(device, Self.activation(8, seed: 1))
        let logits = try Self.buffer(device, Self.activation(8 * 4, seed: 2))
        let ids = try Self.buffer(device, [UInt32](repeating: 0, count: 8))
        let weights = try Self.buffer(device, [Float](repeating: 0, count: 8))
        let router = InklingQ4Tensor(
            operand: .init(try Self.buffer(device, [UInt8](repeating: 0, count: 8))),
            type: .q8_0, rows: 4, columns: 8, byteCount: 8)

        #expect(throws: MetalError.self) {
            try ops.route(
                router: router, bias: router, x: .init(x), logits: .init(logits),
                ids: .init(ids), weights: .init(weights),
                args: InklingQ4Ops.RouterArgs(
                    experts: 4, width: 8, used: 8, weightScale: 1, normEpsilon: 1e-6),
                tokens: 1, encoder: encoder)
        }
        #expect(throws: MetalError.self) {
            try ops.kvFinish(
                k: .init(x), v: .init(x), kWeight: router, vWeight: router,
                state: .init(x), dynamicK: nil, dynamicV: nil,
                kCache: logits, vCache: logits, step: .init(x),
                args: InklingQ4Ops.KVFinishArgs(
                    headDim: 40, kvHeads: 2, hasDynamic: 0, capacity: 8, ring: 0,
                    threshold: 0, epsilon: 1e-6, scale: 1, tokens: 1),
                precision: .int8, encoder: encoder)
        }
        encoder.endEncoding()
    }
}
