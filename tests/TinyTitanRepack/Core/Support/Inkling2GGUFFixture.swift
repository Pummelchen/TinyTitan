import Foundation
import TinyTitanFormat

/// A metadata-only GGUF v3 header for the Inkling 2 architecture reader.
///
/// These tests are about what the reader accepts, not about tensor bytes, so
/// the fixture writes valid keys and values and no tensors. `valid()` is a
/// complete, small Inkling 2 metadata set that satisfies
/// `Inkling2GGUFArch.archInfo`; a test overrides one key to make a negative
/// case.
struct Inkling2GGUFFixture {
    enum Value {
        case u32(UInt32)
        case u64(UInt64)
        case f64(Double)
        case string(String)
        case f64Array([Double])
    }

    private var entries: [(String, Value)] = []
    private var tensors: [(String, [UInt64])] = []

    mutating func set(_ key: String, _ value: Value) {
        if let index = entries.firstIndex(where: { $0.0 == key }) {
            entries[index].1 = value
        } else {
            entries.append((key, value))
        }
    }

    mutating func tensor(_ name: String, dimensions: [UInt64]) {
        tensors.append((name, dimensions))
    }

    func bytes() -> [UInt8] {
        var out: [UInt8] = []
        func u32(_ value: UInt32) {
            out += withUnsafeBytes(of: value.littleEndian, Array.init)
        }
        func u64(_ value: UInt64) {
            out += withUnsafeBytes(of: value.littleEndian, Array.init)
        }
        func string(_ value: String) {
            u64(UInt64(value.utf8.count))
            out += Array(value.utf8)
        }
        out += Array("GGUF".utf8)
        u32(3)
        u64(UInt64(tensors.count))
        u64(UInt64(entries.count))
        for (key, value) in entries {
            string(key)
            switch value {
            case .u32(let value):
                u32(4)
                u32(value)
            case .u64(let value):
                u32(10)
                u64(value)
            case .f64(let value):
                u32(12)
                u64(value.bitPattern.littleEndian)
            case .string(let value):
                u32(8)
                string(value)
            case .f64Array(let values):
                u32(9)
                u32(12)  // f64 elements
                u64(UInt64(values.count))
                for value in values { u64(value.bitPattern.littleEndian) }
            }
        }
        for (name, dimensions) in tensors {
            string(name)
            u32(UInt32(dimensions.count))
            for dimension in dimensions { u64(dimension) }
            u32(0)  // ggml F32; the architecture reader does not look at it
            u64(0)  // data offset
        }
        return out
    }

    /// A complete, valid Inkling 2 metadata set: two stored layers, a two-layer
    /// loop repeated twice (four visits), two experts, head dimension 8.
    static func valid() -> Inkling2GGUFFixture {
        var fixture = Inkling2GGUFFixture()
        fixture.set("general.architecture", .string("inkling2"))
        fixture.set("inkling2.rope.dimension_count", .u32(0))
        fixture.set("inkling2.loop.core_start", .u32(0))
        fixture.set("inkling2.loop.core_end", .u32(2))
        fixture.set("inkling2.num_loops", .u32(2))
        fixture.set("inkling2.block_count", .u32(4))
        fixture.set("inkling2.embedding_length", .u32(64))
        fixture.set("inkling2.feed_forward_length", .u32(32))
        fixture.set("inkling2.expert_feed_forward_length", .u32(32))
        fixture.set("inkling2.attention.head_count", .u32(2))
        fixture.set("inkling2.attention.head_count_kv", .u32(2))
        fixture.set("inkling2.attention.key_length", .u32(8))
        fixture.set("inkling2.vocab_size", .u32(128))
        fixture.set("inkling2.vocab_size_unpadded", .u32(128))
        fixture.set("inkling2.attention.sliding_window", .u32(16))
        fixture.set("inkling2.final_logit_softcapping", .f64(30))
        fixture.set("inkling2.expert_count", .u32(2))
        fixture.set("inkling2.expert_used_count", .u32(2))
        fixture.set("inkling2.attention.sliding_window_pattern", .f64Array([0, 1, 0, 1]))
        fixture.set("inkling2.loop.gate_rank", .f64(4))
        fixture.set("inkling2.attn_res.block_size", .f64(2))
        fixture.set("inkling2.attention.relative_length", .f64(8))
        fixture.set("inkling2.attention.relative_extent", .f64(16))
        fixture.set("inkling2.attention.relative_clip", .f64(8))
        fixture.set("inkling2.shortconv.l_cache", .f64(4))
        fixture.set("inkling2.attention.layer_norm_rms_epsilon", .f64(1e-6))
        fixture.set("inkling2.attention.qk_clip", .f64(20))
        fixture.set("inkling2.attention.log_scale_floor", .f64(8_192))
        fixture.set("inkling2.attention.log_scale_alpha", .f64(0.1))
        fixture.set("inkling2.expert_weights_scale", .f64(8))
        fixture.set("inkling2.activation.situ_beta", .f64(4))
        fixture.set("inkling2.activation.situ_linear_beta", .f64(25))
        fixture.set("inkling2.context_length", .f64(1_024))
        return fixture
    }
}
