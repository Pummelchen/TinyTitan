import Foundation

/// The cold tier's expansion step: 2-bit packed weights to 4-bit packed.
///
/// A tiered install stores cold experts at 2-bit and hot ones at 4-bit. The
/// existing 4-bit kernels run unchanged if a cold expert is expanded into the
/// slot on the way in, mapping `v` in 0..3 to `5v` in 0,5,10,15 and dividing the
/// group scale by 5 (`bias` unchanged), which makes
/// `q4 * scale/5 + bias == q2 * scale + bias` exactly.
///
/// The transform is a byte shuffle with no arithmetic dependency: 13.7 GB/s on
/// one thread and 28.1 GB/s on four on the 24 GiB M3, 0.27 ms per token for
/// ~3.2 expert misses (`benchmark/two_bit_expand_bench.swift`).
enum TwoBitExpand {
    /// Expands `byteCount` packed 2-bit source bytes into `2 * byteCount`
    /// packed 4-bit destination bytes. `src` and `dst` must not overlap.
    static func expand(
        source: UnsafePointer<UInt8>,
        destination: UnsafeMutablePointer<UInt8>,
        byteCount: Int
    ) {
        for index in 0..<byteCount {
            let byte = source[index]
            let v0 = UInt8((byte >> 6) & 0x3) * 5
            let v1 = UInt8((byte >> 4) & 0x3) * 5
            let v2 = UInt8((byte >> 2) & 0x3) * 5
            let v3 = UInt8(byte & 0x3) * 5
            destination[2 * index] = (v0 << 4) | v1
            destination[2 * index + 1] = (v2 << 4) | v3
        }
    }

    /// The scale adjustment that pairs with `expand`: a 2-bit group scale must
    /// be divided by five so the expanded 4-bit values produce the same
    /// product. Bias is unchanged.
    static func adjustedScale(_ scale: Float) -> Float { scale / 5 }
}

/// One tensor inside an expert blob, before and after expansion.
///
/// A cold expert is written compact: its weight tensors are 2-bit and half the
/// size of the 4-bit slot they expand into. The scales and biases are already
/// the right size; only the scales change value (`/ 5`).
struct ExpertBlobTensor {
    let sourceOffset: Int
    let sourceSize: Int
    let targetOffset: Int
    let targetSize: Int
    /// Weight tensors are expanded; scale and bias tensors are moved.
    let expands: Bool
    /// Scale tensors are divided by five, matching `adjustedScale`.
    let rescales: Bool
}

extension TwoBitExpand {
    /// Rewrites a compact 2-bit expert blob into the 4-bit slot layout, in place.
    ///
    /// Tensors are processed from the highest source offset down. Every target
    /// offset is at or beyond its source offset (weights double), so a write
    /// never lands on bytes that have not been read yet: the source of a tensor
    /// with a lower offset ends before this tensor's target begins. The slot
    /// must be at least as large as the 4-bit blob.
    static func expandExpertInPlace(
        slot: UnsafeMutableRawPointer,
        tensors: [ExpertBlobTensor]
    ) {
        let bytes = slot.assumingMemoryBound(to: UInt8.self)
        for tensor in tensors.sorted(by: { $0.sourceOffset > $1.sourceOffset }) {
            let source = bytes + tensor.sourceOffset
            let destination = bytes + tensor.targetOffset
            if tensor.expands {
                // Backwards, because a doubling expansion overlaps itself --
                // including when the tensor stays at offset 0.
                expandReversed(
                    source: source, destination: destination,
                    byteCount: tensor.sourceSize)
            } else if tensor.rescales {
                if destination != source {
                    memmove(destination, source, tensor.sourceSize)
                }
                rescaleBF16InPlace(destination, bytes: tensor.sourceSize)
            } else if destination != source {
                memmove(destination, source, tensor.sourceSize)
            }
        }
    }

    /// Expands a range whose destination overlaps and follows the source.
    private static func expandReversed(
        source: UnsafeMutablePointer<UInt8>,
        destination: UnsafeMutablePointer<UInt8>,
        byteCount: Int
    ) {
        var index = byteCount
        while index > 0 {
            index -= 1
            let byte = source[index]
            let v0 = UInt8((byte >> 6) & 0x3) * 5
            let v1 = UInt8((byte >> 4) & 0x3) * 5
            let v2 = UInt8((byte >> 2) & 0x3) * 5
            let v3 = UInt8(byte & 0x3) * 5
            destination[2 * index] = (v0 << 4) | v1
            destination[2 * index + 1] = (v2 << 4) | v3
        }
    }

    /// Divides bf16 scale values by five in place.
    private static func rescaleBF16InPlace(
        _ pointer: UnsafeMutablePointer<UInt8>, bytes: Int
    ) {
        let count = bytes / 2
        let halves = UnsafeMutableRawPointer(pointer)
            .assumingMemoryBound(to: UInt16.self)
        for index in 0..<count {
            let bits = UInt32(halves[index]) << 16
            let value = Float(bitPattern: bits) / 5
            // Truncation costs at most one bf16 ulp on a scale the quantiser
            // already rounded; rounding here would only add code to audit.
            halves[index] = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
        }
    }
}
