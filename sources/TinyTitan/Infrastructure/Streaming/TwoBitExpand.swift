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
