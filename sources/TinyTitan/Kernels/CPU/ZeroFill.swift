import TinyTitanKernelsC

/// Experimental: `DC ZVA`-backed zero fill.
///
/// `DC ZVA` zeroes one cache block, which on Apple M-series is **64 bytes** —
/// not the 128-byte line `hw.cachelinesize` reports. A caller that strides by
/// the advertised line size leaves every other block dirty, which is the trap
/// this wrapper exists to keep in one place.
///
/// The C side falls back to `memset` for a misaligned base or a range shorter
/// than 256 bytes; `DC ZVA` rounds an unaligned address down and would zero
/// bytes before the buffer.
public enum ZeroFill {
    /// Zeroes `count` bytes at `base`. Does nothing when `count` is zero.
    public static func zero(_ base: UnsafeMutableRawPointer, count: Int) {
        guard count > 0 else { return }
        tinytitan_zero_fill(base, count)
    }
}
