import TinyTitanKernelsC

/// Experimental: Apple's undocumented AMX matrix coprocessor.
///
/// Nothing in the engine calls this. The CPU engine deliberately has no
/// batched prefill (`CPUQwen35`), so there is no GEMM caller yet; this exists
/// so the kernel can be validated against a reference and, if a batched CPU
/// path is ever added, wired behind a measured flag on M1–M3.
///
/// AMX is per-thread state: `AMX_SET` faults if the thread already has it
/// enabled. Do not call this from a thread that is inside an Accelerate call.
public enum AMXKernel {
    /// Whether the AMX instructions executed on this core when probed.
    ///
    /// Cached for the process lifetime by the C probe, which runs `AMX_SET` /
    /// `AMX_CLR` behind a `SIGILL` handler once.
    public static var isAvailable: Bool {
        tinytitan_amx_available() != 0
    }

    /// `c = a * b` for row-major fp32, with `m` and `n` multiples of 16.
    ///
    /// Returns `nil` when AMX is unavailable, a dimension is unsupported, or
    /// the shapes overflow. `k` may be zero, which yields a zero matrix.
    ///
    /// The transpose into AMX's expected layout is part of the experiment: a
    /// real integration would pre-pack weights once rather than per call.
    public static func multiply(
        _ a: [Float],
        _ b: [Float],
        m: Int,
        n: Int,
        k: Int
    ) -> [Float]? {
        guard m > 0, n > 0, k >= 0, m % 16 == 0, n % 16 == 0 else {
            return nil
        }
        let (elementCount, countOverflow) = m.multipliedReportingOverflow(by: k)
        let (bElementCount, bOverflow) = k.multipliedReportingOverflow(by: n)
        let (outputCount, outputOverflow) = m.multipliedReportingOverflow(by: n)
        guard
            !countOverflow, !bOverflow, !outputOverflow,
            a.count >= elementCount, b.count >= bElementCount,
            isAvailable
        else {
            return nil
        }

        // AMX loads a column of A as one 64-byte vector, so A is transposed
        // once here: `at[k * m + row]`.
        var transposed = [Float](repeating: 0, count: elementCount)
        if k > 0 {
            for row in 0..<m {
                let source = row * k
                for column in 0..<k {
                    transposed[column * m + row] = a[source + column]
                }
            }
        }

        var result = [Float](repeating: 0, count: outputCount)
        let status = transposed.withUnsafeBufferPointer { transposedBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                result.withUnsafeMutableBufferPointer { resultBuffer in
                    tinytitan_amx_sgemm16(
                        transposedBuffer.baseAddress,
                        m,
                        bBuffer.baseAddress,
                        n,
                        resultBuffer.baseAddress,
                        n,
                        m,
                        n,
                        k)
                }
            }
        }
        return status == 0 ? result : nil
    }
}
