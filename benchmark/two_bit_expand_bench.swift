// Measures the cold-tier expansion step: 2-bit packed weights to 4-bit packed
// (value v in 0..3 -> 5v), the transform that lets a tiered install keep the
// existing 4-bit kernels. Build and run with `swiftc -O` and no arguments.
//
// Measured 2026-10-20 on the 24 GiB M3: 13.68 GB/s on one thread, 28.13 GB/s on
// four, i.e. 0.27 ms per token for ~3.2 expert misses. That is 0.6 % of a
// 45 ms token against an 11-13 ms byte saving, which is why the expansion path
// replaced the dual-width dispatch.

import Foundation

// Expand 2-bit packed weights to 4-bit packed: value v in 0..3 -> 5v in 0..15,
// two 4-bit values per byte. This is the transform the cold tier would run on
// each expert after its read.
@inline(__always)
func expand(_ src: UnsafePointer<UInt8>, _ dst: UnsafeMutablePointer<UInt8>, _ bytes: Int) {
    for i in 0..<bytes {
        let b = src[i]
        let v0 = UInt8((b >> 6) & 0x3) * 5
        let v1 = UInt8((b >> 4) & 0x3) * 5
        let v2 = UInt8((b >> 2) & 0x3) * 5
        let v3 = UInt8(b & 0x3) * 5
        dst[2 * i] = (v0 << 4) | v1
        dst[2 * i + 1] = (v2 << 4) | v3
    }
}

let values = 3 * 512 * 2048  // gate/up/down of one routed expert
let inBytes = values / 4
let outBytes = values / 2
let src = UnsafeMutablePointer<UInt8>.allocate(capacity: inBytes)
let dst = UnsafeMutablePointer<UInt8>.allocate(capacity: outBytes)
for i in 0..<inBytes { src[i] = UInt8(truncatingIfNeeded: i &* 37 &+ 11) }

for threads in [1, 2, 4, 8] {
    let chunk = (inBytes + threads - 1) / threads
    // warm
    expand(src, dst, inBytes)
    var best = Double.infinity
    for _ in 0..<5 {
        let t0 = DispatchTime.now().uptimeNanoseconds
        DispatchQueue.concurrentPerform(iterations: threads) { t in
            let start = t * chunk
            let end = min(inBytes, start + chunk)
            if start < end {
                expand(src + start, dst + start * 2, end - start)
            }
        }
        let ns = Double(DispatchTime.now().uptimeNanoseconds - t0)
        best = min(best, ns)
    }
    let moved = Double(inBytes + outBytes)
    let gbs = moved / (best / 1e9) / 1e9
    print(
        String(
            format: "threads %d  %.3f ms  %.2f GB/s  per-token(3.2 experts) %.2f ms",
            threads, best / 1e6, gbs, best / 1e6 * 3.2))
}
src.deallocate(); dst.deallocate()
