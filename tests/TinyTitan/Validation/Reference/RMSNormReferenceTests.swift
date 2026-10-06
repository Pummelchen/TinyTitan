import Foundation
import Testing
import TinyTitanValidationSupport

/// Cross-validates `RmsNormRef.apply` (Accelerate-vector form) against a
/// second formulation written from scratch as a scalar loop. The point isn't
/// to prove the kernel right — it's to prove the *reference* right, so the
/// kernel test below has a trustworthy comparator.
///
/// The scalar formulation shares no code path with Accelerate. Both must
/// agree to `Tolerance.identity`.
///
/// Mismatched `x`/`weight` lengths are a caller bug and both formulations
/// trap on them. A trap cannot be observed from a test body — it ends the
/// process, and Swift Testing has no way to catch it — so the contract is
/// written down here rather than asserted by a test that cannot fail.
@Suite struct RMSNormReferenceTests {

    /// Independent scalar reference. Different summation order than both
    /// the kernel and the Accelerate reference.
    private static func scalarRef(x: [Float], weight: [Float], eps: Float) -> [Float] {
        // `zip` would silently truncate to the shorter input, so the
        // comparator used to disagree with the function it validates about
        // what a mismatch means. Same contract, same trap.
        precondition(x.count == weight.count, "x and weight must match length")
        let d = x.count
        var sumSq: Float = 0
        for v in x { sumSq += v * v }
        let inv = 1.0 / (sumSq / Float(d) + eps).squareRoot()
        return zip(x, weight).map { $0 * inv * $1 }
    }

    @Test(
        "Accelerate ref matches scalar ref",
        arguments: [
            (256, UInt64(0xA1)),
            (512, UInt64(0xB2)),
            (2816, UInt64(0xC3)),
        ])
    func acceleratedMatchesScalar(d: Int, seed: UInt64) {
        var rng = SeedTree(seed).key("rmsnorm-ref")
        let x = (0..<d).map { _ in rng.uniform(-1.0, 1.0) }
        let w = (0..<d).map { _ in rng.uniform(0.5, 1.5) }
        let eps: Float = 1e-6

        let accel = RmsNormRef.apply(x: x, weight: w, eps: eps)
        let scalar = Self.scalarRef(x: x, weight: w, eps: eps)

        let relErr = RelError.compute(actual: accel, reference: scalar)
        #expect(
            relErr < Tolerance.identity,
            "D=\(d): relErr=\(relErr)")
    }

    @Test("Reference handles zero-mean small magnitudes")
    func smallMagnitudes() {
        let x: [Float] = [1e-4, -1e-4, 2e-4, -2e-4, 1e-4, -1e-4, 2e-4, -2e-4]
        let w: [Float] = Array(repeating: 1.0, count: x.count)
        let y = RmsNormRef.apply(x: x, weight: w, eps: 1e-6)
        #expect(y.count == x.count)
        for v in y { #expect(v.isFinite) }
    }

    @Test("The empty input returns no output, not a crash")
    func emptyInput() {
        // `apply` has an explicit guard here: it used to reach a force-unwrap
        // on a zero-count buffer's base address.
        #expect(RmsNormRef.apply(x: [], weight: [], eps: 1e-6).isEmpty)
    }
}
