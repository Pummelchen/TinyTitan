import Testing

@testable import TinyTitanBench

/// The cadence arm's gap: a number the operator types, not yet a duration.
///
/// `envIntFlag` only proves the value parses. A negative
/// `TINYTITAN_BENCH_HEAD_CADENCE_MS` used to reach
/// `Thread.sleep(forTimeInterval:)` — documented for a non-negative interval —
/// so the arm either trapped part-way through a measurement or slept nothing
/// and reported back-to-back timings under a cadence label (AUD-0112).
@Suite struct HeadCadenceTests {
    @Test func aNonNegativeCadenceBecomesSeconds() throws {
        #expect(try TinyTitanBench.cadenceSeconds(0) == 0)
        #expect(try TinyTitanBench.cadenceSeconds(50) == 0.05)
        #expect(try TinyTitanBench.cadenceSeconds(20_000) == 20)
    }

    @Test func aNegativeCadenceIsRefusedByName() {
        #expect(throws: BenchHarnessError.self) {
            _ = try TinyTitanBench.cadenceSeconds(-5)
        }
        do {
            _ = try TinyTitanBench.cadenceSeconds(-5)
            Issue.record("a negative cadence was accepted")
        } catch {
            #expect(String(describing: error).contains("TINYTITAN_BENCH_HEAD_CADENCE_MS"))
            #expect(String(describing: error).contains("-5"))
        }
    }
}
