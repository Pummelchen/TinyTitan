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

/// Which arms a `runMoE` mode dispatches, and the bytes it charges for them.
///
/// `moe_phase1_subset` starts with `moe_phase1`, and the prefix test used to be
/// the whole rule: the subset arm dispatched the subset kernel *and* the
/// production phase-1 kernel every iteration, then reported phase-1's bytes as
/// the subset's rate — a mode whose label describes half its work (AUD-0113).
@Suite struct MoEBenchPlanTests {
    private func plan(_ name: String, topK: UInt32 = 8) -> TinyTitanBench.MoEMeasurementPlan {
        TinyTitanBench.MoEMeasurementPlan.make(
            kernelName: name, topK: topK, D: 2048, F: 512, activeCount: topK)
    }

    @Test func theCombinedModeRunsBothPhases() {
        let p = plan("moe")
        #expect(p.runsPhase1 && p.runsPhase2 && !p.runsSubset)
        #expect(p.bytes == 8 * 512 * 2048 + 8 * 512 * 2048 / 2)
    }

    @Test func aPhaseModeRunsOnlyItsPhase() {
        let p1 = plan("moe_phase1_r8")
        #expect(p1.runsPhase1 && !p1.runsPhase2 && !p1.runsSubset)
        #expect(p1.bytes == 8 * 512 * 2048)
        let p2 = plan("moe_phase2")
        #expect(!p2.runsPhase1 && p2.runsPhase2 && !p2.runsSubset)
        #expect(p2.bytes == 8 * 512 * 2048 / 2)
    }

    @Test func theSubsetModeDoesNotAlsoRunTheProductionKernel() {
        let p = plan("moe_phase1_subset")
        #expect(p.runsSubset)
        #expect(!p.runsPhase1, "the subset arm dispatched the production kernel too")
        #expect(!p.runsPhase2)
        #expect(p.bytes == 8 * 512 * 2048)
    }

    @Test func theSubsetModeChargesOnlyItsActiveExperts() {
        let p = TinyTitanBench.MoEMeasurementPlan.make(
            kernelName: "moe_phase1_subset", topK: 8, D: 2048, F: 512, activeCount: 3)
        #expect(p.bytes == 3 * 512 * 2048)
    }
}
