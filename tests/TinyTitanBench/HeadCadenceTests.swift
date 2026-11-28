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

/// The head benchmark's name table: a typo must not measure the default arm.
///
/// `headKernel(named:)` ended in a `default` that picked `head_affine8`, so any
/// misspelt `head_*` name recorded a real measurement under a mode that does not
/// exist — the "a control that is really a measurement" shape the flag reader
/// refuses elsewhere (AUD-0114).
@Suite struct HeadKernelSelectionTests {
    @Test func knownNamesSelectTheirKernelAndRows() throws {
        let affine8 = try TinyTitanBench.headKernel(named: "head_affine8")
        #expect(affine8.bits == 8 && affine8.rowsPerLaunch == 1)
        #expect(affine8.kernel == "affine_quant_gemv_simd")
        let affine4Two = try TinyTitanBench.headKernel(named: "head_affine4_two_rows")
        #expect(affine4Two.bits == 4 && affine4Two.rowsPerLaunch == 2)
        #expect(affine4Two.kernel == "affine_quant_gemv2_simd")
        let int4 = try TinyTitanBench.headKernel(named: "head_int4")
        #expect(int4.kernel == "dequant_int4_gemv_simd")
    }

    @Test func anUnknownNameIsRefusedByName() throws {
        #expect(throws: BenchHarnessError.self) {
            _ = try TinyTitanBench.headKernel(named: "head_affine8x")
        }
        do {
            _ = try TinyTitanBench.headKernel(named: "head_affine8x")
            Issue.record("an unknown head name selected a kernel")
        } catch {
            #expect(String(describing: error).contains("head_affine8x"))
            #expect(String(describing: error).contains("unknown head benchmark"))
        }
    }

    @Test func theInt4FourRowProbeIsStillRefused() {
        #expect(throws: BenchHarnessError.self) {
            _ = try TinyTitanBench.headKernel(named: "head_int4_four_rows")
        }
    }
}

/// The `gdn_chain` arm's head geometry follows the selected bench shape.
///
/// The dispatches carried qwen36's literal counts (16 key, 32 value heads)
/// while the buffers followed `TINYTITAN_BENCH_GDN_SHAPE`, so the qwen38 shape
/// ran a 32-head delta step over 48-head buffers — a number for a shape the
/// model does not have (AUD-0116).
@Suite struct GDNChainShapeTests {
    @Test func theQwen3ShapeKeepsItsThirtyTwoValueHeads() {
        let shape = TinyTitanBench.GDNChainShape.forBenchShape(qwen38: false)
        #expect(
            shape
                == TinyTitanBench.GDNChainShape(
                    kHeads: 16, vHeads: 32, keyDim: 128, valueDim: 128))
        #expect(shape.stateElements == 32 * 128 * 128)
    }

    @Test func theQwen38ShapeHasFortyEightValueHeads() {
        let shape = TinyTitanBench.GDNChainShape.forBenchShape(qwen38: true)
        #expect(shape.vHeads == 48)
        #expect(shape.stateElements == 48 * 128 * 128)
        // The buffer the arm allocates is zRows (48 * 128) halves, so the head
        // count and the shape have to agree.
        #expect(shape.vHeads * shape.valueDim == 6144)
    }
}
