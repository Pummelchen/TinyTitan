import Foundation
import Testing
import TinyTitanCore

@testable import Inkling2NVFP4

/// The NVFP4 install's window plan: the host's 262,144-token default is not
/// this model's choice.
///
/// With no `--ram-budget` and no explicit `--max-context`, the old rule used the
/// request verbatim. The doubled window took the memory the expert cache needed,
/// and this install measured 3.6 tok/s against 8.1 at 128K (AUD-0121).
struct InklingNVFP4LoadPlanTests {
    static let geometry = InklingQ4LoadPlan.Geometry(
        residentBytes: 1_819_979_776, bytesPerSlot: 20 * 4_489_216 + 4 * 8_470_528,
        bytesPerPosition: 32_768, experts: 128, expertsPerToken: 8, modelContext: 1_048_576)
    static let workingSet = 17 << 30

    @Test func anImplicitWindowWithoutABudgetCapsAtOneTwentyEightK() throws {
        let plan = try InklingQ4LoadPlan.make(
            geometry: Self.geometry, requested: 262_144, ramBudget: nil,
            explicitSlots: nil, workingSet: Self.workingSet, maxContextExplicit: false)
        #expect(plan.context == InklingQ4LoadPlan.defaultBudgetedContext)
    }

    @Test func anExplicitWindowIsHonoured() throws {
        let plan = try InklingQ4LoadPlan.make(
            geometry: Self.geometry, requested: 262_144, ramBudget: nil,
            explicitSlots: nil, workingSet: Self.workingSet, maxContextExplicit: true)
        #expect(plan.context == 262_144)
    }
}
