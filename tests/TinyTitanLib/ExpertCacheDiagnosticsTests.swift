import Foundation
import Testing

@testable import TinyTitanLib

/// The `ram profile` / `ram target` lines are the operator's only record of what
/// the engine allocated, so they must name the rung in effect rather than the
/// rung the derivation preferred.
///
/// Geometry used throughout: 0.125 GiB per slot, 3.00 GiB resident floor, so 8
/// slots is `cache=1.00G` and 16 is `cache=2.00G`.
@Suite
struct ExpertCacheDiagnosticsTests {
    private static let slotBytes = 134_217_728.0
    private static let residentFloor = 3_221_225_472

    @Test func profileLineNamesTheAppliedSlotsWhenAnOverrideIsInEffect() {
        let line = ExpertCacheDiagnostics.profileLine(
            appliedSlots: 8,
            derivedSlots: 16,
            cacheBytesPerSlot: Self.slotBytes,
            residentFloorBytes: Self.residentFloor)

        #expect(
            line == """
                TinyTitan ram profile cache=1.00G slots=8 derived_slots=16 \
                resident_floor=3.00G estimate=4.00G \
                (cache budget, not a process target)
                """)
    }

    @Test func profileLineIsUnchangedWhenNothingOverridesTheDerivation() {
        let line = ExpertCacheDiagnostics.profileLine(
            appliedSlots: 16,
            derivedSlots: 16,
            cacheBytesPerSlot: Self.slotBytes,
            residentFloorBytes: Self.residentFloor)

        #expect(
            line == """
                TinyTitan ram profile cache=2.00G slots=16 resident_floor=3.00G \
                estimate=5.00G (cache budget, not a process target)
                """)
    }

    @Test func targetLineNamesTheAppliedSlotsWhenAnOverrideIsInEffect() {
        let lines = ExpertCacheDiagnostics.targetLines(
            targetBytes: 6_442_450_944,
            appliedSlots: 8,
            derivedSlots: 16,
            cacheBytesPerSlot: Self.slotBytes,
            residentFloorBytes: Self.residentFloor)

        #expect(
            lines == [
                """
                TinyTitan ram target=6.00G cache=1.00G slots=8 derived_slots=16 \
                resident_floor=3.00G estimate=4.00G
                """
            ])
    }

    @Test func targetLineWarnsWhenTheAppliedCacheStillExceedsTheNamedTarget() {
        let lines = ExpertCacheDiagnostics.targetLines(
            targetBytes: 3_758_096_384,
            appliedSlots: 8,
            derivedSlots: 16,
            cacheBytesPerSlot: Self.slotBytes,
            residentFloorBytes: Self.residentFloor)

        #expect(lines.count == 2)
        #expect(
            lines[1] == """
                TinyTitan ram warning: 3.50G is below this install's 4.00G floor \
                (3.00G resident + the 8-slot minimum cache); \
                the cache is already at its smallest.
                """)
    }

    @Test func noWarningWhenTheAppliedFootprintFitsTheNamedTarget() {
        let lines = ExpertCacheDiagnostics.targetLines(
            targetBytes: 6_442_450_944,
            appliedSlots: 16,
            derivedSlots: 16,
            cacheBytesPerSlot: Self.slotBytes,
            residentFloorBytes: Self.residentFloor)

        #expect(lines.count == 1)
        #expect(!lines[0].contains("derived_slots"))
    }
}
