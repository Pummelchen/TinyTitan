import Foundation
import Testing

@testable import TinyTitanLib
@testable import TinyTitanCore

/// The batched-serving headroom rule: a width clamped so the requested slots
/// fit beside the wired expert cache.
@Suite struct BatchedMemoryBudgetTests {

    @Test func widthIsClampedToTheBudgetButNeverBelowOne() {
        #expect(
            BatchedMemoryBudget.effectiveSlots(
                requested: 4, perSlotBytes: 100, budgetBytes: 300) == 3)
        #expect(
            BatchedMemoryBudget.effectiveSlots(
                requested: 4, perSlotBytes: 100, budgetBytes: 1_000) == 4)
        #expect(
            BatchedMemoryBudget.effectiveSlots(
                requested: 4, perSlotBytes: 100, budgetBytes: 0) == 1,
            "a model too large for a second slot still serves one")
        #expect(
            BatchedMemoryBudget.effectiveSlots(
                requested: 1, perSlotBytes: 100, budgetBytes: 0) == 1)
        #expect(
            BatchedMemoryBudget.effectiveSlots(
                requested: 4, perSlotBytes: 0, budgetBytes: 1_000) == 4,
            "an unknown size must not silently pin the width to one")
    }

    @Test func budgetLeavesTheWiredExpertCacheOutOfTheKVHeadroom() {
        let physical: UInt64 = 24 * 1_073_741_824
        #expect(
            BatchedMemoryBudget.slotBudgetBytes(
                physicalMemory: physical, expertCacheBudgetBytes: 0) == Int(physical / 2))
        #expect(
            BatchedMemoryBudget.slotBudgetBytes(
                physicalMemory: physical, expertCacheBudgetBytes: 4 * 1_073_741_824)
                == Int(physical / 2) - 4 * 1_073_741_824)
        #expect(
            BatchedMemoryBudget.slotBudgetBytes(
                physicalMemory: physical, expertCacheBudgetBytes: Int(physical)) == 0)
    }

    /// A dense family allocates no expert cache, so its profile's budget for one
    /// must not be taken off the KV headroom. A MoE family's must.
    @Test func aDenseModelHoldsBackNoExpertCache() {
        #expect(
            BatchedMemoryBudget.expertCacheHeldBack(numExperts: 0, configured: 8 * 1_073_741_824)
                == 0)
        #expect(
            BatchedMemoryBudget.expertCacheHeldBack(numExperts: 512, configured: 8 * 1_073_741_824)
                == 8 * 1_073_741_824)
        #expect(BatchedMemoryBudget.expertCacheHeldBack(numExperts: 512, configured: -1) == 0)
    }

    @Test func aPerSequenceStateModelIsServedOneAtATime() {
        #expect(
            BatchedMemoryBudget.effectiveRequestedSlots(
                requested: 4, hasPerSequenceState: true) == 1,
            "QSA/PLE state is not slot-keyed; two sequences would cross-talk")
        #expect(
            BatchedMemoryBudget.effectiveRequestedSlots(
                requested: 4, hasPerSequenceState: false) == 4)
        #expect(
            BatchedMemoryBudget.effectiveRequestedSlots(
                requested: 1, hasPerSequenceState: true) == 1)
    }
}
