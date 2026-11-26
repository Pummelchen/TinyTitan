import Foundation
import Testing

@testable import TinyTitanCore
@testable import TinyTitanLib
@testable import TinyTitanServerCore

/// The stored-response store behind `previous_response_id`, `GET` and `DELETE`.
/// It is bounded by entry count *and* logical retained bytes, and the doc
/// comment states the eviction rule: the oldest entry goes until the new one
/// fits.
@Suite struct ResponseStoreTests {
    private func entry(_ text: String) -> ResponseStore.Entry {
        ResponseStore.Entry(
            responseJSON: Data(text.utf8),
            inputItems: [], outputItems: [], created: Date())
    }

    private func entry(
        jsonBytes: Int, input: Int = 0, output: Int = 0, fill: Character = "a"
    ) -> ResponseStore.Entry {
        ResponseStore.Entry(
            responseJSON: Data(repeating: 0x61, count: jsonBytes),
            inputItems: (0..<input).map { _ in item(String(repeating: fill, count: 400)) },
            outputItems: (0..<output).map { _ in item(String(repeating: fill, count: 400)) },
            created: Date())
    }

    private func item(_ text: String) -> ResponsesAPIRequest.Item {
        ResponsesAPIRequest.Item(
            type: "message", role: "user", content: .string(text))
    }

    @Test func theCapDropsTheOldestEntry() {
        let store = ResponseStore(capacity: 2)
        store.put(id: "a", entry: entry("a"))
        store.put(id: "b", entry: entry("b"))
        store.put(id: "c", entry: entry("c"))

        #expect(store.get("a") == nil)
        #expect(store.get("b") != nil)
        #expect(store.get("c") != nil)
        #expect(store.count == 2)
    }

    /// A re-put is a write: the response just stored is the newest, so it must
    /// not be the one the cap reaches first. Keeping its original position made
    /// the store evict what the caller had just written and keep the entry it
    /// replaced.
    @Test func aRePutRefreshesTheEvictionOrder() {
        let store = ResponseStore(capacity: 2)
        store.put(id: "a", entry: entry("a1"))
        store.put(id: "b", entry: entry("b1"))
        store.put(id: "a", entry: entry("a2"))
        store.put(id: "c", entry: entry("c"))

        #expect(
            store.get("a")?.responseJSON == Data("a2".utf8),
            "the re-put did not replace the entry")
        #expect(store.get("c") != nil)
        #expect(
            store.get("b") == nil,
            "the cap dropped the re-stored entry instead of the oldest one")
        #expect(store.count == 2)
    }

    @Test func deleteRemovesTheEntryAndItsOrderSlot() {
        let store = ResponseStore(capacity: 3)
        store.put(id: "a", entry: entry("a"))
        store.put(id: "b", entry: entry("b"))
        #expect(store.delete("a"))
        #expect(!store.delete("a"))
        store.put(id: "c", entry: entry("c"))
        store.put(id: "d", entry: entry("d"))
        // Capacity 3 with a removed: b, c, d all fit.
        #expect(store.count == 3)
        #expect(store.get("b") != nil)
    }

    // MARK: - Byte budget

    /// The count cap alone let 256 near-limit entries retain an unbounded
    /// history. The byte budget evicts by the same oldest-first rule.
    @Test func theByteBudgetDropsTheOldestEntry() {
        let store = ResponseStore(
            capacity: 10, maximumEntryBytes: 1_000, maximumTotalBytes: 2_500)
        #expect(store.put(id: "a", entry: entry(jsonBytes: 1_000)) == .stored)
        #expect(store.put(id: "b", entry: entry(jsonBytes: 1_000)) == .stored)
        #expect(store.totalRetainedBytes == 2_000)

        #expect(store.put(id: "c", entry: entry(jsonBytes: 1_000)) == .stored)
        #expect(store.get("a") == nil, "the byte budget did not evict the oldest")
        #expect(store.get("b") != nil)
        #expect(store.get("c") != nil)
        #expect(store.totalRetainedBytes == 2_000)
        #expect(store.count == 2)
    }

    @Test func aReplacementAdjustsTheByteAccountingBothWays() {
        let store = ResponseStore(
            capacity: 10, maximumEntryBytes: 3_000, maximumTotalBytes: 10_000)
        store.put(id: "a", entry: entry(jsonBytes: 2_000))
        store.put(id: "b", entry: entry(jsonBytes: 2_000))
        #expect(store.totalRetainedBytes == 4_000)

        store.put(id: "a", entry: entry(jsonBytes: 500))
        #expect(store.get("a")?.responseJSON.count == 500)
        #expect(store.totalRetainedBytes == 2_500, "a smaller replacement kept the old weight")

        store.put(id: "a", entry: entry(jsonBytes: 2_500))
        #expect(store.get("a")?.responseJSON.count == 2_500)
        #expect(store.totalRetainedBytes == 4_500, "a larger replacement was under-counted")
    }

    @Test func anOversizeEntryIsRejectedWithoutChangingTheStore() {
        let store = ResponseStore(
            capacity: 2, maximumEntryBytes: 1_000, maximumTotalBytes: 5_000)
        store.put(id: "a", entry: entry(jsonBytes: 500))
        store.put(id: "b", entry: entry(jsonBytes: 500))

        #expect(store.put(id: "c", entry: entry(jsonBytes: 1_500)) == .rejectedEntryTooLarge)
        #expect(store.get("c") == nil)
        #expect(store.get("a") != nil && store.get("b") != nil)
        #expect(store.count == 2)
        #expect(store.totalRetainedBytes == 1_000, "a refused put changed the accounting")

        // The order was not disturbed either: d's insertion still evicts a.
        store.put(id: "d", entry: entry(jsonBytes: 600))
        #expect(store.get("a") == nil)
        #expect(store.get("b") != nil)
    }

    @Test func aRePutOverTheEntryBudgetLeavesTheOriginal() {
        let store = ResponseStore(
            capacity: 2, maximumEntryBytes: 1_000, maximumTotalBytes: 5_000)
        store.put(id: "a", entry: entry(jsonBytes: 500))
        #expect(store.put(id: "a", entry: entry(jsonBytes: 1_500)) == .rejectedEntryTooLarge)
        #expect(store.get("a")?.responseJSON.count == 500)
        #expect(store.totalRetainedBytes == 500)
    }

    @Test func anEntryLargerThanTheTotalBudgetIsRejected() {
        let store = ResponseStore(
            capacity: 10, maximumEntryBytes: 5_000, maximumTotalBytes: 1_000)
        #expect(store.put(id: "a", entry: entry(jsonBytes: 1_200)) == .rejectedEntryTooLarge)
        #expect(store.count == 0)
        #expect(store.totalRetainedBytes == 0)
    }

    @Test func zeroOrNegativeCapacityStoresNothing() {
        for capacity in [0, -1] {
            let store = ResponseStore(capacity: capacity)
            #expect(
                store.put(id: "a", entry: entry("a")) == .rejectedNoCapacity,
                "capacity \(capacity) accepted an entry")
            #expect(store.count == 0)
            #expect(store.totalRetainedBytes == 0)
        }
    }

    @Test func zeroOrNegativeTotalBudgetStoresNothing() {
        for budget in [0, -1] {
            let store = ResponseStore(
                capacity: 4, maximumEntryBytes: 1_000, maximumTotalBytes: budget)
            #expect(
                store.put(id: "a", entry: entry(jsonBytes: 10)) == .rejectedNoCapacity,
                "budget \(budget) accepted an entry")
            #expect(store.count == 0)
        }
    }

    /// Item arrays are part of the retained memory: a store that counted only
    /// `responseJSON` would let the flattened conversation grow for free.
    @Test func itemsCountTowardTheEntryBudget() throws {
        let plain = entry(jsonBytes: 100)
        let withItem = ResponseStore.Entry(
            responseJSON: Data(repeating: 0x61, count: 100),
            inputItems: [item(String(repeating: "x", count: 500))],
            outputItems: [], created: Date())

        let plainBytes = try #require(ResponseStore.retainedByteCount(of: plain))
        let itemBytes = try #require(ResponseStore.retainedByteCount(of: withItem))
        #expect(plainBytes == 100)
        #expect(itemBytes > 500, "the 500-character item was not counted: \(itemBytes)")

        let store = ResponseStore(
            capacity: 4, maximumEntryBytes: itemBytes - 1, maximumTotalBytes: 100_000)
        #expect(store.put(id: "plain", entry: plain) == .stored)
        #expect(store.put(id: "big", entry: withItem) == .rejectedEntryTooLarge)
        #expect(store.totalRetainedBytes == 100)
    }

    @Test func theTotalNeverExceedsTheBudgetUnderMixedPuts() {
        let store = ResponseStore(
            capacity: 6, maximumEntryBytes: 1_000, maximumTotalBytes: 3_000)
        let sizes = [100, 900, 400, 1_000, 250, 800, 600, 1_000, 50, 700]
        for (index, size) in sizes.enumerated() {
            store.put(id: "r\(index)", entry: entry(jsonBytes: size))
            #expect(store.totalRetainedBytes <= store.maximumTotalBytes)
            #expect(store.count <= store.capacity)
        }
    }

    @Test func deleteAndEvictionKeepTheAccountingConsistent() {
        let store = ResponseStore(
            capacity: 3, maximumEntryBytes: 1_000, maximumTotalBytes: 5_000)
        store.put(id: "a", entry: entry(jsonBytes: 500))
        store.put(id: "b", entry: entry(jsonBytes: 700))
        #expect(store.delete("a"))
        #expect(store.totalRetainedBytes == 700)
        store.put(id: "c", entry: entry(jsonBytes: 900))
        store.put(id: "d", entry: entry(jsonBytes: 1_000))
        for (index, size) in [("e", 1_000), ("f", 1_000), ("g", 1_000)] {
            store.put(id: index, entry: entry(jsonBytes: size))
        }
        #expect(store.totalRetainedBytes <= 5_000)
        #expect(store.get("b") == nil, "byte eviction did not reach the oldest")
        #expect(store.delete("b") == false)
    }

    @Test func defaultsAreTheDocumentedBudgets() {
        let store = ResponseStore()
        #expect(store.capacity == 256)
        #expect(store.maximumEntryBytes == 16 * 1024 * 1024)
        #expect(store.maximumTotalBytes == 64 * 1024 * 1024)
    }

    @Test func itemLookupPrefersOutputThenInput() {
        let store = ResponseStore(capacity: 2)
        let shared = "item-1"
        let fromInput = ResponsesAPIRequest.Item(
            type: "message", id: shared, role: "user", content: .string("input"))
        let fromOutput = ResponsesAPIRequest.Item(
            type: "message", id: shared, role: "assistant", content: .string("output"))
        store.put(
            id: "a",
            entry: ResponseStore.Entry(
                responseJSON: Data("a".utf8),
                inputItems: [fromInput], outputItems: [fromOutput], created: Date()))

        #expect(store.item(withID: shared) == fromOutput)
        #expect(store.item(withID: "absent") == nil)
    }
}
