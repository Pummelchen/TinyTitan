import Foundation
import TinyTitanCore

// Stored responses and the object a response echoes back.
//
// Split out of `ResponsesAPIModels.swift` (2026-09-28) under the 500-line-per-
// file rule (Task 8 of the cleanup runbook) as pure code motion; the store's
// `private` state moved with the class, so no access widened.
// MARK: - Stored responses

/// Finished responses kept for `previous_response_id`, `GET`, `DELETE` and
/// `input_items`. Bounded and in memory: this is the API's storage contract
/// for a single-user local server, not a database.
///
/// Three bounds apply, in this order, and all are checked under `lock`:
///
/// 1. **One entry** may not exceed `maximumEntryBytes`. An entry is the
///    response JSON plus both item arrays; a continuation's input items already
///    contain the whole prior chain, so this is also the chain bound.
/// 2. **One entry** may not exceed `maximumTotalBytes` (it could never fit).
/// 3. **The store** keeps at most `capacity` entries and at most
///    `maximumTotalBytes` of logical retained bytes. When a new entry would
///    exceed either, the oldest entries are evicted until it fits; the new
///    entry is the newest, so it is never the one evicted.
///
/// A refused `put` changes nothing: the existing entry (if any) stays, with its
/// place in the eviction order and its accounting untouched. That is what lets
/// the handler report a truthful `store: false` instead of claiming a stored
/// response that cannot be retrieved.
///
/// "Bytes" are logical, measured as the UTF-8 JSON of the response plus each
/// item; this is a memory budget, not an RSS measurement (allocator overhead
/// and shared buffers are not counted).
/// unchecked-invariant: every field is guarded by `lock`.
public final class ResponseStore: @unchecked Sendable {
    public struct Entry: Sendable {
        /// The response object as returned to the client, JSON-encoded.
        public let responseJSON: Data
        /// The conversation the response was generated from, as input items
        /// (a prior chain already flattened in).
        public let inputItems: [ResponsesAPIRequest.Item]
        /// The response's output, as the input items a follow-up carries.
        public let outputItems: [ResponsesAPIRequest.Item]
        public let created: Date

        /// What a request naming this response as `previous_response_id`
        /// continues from.
        ///
        /// This is an `Array` concatenation, so every read allocates a copy of
        /// `inputItems + outputItems`. That copy is bounded by the same
        /// per-entry budget the store enforces: the entry weight counts both
        /// arrays, so a chain that fits `maximumEntryBytes` cannot make this
        /// copy larger than that budget. `ResponseStoreTests
        /// .conversationCopyIsBoundedByTheEntryBudget` pins the measurement.
        ///
        /// A lazy view would avoid the copy, but `ResponsesAPIMapper.chatRequest`
        /// consumes `[Item]` and the response's exact input is `Array` semantics;
        /// trading those for a micro-optimisation is not worth it while the
        /// budget already bounds the allocation.
        public var conversation: [ResponsesAPIRequest.Item] { inputItems + outputItems }
    }

    /// What a `put` did, so a caller never has to guess whether the response it
    /// just produced can be retrieved.
    public enum PutOutcome: Sendable, Equatable {
        case stored
        /// The entry alone is larger than `maximumEntryBytes` or
        /// `maximumTotalBytes`.
        case rejectedEntryTooLarge
        /// `capacity` or `maximumTotalBytes` admits nothing.
        case rejectedNoCapacity
        /// An item could not be encoded to measure it. Refused rather than
        /// under-counted; no request shape is known to produce this.
        case rejectedUnmeasurableEntry
    }

    private let lock = NSLock()
    private var entries: [String: Entry] = [:]
    private var order: [String] = []
    /// Logical retained bytes per stored id, so `delete` and replacement can
    /// subtract without re-encoding.
    private var bytesByID: [String: Int] = [:]
    private var retainedBytes = 0

    public let capacity: Int
    /// Ceiling for one entry, including the flattened prior conversation.
    public let maximumEntryBytes: Int
    /// Ceiling for every entry together.
    public let maximumTotalBytes: Int

    public init(
        capacity: Int = 256,
        maximumEntryBytes: Int = 16 * 1024 * 1024,
        maximumTotalBytes: Int = 64 * 1024 * 1024
    ) {
        self.capacity = capacity
        self.maximumEntryBytes = maximumEntryBytes
        self.maximumTotalBytes = maximumTotalBytes
    }

    @discardableResult
    public func put(id: String, entry: Entry) -> PutOutcome {
        guard let weight = Self.retainedByteCount(of: entry) else {
            return .rejectedUnmeasurableEntry
        }
        return lock.withLock {
            guard capacity > 0, maximumTotalBytes > 0 else { return .rejectedNoCapacity }
            guard weight <= maximumEntryBytes, weight <= maximumTotalBytes else {
                return .rejectedEntryTooLarge
            }

            let existing = entries[id]
            let existingBytes = existing == nil ? 0 : (bytesByID[id] ?? 0)
            if let index = order.firstIndex(of: id) { order.remove(at: index) }
            entries.removeValue(forKey: id)
            bytesByID.removeValue(forKey: id)
            // The accounting invariant is `retainedBytes >= existingBytes`; a
            // corrupt map is reset to zero rather than allowed to trap.
            let (afterRemoval, underflow) = retainedBytes.subtractingReportingOverflow(
                existingBytes)
            retainedBytes = underflow ? 0 : afterRemoval

            // A re-put is a write: it refreshes the id's place in the order and
            // is the newest thing in the store, so the cap reaches the oldest
            // *other* entry first.
            while (order.count >= capacity
                || Self.exceedsBudget(retainedBytes, weight, maximumTotalBytes)),
                let oldest = order.first
            {
                order.removeFirst()
                if entries.removeValue(forKey: oldest) != nil,
                    let removedBytes = bytesByID.removeValue(forKey: oldest)
                {
                    retainedBytes -= removedBytes
                } else {
                    // Unreachable with the invariant above; reset rather than
                    // let a corrupt map keep bytes accounted after their entry.
                    retainedBytes = 0
                    bytesByID.removeAll()
                    for entry in entries.values {
                        retainedBytes += Self.retainedByteCount(of: entry) ?? 0
                    }
                }
            }

            order.append(id)
            entries[id] = entry
            bytesByID[id] = weight
            retainedBytes += weight
            return .stored
        }
    }

    public func get(_ id: String) -> Entry? {
        lock.withLock { entries[id] }
    }

    @discardableResult
    public func delete(_ id: String) -> Bool {
        lock.withLock {
            guard let entry = entries.removeValue(forKey: id) else { return false }
            if let bytes = bytesByID.removeValue(forKey: id) {
                let (remaining, underflow) = retainedBytes.subtractingReportingOverflow(bytes)
                retainedBytes = underflow ? 0 : remaining
            } else {
                let bytes = Self.retainedByteCount(of: entry) ?? 0
                let (remaining, underflow) = retainedBytes.subtractingReportingOverflow(bytes)
                retainedBytes = underflow ? 0 : remaining
            }
            order.removeAll { $0 == id }
            return true
        }
    }

    public var count: Int { lock.withLock { entries.count } }

    /// Logical retained bytes across every entry. Never exceeds
    /// `maximumTotalBytes` for a positive budget.
    public var totalRetainedBytes: Int { lock.withLock { retainedBytes } }

    /// An item of any stored conversation by its id, for `item_reference`.
    ///
    /// Searches output items first, then input items, as the old
    /// `outputItems + inputItems` concatenation did -- without building the
    /// combined array for every entry it walks.
    public func item(withID id: String) -> ResponsesAPIRequest.Item? {
        lock.withLock {
            for entry in entries.values {
                if let item = entry.outputItems.first(where: { $0.id == id }) {
                    return item
                }
                if let item = entry.inputItems.first(where: { $0.id == id }) {
                    return item
                }
            }
            return nil
        }
    }

    /// The logical size of a whole entry: its response JSON plus the JSON of
    /// every input and output item. Nil when an item cannot be encoded.
    static func retainedByteCount(of entry: Entry) -> Int? {
        var total = entry.responseJSON.count
        let encoder = JSONEncoder()
        for item in entry.inputItems {
            guard let bytes = encodedCount(item, encoder: encoder) else { return nil }
            guard let sum = safeAdd(total, bytes) else { return nil }
            total = sum
        }
        for item in entry.outputItems {
            guard let bytes = encodedCount(item, encoder: encoder) else { return nil }
            guard let sum = safeAdd(total, bytes) else { return nil }
            total = sum
        }
        return total
    }

    /// The logical size of a conversation that a store=true request is about to
    /// flatten into a new entry. Used by the request path to refuse an
    /// over-budget chain before generation, rather than after the model ran.
    static func retainedByteCount(of items: [ResponsesAPIRequest.Item]) -> Int? {
        var total = 0
        let encoder = JSONEncoder()
        for item in items {
            guard let bytes = encodedCount(item, encoder: encoder) else { return nil }
            guard let sum = safeAdd(total, bytes) else { return nil }
            total = sum
        }
        return total
    }

    private static func encodedCount(
        _ item: ResponsesAPIRequest.Item, encoder: JSONEncoder
    ) -> Int? {
        // `Item` is Codable over every field it carries. A failure is not
        // expected; nil makes the caller refuse rather than undercount.
        guard let data = try? encoder.encode(item) else { return nil }
        return data.count
    }

    private static func safeAdd(_ left: Int, _ right: Int) -> Int? {
        let (sum, overflow) = left.addingReportingOverflow(right)
        return overflow ? nil : sum
    }

    /// True when `retained + additional` is over `budget`, with the arithmetic
    /// overflow treated as over-budget rather than as room.
    private static func exceedsBudget(_ retained: Int, _ additional: Int, _ budget: Int) -> Bool {
        guard let sum = safeAdd(retained, additional) else { return true }
        return sum > budget
    }
}

// MARK: - Response object builders

/// Everything a response object echoes back: the request's own fields, plus the
/// sampling the validator resolved from them and the served model's profile.
/// Built once per request so the in_progress, completed and stored objects agree.
public struct ResponsesAPIEcho: Sendable {
    public let instructions: String?
    public let maxOutputTokens: Int?
    public let maxToolCalls: Int?
    public let previousResponseID: String?
    public let metadata: JSONValue
    public let user: String?
    public let safetyIdentifier: String?
    public let promptCacheKey: String?
    public let serviceTier: String
    public let truncation: String
    /// The sampling the server actually ran, resolved by the validator from the
    /// request where the client named a value and from the served model's own
    /// profile where it did not. These are numbers, never nil: the Response
    /// schema requires them, and a client cannot otherwise see what was applied.
    /// The *request-side* fields stay nil when the client named none (C11), so
    /// the profile still supplies the value here rather than the echo doing it.
    public let temperature: Float
    public let topP: Float
    public let presencePenalty: Float
    public let frequencyPenalty: Float
    public let store: Bool
    public let tools: [ResponsesAPIRequest.Tool]
    public let namespaces: [String: String]
    public let toolChoice: JSONValue
    public let textVerbosity: String
    public let reasoningEffort: String?
    public let reasoningSummary: String?
    public let parallelToolCalls: Bool

    public init(
        request: ResponsesAPIRequest,
        effectiveEffort: ModelReasoningEffort?,
        applied: GenerationConfig
    ) {
        instructions = request.instructions
        maxOutputTokens = request.maxOutputTokens
        maxToolCalls = request.maxToolCalls
        previousResponseID = request.previousResponseID
        metadata = request.metadata ?? .object([:])
        user = request.user
        safetyIdentifier = request.safetyIdentifier
        promptCacheKey = request.promptCacheKey
        serviceTier = "default"
        truncation = request.truncation ?? "disabled"
        // Echo what the sampler was handed, not the raw request. The mapper left
        // an omitted field nil so validation could resolve it; by this point the
        // validator has, exactly as the generation config below shows.
        temperature = applied.temperature
        topP = applied.topP ?? SamplingProfile.house.topP
        presencePenalty = applied.presencePenalty
        // frequency_penalty has no runtime knob: the validator admits only the
        // neutral value, so the number this server applied is always zero. The
        // schema requires the field, so it is echoed as that zero.
        frequencyPenalty = 0
        store = request.stores
        tools = request.tools ?? []
        namespaces = ResponsesAPIMapper.functionTools(request.tools).namespaces
        toolChoice = request.toolChoice ?? .string("auto")
        textVerbosity = request.text?.verbosity ?? "medium"
        reasoningEffort = request.reasoning?.effort ?? effectiveEffort?.rawValue
        reasoningSummary = request.reasoning?.summary
        parallelToolCalls = request.parallelToolCalls ?? true
    }
}
