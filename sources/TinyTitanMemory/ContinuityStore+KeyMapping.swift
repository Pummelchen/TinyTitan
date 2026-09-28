import Foundation
import ContinuityCore

// Key and scope mapping for the continuity store: the prefix plan, the
// address/key text forms, the normalisers and the error translation.
//
// Split out of `ContinuityStore.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion.
extension ContinuityStore {

    static func pushDown(prefix: String) -> PrefixPlan {
        let normalized = normalizeKeyText(prefix)
        guard !normalized.isEmpty else { return PrefixPlan(namespace: nil, residual: "") }
        let segments = normalized.split(separator: "/").map(String.init)
        guard !segments.isEmpty else { return PrefixPlan(namespace: nil, residual: "") }
        if normalized.hasSuffix("/") {
            // Whole segments: "decisions/" is exactly the namespace k.decisions.
            return PrefixPlan(
                namespace: (["k"] + segments).joined(separator: "."),
                residual: normalized)
        }
        guard segments.count >= 2 else {
            // One partial segment. It could be a namespace or a bare key, so
            // nothing can be pushed down without risking a wrong answer.
            return PrefixPlan(namespace: nil, residual: normalized)
        }
        let leading = segments.dropLast()
        return PrefixPlan(
            namespace: (["k"] + leading).joined(separator: "."),
            residual: normalized)
    }

    func task(for scope: MemoryScope) async throws -> UUID {
        if let existing = taskIDs[scope] { return existing }
        let id = Self.taskIdentifier(for: scope)
        if await engine.task(id) == nil {
            _ = try await engine.createTask(
                title: "\(scope.workspace)",
                objective: "Durable memory for "
                    + "\(scope.namespace)/\(scope.user)/"
                    + "\(scope.workspace)",
                id: id)
        }
        taskIDs[scope] = id
        return id
    }

    // MARK: - Address mapping

    public struct Address: Equatable {
        public let namespace: String
        public let key: String
    }

    /// `decisions/sync` becomes namespace `k.decisions`, key `sync`.
    ///
    /// The leading `k` keeps a one-segment key from colliding with a
    /// two-segment one, and it is added on every address, so no key the model
    /// writes can produce it by accident.
    public static func address(for key: MemoryKey) -> Address {
        let segments = normalizeKeyText(key.rawValue).split(separator: "/").map(String.init)
        guard let last = segments.last else { return Address(namespace: "k", key: "empty") }
        let leading = segments.dropLast()
        let namespace = (["k"] + leading).joined(separator: ".")
        return Address(namespace: namespace, key: last)
    }

    public static func keyText(for item: ContinuityCore.MemoryItem) -> String {
        var segments = item.namespace.split(separator: ".").map(String.init)
        if segments.first == "k" { segments.removeFirst() }
        segments.append(item.key)
        return segments.joined(separator: "/")
    }

    /// The continuity address alphabet is narrower than a memory key's: no
    /// uppercase, and no dots inside a segment. Folding is deliberate and
    /// idempotent, so a key handed back to the model resolves to the same
    /// address when it comes round again.
    static func normalizeKeyText(_ raw: String) -> String {
        String(
            raw.lowercased().map { character in
                if character == "." { return "-" }
                return character
            })
    }

    static func normalize(_ record: MemoryRecord) -> MemoryRecord {
        guard let key = try? MemoryKey(validating: normalizeKeyText(record.key.rawValue)) else {
            return record
        }
        var copy = record
        copy.key = key
        return copy
    }

    /// A stable task id for a scope, so a restart finds the same memory.
    /// Swift's own hashing is seeded per process and cannot be used for this.
    static func taskIdentifier(for scope: MemoryScope) -> UUID {
        let text = "\(scope.namespace)/\(scope.user)/\(scope.workspace)"
        var high: UInt64 = 0xcbf2_9ce4_8422_2325
        var low: UInt64 = 0x9e37_79b9_7f4a_7c15
        for byte in text.utf8 {
            high ^= UInt64(byte)
            high = high &* 0x0000_0100_0000_01b3
            low = (low &+ UInt64(byte)) &* 0xff51_afd7_ed55_8ccd
            low ^= low >> 33
        }
        var bytes = [UInt8]()
        for shift in stride(from: 56, through: 0, by: -8) {
            bytes.append(UInt8(truncatingIfNeeded: high >> UInt64(shift)))
        }
        for shift in stride(from: 56, through: 0, by: -8) {
            bytes.append(UInt8(truncatingIfNeeded: low >> UInt64(shift)))
        }
        bytes[6] = (bytes[6] & 0x0F) | 0x40
        bytes[8] = (bytes[8] & 0x3F) | 0x80
        return UUID(
            uuid: (
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
                bytes[8], bytes[9], bytes[10], bytes[11],
                bytes[12], bytes[13], bytes[14], bytes[15]
            ))
    }

    static func translate(_ error: ContinuityError) -> MemoryError {
        switch error {
        case .valueTooLarge(let bytes, let limit):
            return .valueTooLarge(bytes: bytes, limit: limit)
        case .invalidKey(let value, let reason), .invalidNamespace(let value, let reason):
            return .invalidKey(value, reason)
        case .notPersisted(let detail):
            return .notPersisted(detail)
        default:
            return .backendUnavailable(String(describing: error))
        }
    }
}
