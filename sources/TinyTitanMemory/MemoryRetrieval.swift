import Foundation

/// The ranking a background T7 pass produced for one question.
///
/// One entry per fact the engine judged able to answer the question, keyed by
/// the fact and carrying a fingerprint of the *value it judged*. The
/// fingerprint is what makes a stale hint safe: T7 answered about a value, and
/// once that value changes the answer no longer describes the fact, so a
/// search must not honour it. A hint is advisory in exactly the way the other
/// side-engine judgements are: no hint at all leaves the token ranking exactly
/// as it was.
public struct MemoryRetrievalHint: Sendable, Equatable {
    /// Key -> FNV-1a of the value the engine was shown.
    public let answered: [MemoryKey: UInt64]

    public init(answered: [MemoryKey: UInt64] = [:]) {
        self.answered = answered
    }

    public static let none = MemoryRetrievalHint()
    public var isEmpty: Bool { answered.isEmpty }

    /// The hinted facts first, then the token ranking's own order.
    ///
    /// This is the caller's half of the measured result: on the authored
    /// recall set, putting the facts T7 says could answer first is recall@1
    /// 4 of 4 against the token ranking's 1 of 4
    /// (`benchmark/side_engine_recall.py`). A fact the token ranking never
    /// returned at all — the semantic miss, no term in common — is fetched by
    /// key here, which is the whole reason the hint exists. The query's own
    /// filters still apply, so a hint cannot return a fact the caller excluded
    /// by prefix, tag or importance.
    public func applied(
        to records: [MemoryRecord],
        query: MemoryQuery,
        store: any MemoryStore,
        scope: MemoryScope
    ) async -> [MemoryRecord] {
        guard !answered.isEmpty else { return records }
        let limit = max(0, query.limit)
        let present = Set(records.map(\.key))
        let recalled = records.filter { record in
            guard let fingerprint = answered[record.key] else { return false }
            return Self.fingerprint(of: record.value) == fingerprint
        }
        let recalledKeys = Set(recalled.map(\.key))
        let rest = records.filter { !recalledKeys.contains($0.key) }
        var promoted: [MemoryRecord] = []
        // One failed fetch here costs one promotion and cannot misinform: the
        // list that comes back is the token ranking's own, which is exactly what
        // the caller had before any hint existed, and nothing about the store's
        // contents is claimed by it. This is the deliberate other half of the
        // decide-once in AUD-138 — every other swallowed read on this path now
        // says it could not read, and this one is left silent because the answer
        // it degrades to is already honest and there is no log at this site to
        // say it through.
        for key in answered.keys.sorted(by: { $0.rawValue < $1.rawValue })
        where !present.contains(key) {
            guard recalled.count + promoted.count < limit else { break }
            guard let record = try? await store.get(key, in: scope),
                Self.matches(record, query: query),
                Self.fingerprint(of: record.value) == answered[key]
            else { continue }
            promoted.append(record)
        }
        return Array((recalled + promoted + rest).prefix(limit))
    }

    /// The filters `MemoryRanking` applies before it scores, so a promoted
    /// fact is subject to the same query the token ranking answered.
    private static func matches(_ record: MemoryRecord, query: MemoryQuery) -> Bool {
        if let prefix = query.prefix, !prefix.isEmpty,
            !record.key.rawValue.hasPrefix(prefix)
        {
            return false
        }
        if let minimum = query.minimumImportance, (record.importance ?? 0) < minimum {
            return false
        }
        if !query.tags.isEmpty {
            let lowered = Set(record.tags.map { $0.lowercased() })
            guard query.tags.contains(where: { lowered.contains($0.lowercased()) }) else {
                return false
            }
        }
        return true
    }

    /// FNV-1a over the value's UTF-8. A hint only has to survive one process
    /// and be cheap to compare, but it must not be `hashValue`, which is
    /// seeded per process and would make a stored answer meaningless.
    static func fingerprint(of value: String) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in value.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return hash
    }
}

/// Runs T7 off the request path, in the idle window, and leaves the answers as
/// a ranking hint a later search can use.
///
/// **Why it is not on the request path.** A judgement is a full CPU
/// generation: 15.2 s on the 4B, measured over the wired cases
/// (`docs/side-engine-tasks.md`). `memory_search` is a tool call the turn
/// waits on, and a candidate loop over a store would cost minutes. So
/// `register` records the question and returns; one background task walks the
/// scope's facts while the server is idle, and a search reads only the hints
/// that already exist. Nothing here is ever awaited by a turn.
///
/// **The idle window.** `isIdle` is the server's own "no client is waiting"
/// read. The task checks it before every judgement and sleeps between checks,
/// so a busy server is never slowed and a judgement that is already running
/// keeps its width policy (one thread while someone waits). The task gives up
/// after `maximumIdleWaitSeconds` of never seeing an idle window; the next
/// search registers the question again.
///
/// **Bounded three ways.** One question covers at most `coverageLimit` facts —
/// a store larger than that keeps the token ranking for the rest. At most
/// `maximumTrackedQuestions` questions keep hints, least recently asked
/// evicted. And a verdict is a fact plus a value fingerprint, so a fact whose
/// value changed cannot be promoted on an answer about the old one.
public actor MemoryRetrievalHinter {
    /// How many of a scope's facts one question may cover.
    public static let coverageLimit = 64
    /// How many questions keep a hint. Least recently asked is evicted.
    public static let maximumTrackedQuestions = 16
    /// How long a sweep waits for an idle window before giving up.
    static let maximumIdleWaitSeconds = 900.0
    static let idlePollMilliseconds = 250

    /// One question's sweep: the candidates, what has been judged, and what
    /// the engine said could answer.
    private struct Question: Sendable {
        let question: String
        let scope: MemoryScope
        let store: any MemoryStore
        var candidateKeys: [MemoryKey]
        var judged: Set<MemoryKey>
        /// Key -> fingerprint of the value the engine judged.
        var answered: [MemoryKey: UInt64]
        var cursor: Int
        var lastAsked: Date
    }

    private let engine: any MemorySideEngine
    private let isIdle: @Sendable () -> Bool
    private let log: @Sendable (MemoryLogEvent) -> Void
    private var questions: [String: Question] = [:]
    /// Least recently asked first, for eviction.
    private var recency: [String] = []
    /// Questions waiting for a sweep, in the order they were first asked.
    private var queue: [String] = []
    private var background: Task<Void, Never>?
    private var stopped = false
    /// Scopes whose candidate-pool read has already been reported, so a store
    /// that stays broken says it once rather than once per question.
    private var reportedSweepFailures: Set<String> = []

    init(
        engine: any MemorySideEngine,
        isIdle: @escaping @Sendable () -> Bool,
        log: @escaping @Sendable (MemoryLogEvent) -> Void = { _ in }
    ) {
        self.engine = engine
        self.isIdle = isIdle
        self.log = log
    }

    /// Records a question and schedules a sweep. Never runs the engine here:
    /// it returns as soon as the work is queued.
    func register(question: String, in scope: MemoryScope, store: any MemoryStore) {
        guard !stopped,
            !question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return
        }
        let key = Self.key(question, scope)
        var entry =
            questions[key]
            ?? Question(
                question: question, scope: scope, store: store,
                candidateKeys: [], judged: [],
                answered: [:], cursor: 0, lastAsked: Date())
        entry.lastAsked = Date()
        questions[key] = entry
        Self.touch(key, in: &recency)
        evictIfNeeded()
        if !queue.contains(key) { queue.append(key) }
        startBackgroundIfNeeded()
    }

    /// What the background pass has decided for this question so far.
    func hint(question: String, in scope: MemoryScope) -> MemoryRetrievalHint {
        guard let entry = questions[Self.key(question, scope)] else { return .none }
        return MemoryRetrievalHint(answered: entry.answered)
    }

    /// Awaits the queue, so a test can make the hint observable. Nothing on
    /// the request path calls this.
    func waitForBackgroundWork() async {
        while let task = background {
            await task.value
            if queue.isEmpty { break }
        }
    }

    /// Stops the sweep and drops every hint. Called on the service's shutdown
    /// so the engine is not asked a question while it is being released.
    func shutdown() {
        stopped = true
        background?.cancel()
        background = nil
        questions.removeAll()
        recency.removeAll()
        queue.removeAll()
    }

    // MARK: - the sweep

    private func startBackgroundIfNeeded() {
        guard background == nil, !stopped, !queue.isEmpty else { return }
        background = Task { [weak self] in await self?.runQueue() }
    }

    /// One background worker, so the engine is driven by one caller and the
    /// sweeps cannot pile up behind a burst of searches.
    private func runQueue() async {
        defer { background = nil }
        var idleWaited = 0.0
        var judgedNow = 0
        var answeredNow = 0
        while !stopped, !queue.isEmpty {
            let key = queue[0]
            guard var entry = questions[key] else {
                queue.removeFirst()
                idleWaited = 0
                continue
            }
            if entry.candidateKeys.isEmpty {
                do {
                    let records = try await entry.store.search(
                        MemoryQuery(limit: Self.coverageLimit), in: entry.scope)
                    entry.candidateKeys = records.map(\.key)
                    questions[key] = entry
                } catch {
                    // Said once per scope, and the question is dropped rather
                    // than left to look like a workspace with nothing in it: a
                    // silent zero-hint run and an unreadable store produce the
                    // same queue, and only one of them is fine.
                    if reportedSweepFailures.insert(entry.scope.workspace).inserted {
                        log(
                            .degraded(
                                operation: "hint-coverage",
                                detail: "candidate read failed in workspace "
                                    + "'\(entry.scope.workspace)', no facts were "
                                    + "judged for this question: \(error)"))
                    }
                    queue.removeFirst()
                    idleWaited = 0
                    continue
                }
            }
            while entry.cursor < entry.candidateKeys.count,
                entry.judged.contains(entry.candidateKeys[entry.cursor])
            {
                entry.cursor += 1
            }
            if entry.cursor >= entry.candidateKeys.count
                || entry.judged.count >= Self.coverageLimit
            {
                queue.removeFirst()
                idleWaited = 0
                continue
            }
            if !isIdle() {
                if idleWaited >= Self.maximumIdleWaitSeconds {
                    queue.removeFirst()
                    idleWaited = 0
                    continue
                }
                try? await Task.sleep(for: .milliseconds(Self.idlePollMilliseconds))
                idleWaited += Double(Self.idlePollMilliseconds) / 1000
                continue
            }
            idleWaited = 0
            let factKey = entry.candidateKeys[entry.cursor]
            entry.cursor += 1
            entry.judged.insert(factKey)
            questions[key] = entry
            if let fingerprint = await judge(entry, factKey: factKey) {
                // Re-read: `register` may have touched the entry while the
                // engine was generating, and if the question was evicted
                // meanwhile there is nothing to update — resurrecting it here
                // would defeat the LRU bound.
                guard var current = questions[key] else { continue }
                current.cursor = entry.cursor
                current.judged.insert(factKey)
                current.answered[factKey] = fingerprint
                questions[key] = current
                answeredNow += 1
            }
            judgedNow += 1
        }
        if judgedNow > 0 {
            log(.retrievalHints(judged: judgedNow, answered: answeredNow))
        }
    }

    /// One fact, one question. A `nil` answer — a shut-down or confused
    /// engine, an unparsable completion — records only that the fact was
    /// judged, so a failure is never a hint. Returns the value fingerprint
    /// when the engine said the fact could answer, and nil otherwise.
    private func judge(_ entry: Question, factKey: MemoryKey) async -> UInt64? {
        guard let record = try? await entry.store.get(factKey, in: entry.scope) else {
            return nil
        }
        let fact = MemoryFact(key: factKey.rawValue, value: record.value)
        guard let answer = await engine.couldAnswer(entry.question, fact), answer else {
            return nil
        }
        return MemoryRetrievalHint.fingerprint(of: record.value)
    }

    // MARK: - bookkeeping

    private static func key(_ question: String, _ scope: MemoryScope) -> String {
        let normalized = question.lowercased()
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
        return "\(scope.namespace)/\(scope.user)/\(scope.workspace)/\(normalized)"
    }

    private static func touch(_ key: String, in recency: inout [String]) {
        recency.removeAll { $0 == key }
        recency.append(key)
    }

    private func evictIfNeeded() {
        while questions.count > Self.maximumTrackedQuestions, let oldest = recency.first {
            recency.removeFirst()
            questions.removeValue(forKey: oldest)
            queue.removeAll { $0 == oldest }
        }
    }
}
