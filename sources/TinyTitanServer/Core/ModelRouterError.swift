import Foundation
import TinyTitan
import TinyTitanKit

// The router's failure modes and their wire descriptions.
//
// Split out of `ModelRouter.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion.
public enum ModelRouterError: Error, CustomStringConvertible {
    case notInCatalog(String)

    public var description: String {
        switch self {
        case .notInCatalog(let model):
            "\(model) is not a model the catalog can serve; run with --catalog to list them"
        }
    }
}

/// Serves every catalog model through one server, keeping at most one resident.
///
/// A request names its model; if that is not the resident one, the router
/// waits for in-flight work on the resident model to drain, releases it, and
/// loads the requested one. It never holds two: on a 26 GB machine two 35B
/// models would swap, so the old weights are dropped before the new ones are
/// mapped, and a switch costs a full load.
///
/// The HTTP coordinator already runs one generation at a time, so a switch
/// normally happens inside that serialised section with nothing else running.
/// The router does not rely on it: token counting and memory consolidation
/// reach the backend outside the coordinator, so residency is guarded here by
/// an in-flight count, exactly as `ManagedModelBackend` guards its unloads.
public actor ModelRouter: ServerInferenceBackend, ResidencyManaging, PromptTokenCounting,
    ModelRouting
{
    /// Builds a backend for one catalog entry. Injectable so switching can be
    /// tested against stubs without a model on disk.
    package typealias Loader =
        @Sendable (ModelCatalog.Entry, ReasoningChoice) async throws -> any ServerInferenceBackend
    /// Counts a request's prompt tokens for a model that is not resident,
    /// from its tokenizer alone. Injectable for the same reason.
    package typealias Counter =
        @Sendable (ModelCatalog.Entry, ReasoningChoice, ValidatedChatRequest) async throws -> Int

    public nonisolated let servedModels: [ServedModel]
    public nonisolated let initialModelID: String
    private nonisolated let initial: ServedModel
    private nonisolated let entries: [String: ModelCatalog.Entry]
    private nonisolated let choices: [String: ReasoningChoice]
    private let loader: Loader
    private let counter: Counter

    /// The protocol's single-model view, answered for the model loaded first.
    /// The HTTP layer asks `servedModel(named:)` instead once it routes.
    public nonisolated var maximumContext: Int { initial.maximumContext }
    public nonisolated var samplingDefaults: GenerationDefaults.Sampling { initial.sampling }

    private struct Resident {
        let id: String
        let backend: any ServerInferenceBackend
    }

    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Never>
    }

    private var resident: Resident?
    /// Callers currently using the resident backend. Non-zero blocks a switch.
    private var inFlight = 0
    /// A load is running; nothing is resident until it finishes.
    private var switching = false
    /// Callers waiting to switch away from the resident model. While any wait,
    /// new work for the resident model queues behind them rather than keeping
    /// the in-flight count above zero forever.
    private var pendingSwitches = 0
    private var waiters: [Waiter] = []

    package init(
        catalog: ModelCatalog,
        initialModelID: String,
        reasoning: ReasoningLevel,
        maximumContext: Int,
        loader: @escaping Loader,
        counter: @escaping Counter = ModelRouter.standardCounter
    ) throws {
        var served: [ServedModel] = []
        var choices: [String: ReasoningChoice] = [:]
        var entries: [String: ModelCatalog.Entry] = [:]
        /// Registers an entry under its own id, and remembers it for the
        /// `/v1/models` listing only once per install.
        func register(_ entry: ModelCatalog.Entry, into servedList: inout [ServedModel]) throws {
            let choice = try ReasoningFallback.choice(for: entry.kind, requested: reasoning)
            choices[entry.id] = choice
            entries[entry.id] = entry
            servedList.append(
                ServedModel(
                    id: entry.id,
                    displayName: entry.name,
                    maximumContext: Self.context(for: entry, configured: maximumContext),
                    sampling: entry.sampling,
                    reasoningProfile: Self.profile(for: entry, choice: choice)))
        }
        for entry in catalog.entries {
            try register(entry, into: &served)
            // Every engine the install can be served by, named by a suffix, so
            // one request can ask for the CPU copy of a dense model and the
            // next for the GPU one: `qwen3.5-2b_4-Bit@cpu`. The aliases share
            // the install and are listed beside it.
            for engine in entry.engines {
                let aliasID = "\(entry.id)@\(engine.rawValue)"
                guard aliasID != entry.id,
                    let alias = entry.served(by: engine, id: aliasID)
                else { continue }
                // A single-engine install gets the alias so the explicit
                // spelling resolves, but it is not *listed*: it names the same
                // engine as the bare id, and two entries for one model in
                // `/v1/models` is noise. A second engine is a real choice and
                // is listed.
                if entry.engines.count > 1, engine != entry.backend {
                    try register(alias, into: &served)
                } else {
                    let choice = try ReasoningFallback.choice(
                        for: alias.kind,
                        requested: reasoning)
                    choices[alias.id] = choice
                    entries[alias.id] = alias
                }
            }
        }
        guard let initial = served.first(where: { $0.id == initialModelID }) else {
            throw ModelRouterError.notInCatalog(initialModelID)
        }
        self.servedModels = served
        self.initialModelID = initialModelID
        self.initial = initial
        self.entries = entries
        self.choices = choices
        self.loader = loader
        self.counter = counter
    }

    /// Mirrors `CPUModelBackend`'s own clamp, so validation bounds max_tokens
    /// by the context the CPU engine will actually give the request.
    static func context(for entry: ModelCatalog.Entry, configured: Int) -> Int {
        switch entry.backend {
        case .gpu: configured
        case .cpu:
            min(
                configured, entry.contextLimit ?? CPUModelBackend.contextCeiling,
                CPUModelBackend.contextCeiling)
        }
    }

    /// A CPU family's template has the binary switch Qwen 3.6 has, so it is
    /// validated as that family, as the single-model CPU path already does.
    static func profile(
        for entry: ModelCatalog.Entry,
        choice: ReasoningChoice
    ) -> ServerReasoningProfile {
        let family: ModelFamily
        switch entry.kind {
        case .gpu(let gpuFamily): family = gpuFamily
        case .cpu: family = .qwen36
        }
        return ServerReasoningProfile(
            family: family, thinkingMode: choice.thinking,
            reasoningEffort: choice.effort)
    }

    public func reasoningChoice(for id: String) -> ReasoningChoice? { choices[id] }

    // MARK: - ServerInferenceBackend

    /// A request with no model -- the engine's own, such as memory
    /// consolidation -- runs on whatever is resident rather than forcing a load.
    package func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let active = try await acquire(request.model)
        defer { release() }
        return try await active.generate(request, onEvent: onEvent)
    }

    /// A count is a question about text, not a reason to switch. One naming
    /// a model that is not resident is answered from that model's tokenizer
    /// and the resident model stays loaded; routing it like a generation
    /// made the next generation pay a full reload for a number.
    package func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        let target = request.model ?? resident?.id ?? initialModelID
        if target != resident?.id, let entry = entries[target], let choice = choices[target] {
            return try await counter(entry, choice, request)
        }
        let active = try await acquire(request.model)
        defer { release() }
        guard let counting = active as? any PromptTokenCounting else {
            throw ServerRequestError.unsupportedOperation("count_tokens")
        }
        return try await counting.countPromptTokens(request)
    }

    /// Loads the initial model, so a server that was not asked to defer pays
    /// the load at startup rather than on the first request.
    public func preload() async throws {
        _ = try await acquire(initialModelID)
        release()
    }

    // MARK: - ResidencyManaging

    /// Releases the resident model once in-flight work drains. The next
    /// request loads whichever model it names.
    public func unload() async -> Bool {
        while resident != nil || switching {
            if !switching, inFlight == 0, let released = resident {
                resident = nil
                ServerLog.residency("unloaded \(released.id)")
                return true
            }
            if Task.isCancelled { return false }
            await waitForTurn()
        }
        return false
    }

    public func shutdown() {
        resident = nil
    }

    public var residentModelID: String? { resident?.id }

    // MARK: - Residency

    /// Marks the caller in flight on the backend for `requested`, switching to
    /// it first when it is not resident. Every exit that does not return a
    /// backend leaves the in-flight count untouched.
    private func acquire(_ requested: String?) async throws -> any ServerInferenceBackend {
        let target = requested ?? resident?.id ?? initialModelID
        guard let entry = entries[target], let choice = choices[target] else {
            throw ServerRequestError.unknownModel
        }
        while true {
            try Task.checkCancellation()
            if let resident, resident.id == target, !switching, pendingSwitches == 0 {
                inFlight += 1
                return resident.backend
            }
            if resident?.id != target, !switching, inFlight == 0 {
                return try await switchTo(entry, choice: choice)
            }
            if resident?.id == target {
                await waitForTurn()
                continue
            }
            pendingSwitches += 1
            await waitForTurn()
            pendingSwitches -= 1
            if Task.isCancelled {
                // Work for the resident model may be queued behind this
                // switch; with it abandoned they must re-check, not sleep on.
                wakeWaiters()
                throw CancellationError()
            }
        }
    }

    private func switchTo(
        _ entry: ModelCatalog.Entry,
        choice: ReasoningChoice
    ) async throws -> any ServerInferenceBackend {
        switching = true
        defer {
            switching = false
            wakeWaiters()
        }
        if let previous = resident {
            // Dropped before the next model is mapped: this reference is the
            // last one, so the old weights are gone before the new ones load.
            resident = nil
            ServerLog.residency("unloaded \(previous.id) to load \(entry.id)")
        }
        // Unstructured, so a client that disconnects mid-load does not abort a
        // load that the requests queued behind it are waiting for.
        let loader = self.loader
        let loaded = try await Task { try await loader(entry, choice) }.value
        resident = Resident(id: entry.id, backend: loaded)
        inFlight += 1
        let fitted =
            choice.effective == choice.requested
            ? "" : " (server level \(choice.requested.rawValue))"
        ServerLog.residency(
            "loaded \(entry.id) on the \(entry.backend.rawValue)"
                + ServerLog.promptCacheField(for: loaded)
                + " reasoning=\(choice.effective.rawValue)\(fitted)")
        return loaded
    }

    private func release() {
        inFlight -= 1
        if inFlight == 0 { wakeWaiters() }
    }

    private func waitForTurn() async {
        let id = UUID()
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                // Checked here, on the actor, so a cancellation that landed
                // before this waiter was queued cannot leave it asleep.
                if Task.isCancelled {
                    continuation.resume()
                } else {
                    waiters.append(Waiter(id: id, continuation: continuation))
                }
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        waiters.remove(at: index).continuation.resume()
    }

    /// Every waiter re-checks its own condition, so waking all of them is
    /// always safe; the ones that still cannot proceed queue again.
    private func wakeWaiters() {
        let woken = waiters
        waiters.removeAll()
        for waiter in woken {
            waiter.continuation.resume()
        }
    }

    // MARK: - Test hooks

    var inFlightCount: Int { inFlight }
    var waiterCount: Int { waiters.count }
}
