import ContinuityCore
import Foundation

// The session surface: begin/end, instructions, tools, execution and retrieval.
//
// Split out of `MemoryService.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion. Members whose
// callers stay behind widened from `private` to internal.
extension MemoryService {

    /// Starts a session and returns what the engine needs to install.
    ///
    /// A failure here degrades rather than propagates: the session continues
    /// with local memory when that is allowed, and with none when it is not.
    /// - Parameter tag: what the session is about, when the caller could
    ///   tell. Recorded on the session, shown in the log; not a scope.
    public func beginSession(
        id: String,
        workspaceOverride: String? = nil,
        modelID: String? = nil,
        tag: String? = nil,
        focus: String? = nil
    ) async -> MemorySessionContext? {
        guard configuration.isEnabled else { return nil }
        guard let scope = configuration.scope(workspaceOverride: workspaceOverride) else {
            log(.rejectedScope(workspaceOverride ?? configuration.workspace))
            return nil
        }
        let session = MemorySession(id: id, modelID: modelID, tag: tag, focus: focus)
        var bootstrap = MemoryBootstrap.empty
        if let workspace = await workspace(for: scope) {
            do {
                bootstrap = try await workspace.store.sessionInit(session, in: scope)
                isDegraded = false
            } catch {
                isDegraded = true
                log(.degraded(operation: "sessionInit", detail: "\(error)"))
                guard configuration.degradesToLocalStore else { return nil }
                bootstrap = (try? await localStore.sessionInit(session, in: scope)) ?? .empty
            }
        } else {
            bootstrap = (try? await localStore.sessionInit(session, in: scope)) ?? .empty
        }
        // The person's own facts, from the shared workspace, ride along on
        // every project's bootstrap. Bounded small: they are preferences,
        // not state, and there should be a handful.
        if let sharedScope = configuration.sharedScope, scope != sharedScope,
            await workspace(for: sharedScope) != nil
        {
            let shared = await recordedFacts(in: sharedScope, limit: 12)
            if !shared.isEmpty { bootstrap = bootstrap.withShared(shared) }
        }
        let durable = await isDurable(in: scope)
        log(
            .sessionStarted(
                session: session.id, scope: scope,
                bootstrapRecords: bootstrap.records.count,
                bootstrapBytes: bootstrap.totalBytes))
        return MemorySessionContext(
            session: session,
            scope: scope,
            bootstrap: bootstrap,
            isDurable: durable)
    }

    /// The system-prompt fragment for a session.
    public func instructions(for context: MemorySessionContext) -> String {
        MemoryPrompt.instructions(
            scope: context.scope,
            session: context.session,
            bootstrap: context.bootstrap,
            isDurable: context.isDurable,
            tools: toolDefinitions().map(\.name))
    }

    /// The tool definitions to advertise, or none when tools are off.
    public func toolDefinitions() -> [MemoryToolDefinition] {
        guard configuration.isEnabled else { return [] }
        return MemoryTools.definitions(surface: configuration.toolSurface)
    }

    /// Runs one memory tool call in a session's scope.
    ///
    /// The scope comes from the session context, never from the call, so a
    /// model cannot reach another workspace by naming one.
    public func execute(
        name: String,
        arguments: [String: MemoryToolValue],
        in context: MemorySessionContext
    ) async -> MemoryToolResult {
        guard configuration.isEnabled else { return .failure("memory is disabled") }
        let store = await activeStore(for: context.scope)
        let (hint, onSearch) = await retrievalContext(
            name: name, arguments: arguments,
            store: store, scope: context.scope)
        let result = await MemoryTools.execute(
            name: name,
            arguments: arguments,
            store: store,
            scope: context.scope,
            session: context.session,
            limits: configuration.limits,
            guarding: configuration.guardsUserFacts,
            retrievalHint: hint,
            onSearch: onSearch)
        // Checked whatever the outcome: a call whose own write landed can
        // still have had a session event refused.
        let journalLost = await journalFailed(in: context.scope)
        if case .failure(let message) = result {
            log(.toolFailed(tool: name, detail: message))
            // A durable backend that failed sends later work to the local
            // store, and marks the session as no longer persisting. A failed
            // journal does not: the engine still holds every fact, and a
            // local retry would answer "stored" for a write that ends with
            // the process.
            if !isDegraded, !journalLost,
                message.contains("unavailable")
                    || message.contains("timed out")
            {
                isDegraded = true
                log(.degraded(operation: name, detail: message))
                if configuration.degradesToLocalStore {
                    return await MemoryTools.execute(
                        name: name,
                        arguments: arguments,
                        store: localStore,
                        scope: context.scope,
                        session: context.session,
                        limits: configuration.limits,
                        guarding: configuration.guardsUserFacts,
                        retrievalHint: hint,
                        onSearch: onSearch)
                }
            }
        } else {
            log(.toolSucceeded(tool: name))
            // Checked after the write, not only when a workspace is opened.
            // A ceiling that only holds while the set of workspaces is
            // changing is not a ceiling.
            await enforceResidencyBudget(keeping: context.scope)
        }
        return result
    }

    /// Ends a session. With consolidation off this only logs; the hook for
    /// asking the model what to keep lives in the engine, which owns
    /// generation.
    public func endSession(_ context: MemorySessionContext) async {
        log(.sessionEnded(session: context.session.id, scope: context.scope))
    }

    /// The ranking hint this call may read, and the closure that schedules the
    /// background pass the next search will read from.
    ///
    /// The closure queues the question on the hinter before the search returns,
    /// so the question is registered by the time the answer is; it still never
    /// waits on a judgement, because `register` only enqueues. That ordering is
    /// why an observer that awaits the hint right after a search cannot race the
    /// registration. Both are empty unless this really is a text search and
    /// a side-engine is wired, which is what leaves the deterministic path
    /// byte-for-byte what it was.
    func retrievalContext(
        name: String,
        arguments: [String: MemoryToolValue],
        store: any MemoryStore,
        scope: MemoryScope
    )
        async -> (MemoryRetrievalHint, (@Sendable (MemoryQuery) async -> Void)?)
    {
        guard name == "memory_search", let hinter = retrievalHinter,
            let text = arguments["query"]?.stringValue,
            !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return (.none, nil)
        }
        let hint = await hinter.hint(question: text, in: scope)
        let schedule: @Sendable (MemoryQuery) async -> Void = { query in
            guard let asked = query.text,
                !asked.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                return
            }
            await hinter.register(question: asked, in: scope, store: store)
        }
        return (hint, schedule)
    }

    /// Awaits the background T7 sweep, so a test can make a hint observable.
    /// Nothing on the request path calls this.
    func waitForRetrievalHints() async {
        await retrievalHinter?.waitForBackgroundWork()
    }

    /// Facts already in a scope, most important first, so a consolidation
    /// can update an address instead of inventing a near-duplicate beside
    /// it -- and can see the value it would be replacing.
    ///
    /// Values, not only keys. Shown keys alone, a model re-derived every one
    /// of them from a session that said nothing about them, and wrote "not
    /// specified" over a character's eye colour.
    public func recordedFacts(in scope: MemoryScope, limit: Int = 60) async -> [MemoryRecord] {
        let store = await activeStore(for: scope)
        return (try? await store.search(MemoryQuery(limit: limit), in: scope)) ?? []
    }

}
