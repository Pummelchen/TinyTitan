import Darwin
import Foundation
import TinyTitanCore
import TinyTitanLib
import TinyTitanMemory
import TinyTitanServerCore
import TinyTitanPluginAPI
import TinyTitanVariants

let arguments: ServerArguments
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
} catch ServerArgumentError.help {
    print(ServerArguments.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerArguments.usage)\n".utf8))
    exit(2)
}
if let line = arguments.calibrationLogLine {
    FileHandle.standardError.write(Data((line + "\n").utf8))
}

// A launcher parses stdout, so the catalog is the only thing written there;
// skipped directories go to stderr.
if arguments.catalogOnly, let directory = arguments.modelsDirectory {
    let catalog = ModelCatalog.scan(directory: URL(fileURLWithPath: directory))
    catalog.reportSkipped()
    do {
        FileHandle.standardOutput.write(try catalog.jsonData() + Data("\n".utf8))
        exit(0)
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        exit(1)
    }
}

do {
    // The listener binds after the model is loaded, so an occupied port would
    // otherwise cost a full multi-gigabyte load before it was reported. The
    // bind below stays authoritative; this only moves the common refusal ahead
    // of the load.
    if !ServerPortCheck.isAvailable(port: arguments.port) {
        FileHandle.standardError.write(
            Data("error: --port \(arguments.port) is already in use on 127.0.0.1\n".utf8))
        exit(2)
    }
    let signals = ServerTerminationSignals()
    let modelURL = URL(fileURLWithPath: arguments.model).standardizedFileURL
    // The plugin's parameter table (TPP/1 §4): an explicit value the engine
    // cannot honour is refused with its reason; a default it cannot honour is
    // replaced by the engine's own, announced rather than ignored.
    if arguments.modelsDirectory == nil,
        let resolved = try? EngineFactories.pluginRegistry.resolve(modelURL)
    {
        let gate = PluginParameterGate.check(arguments, plugin: resolved)
        if let refusal = gate.refusal { throw ServerArgumentError.invalid(refusal) }
        for notice in gate.notices { ServerLog.diagnostic(notice) }
    }
    // Without --reasoning these are the --thinking / --reasoning-effort
    // values verbatim and nothing is read from disk; the router fits the
    // level per model itself, so it needs nothing from here.
    let reasoning =
        arguments.modelsDirectory == nil
        ? try arguments.singleModelReasoning(directory: modelURL)
        : (thinking: arguments.thinkingMode, effort: arguments.reasoningEffort)
    // Batched serving: one runner with this many KV/GDN slots, and a
    // coordinator that admits that many at once. `sessionSlots` caps the width
    // to one under MTP, and both the plan and the coordinator read it so they
    // cannot disagree.
    let concurrency = arguments.sessionSlots
    let makePlan = {
        ModelSessionPlan.from(
            arguments: arguments,
            modelDirectory: modelURL,
            thinking: reasoning.thinking,
            reasoningEffort: reasoning.effort,
            mtpModelDirectory: arguments.mtpModel.map {
                URL(fileURLWithPath: $0).standardizedFileURL
            })
    }

    let backend: any ServerInferenceBackend
    let facts: ModelSessionFacts
    var managed: ManagedModelBackend?
    var router: ModelRouter?
    /// The initial model's engine, set on the catalog path so the routing banner
    /// can name it beside `prompt_cache`. Every route is a GPU plugin (TT-142
    /// slice (b)), so it is a constant rather than a per-entry field.
    var initialEngine: String?
    // The reasoning profile a single-model server resolved before the session
    // exists; the catalog path takes it from the routed entry instead.
    var reasoningProfile: ServerReasoningProfile?

    if let modelsDirectory = arguments.modelsDirectory {
        var catalog = ModelCatalog.scan(directory: URL(fileURLWithPath: modelsDirectory))
        // --model may name a directory outside the models directory; it is
        // served beside the catalog rather than refused.
        var initial = catalog.entry(idOrPath: arguments.model)
        if initial == nil, FileManager.default.fileExists(atPath: modelURL.path) {
            initial = catalog.add(probing: modelURL)
        }
        catalog.reportSkipped()
        guard let initial else { throw ModelRouterError.notInCatalog(arguments.model) }
        let routing = try ModelRouter(
            catalog: catalog,
            initialModelID: initial.id,
            reasoning: arguments.requestedReasoningLevel,
            maximumContext: arguments.maxContext,
            loader: ModelRouter.standardLoader(arguments: arguments))
        if !arguments.lazyLoad {
            try await routing.preload()
        }
        print(
            "catalog: \(catalog.entries.count) models in \(catalog.directory.path); "
                + (arguments.lazyLoad
                    ? "\(initial.id) loads on the first request" : "loaded \(initial.id)"))
        router = routing
        backend = routing
        initialEngine = "gpu"
        facts = ModelSessionFacts(
            modelID: initial.id,
            prefillChunkTokens: 0,
            // The mode the initial model will really run, not the one
            // requested: the rule lives in `initialPromptCacheMode` so the
            // banner, the residency line and a test all read the same answer.
            promptCacheMode: ServerModelSession.initialPromptCacheMode(
                requested: arguments.promptCacheMode,
                maxConcurrentSequences: concurrency))
        reasoningProfile = routing.servedModel(named: initial.id)?.reasoningProfile
    } else if arguments.managesResidency {
        let plan = makePlan()
        // Reads manifest.json only; a bad --model still fails here at launch
        // rather than on the first request.
        facts = try plan.previewFacts(
            modelIDOverride: arguments.modelIDOverride,
            factory: EngineFactories.gpuFactory(directoryURL: modelURL))
        let residency = ManagedModelBackend(
            plan: plan,
            facts: facts,
            idleTimeout: arguments.idleUnloadSeconds > 0
                ? .seconds(arguments.idleUnloadSeconds) : nil)
        managed = residency
        backend = residency
    } else {
        let plan = makePlan()
        let session = try await plan.makeSession(
            factory: EngineFactories.gpuFactory(directoryURL: modelURL))
        backend = session
        facts = ModelSessionFacts(
            modelID: arguments.modelIDOverride ?? session.defaultModelID,
            prefillChunkTokens: session.prefillChunkTokens,
            promptCacheMode: session.promptCacheMode,
            expertCacheSlots: session.expertCacheSlots)
    }

    // The coordinator owns the "a client generation is in flight" signal, which
    // memory's background pass reads to stay out of a waiting client's way.
    let coordinator = ServerCoordinator(
        queueLimit: arguments.queueLimit,
        width: concurrency)

    // Persistent memory wraps whatever backend was built: one decorator on
    // the way in, and nothing at all when it is disabled.
    let servingBackend = ServerMemoryFactory.wrap(
        backend,
        isClientGenerating: { coordinator.generating.isBusy })

    let server = TinyTitanHTTPServer(
        modelID: facts.modelID,
        queueLimit: arguments.queueLimit,
        maxConcurrentSequences: concurrency,
        backend: servingBackend,
        reasoningProfile: try reasoningProfile
            ?? makePlan().reasoningProfile(
                factory: EngineFactories.gpuFactory(directoryURL: modelURL)),
        router: router,
        coordinator: coordinator)
    _ = try await server.start(port: arguments.port)
    let diskCache =
        facts.promptCacheMode == .off
        ? "off" : arguments.promptCacheDiskDirectory ?? "off"
    let cacheMemoryMiB =
        facts.promptCacheMode == .off
        ? 0 : arguments.promptCacheMemoryMiB
    let mtp = arguments.mtpModel == nil ? "off" : "on:\(arguments.mtpMemoryMiB)MiB"
    let residencyBanner =
        arguments.managesResidency
        ? " lazy_load=on idle_unload=\(arguments.idleUnloadSeconds > 0 ? "\(arguments.idleUnloadSeconds)s" : "off")"
        : ""
    if let router {
        // Prefill chunk and expert slots belong to whichever install is
        // loaded, so the routing banner states the server's own settings. The
        // exception is `prompt_cache`, which belongs to the model: it reports
        // the initial model's engine and real mode, and the residency line
        // reports both again on every load and switch.
        let engine = initialEngine.map { " engine=\($0)" } ?? ""
        print(
            "TinyTitanServer \(ServerVersion.current) ready at http://127.0.0.1:\(arguments.port) models=\(router.servedModels.count) initial=\(facts.modelID)\(engine) context=\(arguments.maxContext) concurrency=\(concurrency) prompt_cache=\(facts.promptCacheMode.rawValue) reasoning=\(arguments.requestedReasoningLevel.rawValue) dynamic=on"
        )
    } else {
        print(
            "TinyTitanServer \(ServerVersion.current) ready at http://127.0.0.1:\(arguments.port) model=\(facts.modelID) context=\(arguments.maxContext) concurrency=\(concurrency) prefill_chunk=\(facts.prefillChunkTokens)\(facts.expertCacheSlots > 0 ? " expert_slots=\(facts.expertCacheSlots)" : "") prompt_cache=\(facts.promptCacheMode.rawValue) prompt_cache_memory_mib=\(cacheMemoryMiB) prompt_cache_disk=\(diskCache) thinking=\(reasoning.thinking.rawValue) mtp=\(mtp)\(residencyBanner)"
        )
    }
    WatchdogConfiguration.shared.announce()
    if arguments.unloadDiscardsWarmCache {
        FileHandle.standardError.write(
            Data(
                ("warning: --idle-unload-seconds drops the in-memory prompt cache with "
                    + "the model; add --prompt-cache-disk <dir> so entries survive an "
                    + "unload, or the first request after each unload pays a full "
                    + "cold prefill\n").utf8))
    }

    _ = await signals.wait()
    try await server.shutdown()
    // After the server, so nothing is still writing: this flushes memory that
    // has not reached a session boundary and releases the workspace lock.
    if let memory = servingBackend as? MemoryBackend {
        await memory.shutDown()
    }
    // After the server, so the reaper cannot outlive it.
    await managed?.shutdown()
    await router?.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
