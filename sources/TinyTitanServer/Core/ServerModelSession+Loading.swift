// The `load` factory: verification, tokenizer, scratch and cache setup.
//
// Split out of ServerInference.swift (2026-09-27) under the 500-line-per-file
// rule; declarations and their doc comments are unchanged, except that
// `private` became internal where a declaration now lives beside an extension
// in another file.
import CryptoKit
import Foundation
import TinyTitan

extension ServerModelSession {
    /// lint:allow-long a sequential construction pipeline: tokenizer, Metal
    /// context, runtime config, model, optional MTP sidecar, runner, scratch.
    /// Each step consumes the last, so extracting any of them would return a
    /// tuple straight back into the next -- the same shape as Model.load.
    public static func load(
        modelDirectory: URL,
        maxContext: Int,
        slots: Int = 1,
        promptCacheMode: ServerPromptCacheMode = .multiPrefix,
        promptCacheMaximumEntries: Int = 4,
        promptCacheMemoryLimitBytes: Int = 256 * 1_048_576,
        promptCacheDiskDirectory: URL? = nil,
        promptCacheDiskLimitBytes: Int = 8_192 * 1_048_576,
        prefillChunkTokens requestedPrefillChunkTokens: Int? = nil,
        kvCachePrecision: KVCachePrecision = .int8,
        ropeScalingMode: RuntimeRoPEScalingMode = .none,
        thinkingMode: ModelThinkingMode = .off,
        reasoningEffort: ModelReasoningEffort? = nil,
        expertCacheSlots requestedExpertCacheSlots: Int? = nil,
        expertCacheBudgetBytes: Int? = nil,
        mtpModelDirectory: URL? = nil,
        mtpMemoryMiB: Int = StreamingMTPMemoryPlan.defaultBudgetMiB,
        reusingContext: MetalContext? = nil
    ) async throws -> ServerModelSession {
        _ = try GFTokenizer.requireModelDirectory(modelDirectory)
        let tokenizerFolder = GFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        guard let tokenizerFolder else {
            throw GFTokenizerError.missingToolTemplate
        }
        let templateURL = tokenizerFolder.appendingPathComponent("chat_template.jinja")
        guard FileManager.default.fileExists(atPath: templateURL.path) else {
            throw GFTokenizerError.missingToolTemplate
        }
        // Reasoning effort is defined per family; reject it before the
        // tokenizer bakes an unsupported control into its rendering. An
        // unreadable manifest is left for Model.load, which reports it better.
        if reasoningEffort != nil,
            let family = try? ManifestReader.peekFamily(directoryURL: modelDirectory)
        {
            try family.validateReasoning(
                thinkingMode: thinkingMode,
                effort: reasoningEffort)
        }
        let tokenizer = try await GFTokenizer.load(
            from: tokenizerFolder,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort)
        // A caller managing model residency supplies its own context so one
        // MTLCommandQueue and one compiled shader library survive across
        // unload/reload cycles (MetalContext.deinit documents that queue
        // teardown is not deinit-safe). Nil for every ordinary caller.
        let context = try reusingContext ?? MetalContext()
        let loadRuntime = try RuntimeConfiguration(
            forceLogitsHead: true,
            decodeExpertExecution: try RuntimeDecodeExpertExecution.environmentValue(),
            expertIOSynchronization: try RuntimeExpertIOSynchronization.environmentValue(),
            expertIOSubmission: try RuntimeExpertIOSubmission.environmentValue())
        let slotOverride = ProcessInfo.processInfo.environment["TINYTITAN_EXPERT_CACHE_SLOTS"]
            .flatMap(Int.init)
        // Precedence: --expert-cache-slots flag, then the env override, then a
        // count derived from the model's own expert stride against a 1 GiB budget.
        //
        // Derived rather than fixed because the right count depends on the
        // quantisation: 1 GiB is 16 slots at 4-bit and 8 at 8-bit, which are the
        // measured optima for each. The previous fixed default of 64 was slower
        // *and* larger than either -- benchmarked at the shipped 262144 context,
        // 4-bit managed 9.85 tok/s at 64 slots against 13.61 at 16.
        // The architecture comes from the manifest rather than being assumed,
        // exactly as the CLI resolves it: a payload of any other family should
        // load, not fail on a dimension mismatch.
        let modelFamily = try ManifestReader.peekFamily(directoryURL: modelDirectory)
        let expectedArch: ArchConfig
        do {
            // The family's preset, or -- for a family with more than one
            // geometry, like the dense Qwen 3.5 models -- the manifest's own
            // declaration. `ServerInferenceError` keeps the shape callers
            // expect; the reason travels in the message.
            expectedArch = try ArchConfig.resolved(
                forFamily: modelFamily,
                directoryURL: modelDirectory)
        } catch {
            throw ServerInferenceError.unsupportedModel("\(error)")
        }
        let derivedSlots: Int
        // An explicit --ram-budget names what the whole server may hold; this
        // only supplies the cache default, and it is clamped so a family tuned
        // on a 24 GiB machine cannot hand a smaller one a budget it has no room
        // for.
        let tunedBudget: Int
        if let identity = try? ManifestReader.peekIdentity(directoryURL: modelDirectory) {
            tunedBudget = RuntimeConfiguration.affordableExpertCacheBudget(
                ModelProfile.resolve(identity: identity).expertCacheBudgetBytes)
        } else {
            tunedBudget = RuntimeConfiguration.defaultExpertCacheBudgetBytes
        }
        let loadedManifest = try? ManifestReader.load(
            directoryURL: modelDirectory,
            expecting: expectedArch)
        // Bytes one expert occupies in the cache, across every layer.
        let cachePerSlotBytes =
            loadedManifest.map {
                Double($0.expertStride) * Double($0.arch.numLayers)
            } ?? 0
        let residentFloor = RuntimeConfiguration.residentFloorBytes(
            residentWeightBytes: loadedManifest?.files["model_weights.bin"]
                .map { Int($0.size) } ?? 0)
        let gib = { (bytes: Double) in bytes / 1_073_741_824 }
        let slotsGib = { (slots: Int) in gib(Double(slots) * cachePerSlotBytes) }
        if let explicitTarget = expertCacheBudgetBytes {
            // The flag is a target for the whole process, not just the cache:
            // the weights and the runtime are resident either way, so the cache
            // is what is left, and the slot count steps down to stay inside the
            // number the user named.
            derivedSlots = RuntimeConfiguration.expertCacheSlotsFitting(
                expertStrideBytes: loadedManifest?.expertStride ?? 0,
                layers: loadedManifest?.arch.numLayers ?? 0,
                cacheBytes: explicitTarget - residentFloor)
            let targetGib = gib(Double(explicitTarget))
            let floorGib = gib(Double(residentFloor))
            let cacheGib = slotsGib(derivedSlots)
            print(
                String(
                    format: "TinyTitan ram target=%.2fG cache=%.2fG slots=%d "
                        + "resident_floor=%.2fG estimate=%.2fG",
                    targetGib, cacheGib, derivedSlots, floorGib,
                    floorGib + cacheGib))
            if targetGib < floorGib + cacheGib {
                print(
                    String(
                        format: "TinyTitan ram warning: %.2fG is below this install's "
                            + "%.2fG floor (%.2fG resident + the %d-slot minimum cache); "
                            + "the cache is already at its smallest.",
                        targetGib, floorGib + cacheGib, floorGib, derivedSlots))
            }
        } else if let manifest = loadedManifest {
            derivedSlots = RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: manifest.expertStride,
                layers: manifest.arch.numLayers,
                budgetBytes: tunedBudget)
            let floorGib = gib(Double(residentFloor))
            print(
                String(
                    format: "TinyTitan ram profile cache=%.2fG slots=%d "
                        + "resident_floor=%.2fG estimate=%.2fG (cache budget, not a "
                        + "process target)",
                    slotsGib(derivedSlots), derivedSlots, floorGib,
                    floorGib + slotsGib(derivedSlots)))
        } else {
            // Unreadable manifest means the load below will fail with a better
            // message than anything this could throw, so pick the safe small end.
            derivedSlots = RuntimeConfiguration.allowedExpertCacheSlots.first ?? 8
        }
        let loadSlots = requestedExpertCacheSlots ?? slotOverride ?? derivedSlots
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            expecting: expectedArch,
            streamingMode: .pread(slotCount: loadSlots),
            expertCachePolicy: loadRuntime.modelExpertCachePolicy,
            integrityPolicy: .resolved(directoryURL: modelDirectory))
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: loadSlots,
            expertCachePolicy: loadRuntime.expertCachePolicy,
            rdadvisePolicy: ProcessInfo.processInfo.environment["TINYTITAN_RDADVISE_POLICY"]
                .map(RDAdvicePolicyMode.parse)
                ?? loadRuntime.rdadvisePolicy,
            prefillChunkTokens: requestedPrefillChunkTokens
                ?? ModelProfile.resolve(
                    modelID: model.modelID, family: model.config.family,
                    weightBits: model.routedExpertWeightBits
                ).prefillChunkTokens
                ?? defaultPrefillChunkTokens(
                    family: model.config.family,
                    fallback: loadRuntime.prefillChunkTokens),
            prefillAttentionPath: loadRuntime.prefillAttentionPath,
            forceLogitsHead: true,
            decodeExpertExecution: loadRuntime.decodeExpertExecution,
            expertIOSynchronization: loadRuntime.expertIOSynchronization,
            expertIOSubmission: loadRuntime.expertIOSubmission,
            kvCachePrecision: kvCachePrecision,
            ropeScalingMode: ropeScalingMode,
            yarnContextTokens: ropeScalingMode == .yarn
                ? maxContext : RuntimeConfiguration.defaultYaRNContextTokens)
        // MTP owns the target's runner and is single-sequence, so it cannot
        // batch. Otherwise the requested width is capped by what the worst-case
        // per-slot stores can hold beside the wired expert cache: the whole
        // point of the cap is that the cache cannot be paged out to rescue an
        // over-commit. The clamp is per load, so a catalog switch re-evaluates
        // it against the model actually being loaded.
        let requestedSlots = mtpModelDirectory == nil ? slots : 1
        let perSlotBytes = BatchedMemoryBudget.perSlotBytes(
            config: model.config,
            maxContext: maxContext,
            precision: runtime.kvCachePrecision,
            fp16RingEnabled: runtime.fp16RingEnabled,
            slidingWindow: model.config.slidingWindow,
            maxPrefillChunkTokens: runtime.prefillChunkTokens,
            vocab: model.config.vocabSize)
        let slotBudget = BatchedMemoryBudget.slotBudgetBytes(
            physicalMemory: ProcessInfo.processInfo.physicalMemory,
            // A dense family has no routed experts, so it has no expert cache to
            // hold back; see `expertCacheHeldBack`.
            expertCacheBudgetBytes: BatchedMemoryBudget.expertCacheHeldBack(
                numExperts: model.config.numExperts,
                configured: expertCacheBudgetBytes ?? tunedBudget))
        let effectiveSlots = BatchedMemoryBudget.effectiveSlots(
            requested: requestedSlots,
            perSlotBytes: perSlotBytes,
            budgetBytes: slotBudget)
        if effectiveSlots < requestedSlots {
            FileHandle.standardError.write(
                Data(
                    ("TinyTitan batch width \(requestedSlots) exceeds the memory budget "
                        + "(per-slot \(perSlotBytes / 1_048_576) MiB, budget "
                        + "\(slotBudget / 1_048_576) MiB); serving \(effectiveSlots) at once\n")
                        .utf8))
        }
        let mtpDecoder: StreamingMTPDecoder?
        let runner: RealForwardRunner
        if let mtpModelDirectory {
            let sidecarFamily = try ManifestReader.peekFamily(
                directoryURL: mtpModelDirectory)
            guard let sidecarArch = ArchConfig.knownArchitectures[sidecarFamily] else {
                throw ServerInferenceError.unsupportedModel(
                    "MTP sidecar declares family \(sidecarFamily.rawValue), "
                        + "which this runtime does not implement")
            }
            let sidecar = try Model.load(
                directoryURL: mtpModelDirectory,
                device: context.device,
                expecting: sidecarArch,
                streamingMode: .pread(slotCount: StreamingMTPMemoryPlan.expertSlots),
                expertCachePolicy: runtime.modelExpertCachePolicy,
                integrityPolicy: .resolved(directoryURL: mtpModelDirectory))
            let decoder = try StreamingMTPDecoder(
                targetModel: model,
                mtpSidecar: sidecar,
                context: context,
                maxContext: maxContext,
                memoryBudgetMiB: mtpMemoryMiB,
                runtimeConfiguration: runtime)
            mtpDecoder = decoder
            runner = decoder.target
        } else {
            mtpDecoder = nil
            runner = try RealForwardRunner(
                model: model,
                context: context,
                maxContext: maxContext,
                slots: effectiveSlots,
                runtimeConfiguration: runtime)
        }
        let scratches = try (0..<effectiveSlots).map { _ in
            try RawCompletionScratch(
                context: context, vocab: model.config.vocabSize,
                logitSoftcap: Float(model.config.finalLogitSoftcap))
        }
        let templateDigest = SHA256.hash(data: try Data(contentsOf: templateURL))
            .map { String(format: "%02x", $0) }
            .joined()
        let runtimeIdentity = [
            String(runtime.expertCacheSlots),
            runtime.expertCachePolicy.rawValue,
            runtime.rdadvisePolicy.rawValue,
            runtime.prefillPolicy.rawValue,
            String(runtime.prefillChunkTokens),
            runtime.headPath.rawValue,
            String(runtime.kvCachePrecision.rawValue),
            runtime.ropeScalingMode.rawValue,
            String(runtime.yarnContextTokens),
        ].joined(separator: ":")
        let runtimeDigest = SHA256.hash(data: Data(runtimeIdentity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let promptCacheDomain = ServerPromptCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: runtime.kvCachePrecision.label,
            fp16RingEnabled: runtime.fp16RingEnabled,
            templateSHA256: templateDigest)
        let effectivePromptCacheMode = Self.effectivePromptCacheMode(
            requested: promptCacheMode,
            mtpEnabled: mtpDecoder != nil,
            slots: effectiveSlots)
        let promptStateStore: ServerPromptStateStore?
        let promptCache: ServerPromptCache
        if effectivePromptCacheMode == .multiPrefix {
            let store = try ServerPromptStateStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: promptCacheMemoryLimitBytes,
                    diskDirectory: promptCacheDiskDirectory,
                    diskLimitBytes: promptCacheDiskLimitBytes))
            let persisted = store.loadEntries(domain: promptCacheDomain)
            if persisted.count > promptCacheMaximumEntries {
                store.remove(
                    entryIDs:
                        persisted
                        .dropLast(promptCacheMaximumEntries)
                        .map(\.id))
            }
            promptStateStore = store
            promptCache = ServerPromptCache(
                maximumEntries: promptCacheMaximumEntries,
                entries: persisted)
        } else {
            promptStateStore = nil
            promptCache = ServerPromptCache(maximumEntries: 1)
        }
        return ServerModelSession(
            context: context,
            model: model,
            tokenizer: tokenizer,
            tokenizerFolder: tokenizerFolder,
            loadedReasoning: RequestReasoning(
                thinkingMode: thinkingMode,
                effort: reasoningEffort),
            runner: runner,
            mtpDecoder: mtpDecoder,
            scratches: scratches,
            prefillConfig: runtime.prefillConfig,
            expertCacheSlots: loadSlots,
            slots: effectiveSlots,
            maxContext: maxContext,
            promptCacheMode: effectivePromptCacheMode,
            promptCacheDomain: promptCacheDomain,
            promptCache: promptCache,
            promptStateStore: promptStateStore,
            concisePrompt: conciseModeEnabled()
                ? ConcisePrompt.standard : nil)
    }
}
