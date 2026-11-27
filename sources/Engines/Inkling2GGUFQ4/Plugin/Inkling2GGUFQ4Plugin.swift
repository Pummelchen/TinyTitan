import Foundation
import Metal
import TinyTitanCore
import TinyTitanPluginAPI

/// Inkling 2 Mini, Q4 `.ssdai` install, as a TPP/1 engine plugin (TT-130).
///
/// Everything model-specific is here: the probe, the parameter table, the
/// reasoning levels, the load plan (`InklingQ4LoadPlan`), the decoder and its
/// streamed experts, the completion loop and sampler, and the tokenizer derived
/// from the install's verified GGUF header. The host holds no Inkling code.
public enum Inkling2GGUFQ4Plugin: EnginePlugin {
    public static let pluginID = "ai.tinytitan.inkling2-gguf-q4"
    public static let protocolVersion = ProtocolVersion.current
    public static let displayName = "Inkling 2 Mini UD-Q4_K_XL (Unsloth)"

    /// Thinking off, or one of the template's effort levels (`high`, 0.9, is
    /// its own default).
    public static let reasoning = ReasoningControl.thinkingWithEffortLevels(
        defaultEffort: .high, supported: [.low, .medium, .high, .xhigh])

    public static func probe(_ install: InstallProbe) -> MatchStrength? {
        InklingQ4Install.recognizes(install.directory) ? .exact : nil
    }

    public static func parameters(for install: InstallProbe) -> ParameterSupport {
        ParameterSupport(
            kvCachePrecision: .supported,
            ropeScaling: .only(
                [.none], reason: "Inkling 2 has a 1M native window and no YaRN factor"),
            hostKVWindow: .refused(reason: "it needs a QSA model"),
            concurrentSequences: .supported,
            multiTokenPrediction: .refused(
                reason: "the Q4 install has no MTP head (the NVFP4 install's is TT-129)"),
            expertStreaming: .supported)
    }

    public static func reasoningControl(for install: InstallProbe) -> ReasoningControl? {
        reasoning
    }

    public static func chatDialect(for install: InstallProbe) -> (any ChatDialect)? {
        InklingDialect()
    }

    /// lint:allow-long one factory literal: every stage is a short closure over
    /// the variant, and splitting them would only move the list elsewhere.
    public static func engineFactory(host: HostServices) -> EngineFactory {
        @Sendable func metadata(_ directory: URL) throws -> EngineModelMetadata {
            let manifest = try SSDAIManifestReader.read(directoryURL: directory)
            return EngineModelMetadata(
                modelID: manifest.modelID, family: "inkling2", weightBits: 4,
                sampling: .house, reasoningControl: reasoning)
        }
        return EngineFactory(
            prepareLoad: { options in
                let manifest = try SSDAIManifestReader.read(directoryURL: options.modelDirectory)
                let workingSet = Int(
                    options.context?.device.recommendedMaxWorkingSetSize
                        ?? MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0)
                let plan = try InklingQ4LoadPlan.make(
                    geometry: try InklingQ4LoadPlan.Geometry(
                        directory: options.modelDirectory, kvPrecision: options.kvCachePrecision),
                    requested: options.maxContext, ramBudget: options.expertCacheBudgetBytes,
                    explicitSlots: options.expertCacheSlots, workingSet: workingSet,
                    maxContextExplicit: options.maxContextExplicit,
                    kvCachePrecision: options.kvCachePrecision)
                host.log(
                    "Inkling 2 Q4: \(options.kvCachePrecision.label) K/V, \(plan.slots) expert "
                        + "slots per layer, context \(plan.context) tokens (asked "
                        + "\(options.maxContext)), planned \(plan.plannedBytes >> 20) MiB")
                return EngineLoadPlan(
                    architecture: InklingQ4EngineModel.architecture(manifest.arch),
                    expertCacheSlots: plan.slots,
                    tunedExpertCacheBudgetBytes: InklingQ4LoadPlan.defaultExpertBytes,
                    prefillConfiguration: .defaultChunked, fp16RingEnabled: false,
                    yarnContextTokens: 0,
                    runtimeIdentity: "inkling2-gguf-q4-kv\(options.kvCachePrecision.rawValue)",
                    engineConfiguration: plan)
            },
            metadata: metadata,
            identity: { directory in
                let model = try metadata(directory)
                return ModelIdentity(
                    modelID: model.modelID, catalogID: model.modelID,
                    displayName: displayName,
                    family: model.family, weightBits: model.weightBits,
                    sampling: model.sampling, reasoningControl: model.reasoningControl)
            },
            validateReasoning: { _, thinking, effort in
                guard let effort else { return }
                guard thinking.isEnabled else {
                    throw EngineFactoryError.unsupported(
                        "reasoning effort \(effort.rawValue) needs thinking on for Inkling 2")
                }
                guard case .thinkingWithEffortLevels(_, let supported) = reasoning,
                    supported.contains(effort)
                else {
                    throw EngineFactoryError.unsupported(
                        "Inkling 2 has no reasoning effort \(effort.rawValue)")
                }
            },
            resolveArchitecture: { directory in
                InklingQ4EngineModel.architecture(
                    try SSDAIManifestReader.read(directoryURL: directory).arch)
            },
            preview: { directory, requestedChunk in
                EngineSessionPreview(
                    modelID: try metadata(directory).modelID,
                    prefillChunkTokens: requestedChunk ?? 256)
            },
            buildModel: { directory, _, _ in
                try InklingQ4EngineModel(directory: directory, slots: 0)
            },
            buildModelWithSlots: { directory, _, _, slots in
                try InklingQ4EngineModel(directory: directory, slots: slots)
            },
            buildRunner: { model, options in
                guard let model = model as? InklingQ4EngineModel, let context = options.context,
                    let plan = options.engineConfiguration as? InklingQ4LoadPlan
                else {
                    throw MetalError.invalidState(
                        "the Inkling 2 Q4 runner needs its model, plan and Metal context")
                }
                let decoder = try model.makeDecoder(
                    context: context, maxContext: plan.context,
                    maxChunk: options.prefillChunkTokens ?? 256,
                    sequenceSlots: options.slots,
                    kvCachePrecision: plan.kvCachePrecision)
                return InklingQ4Runner(model: model, decoder: decoder)
            },
            makeScratch: { context, vocabSize, _ in
                RawCompletionScratch(
                    payload: try InklingQ4LogitsScratch(context: context, vocabSize: vocabSize))
            },
            runRawCompletion: { request in try await runInklingQ4Completion(request) },
            perSlotBytes: { _, _, loadPlan in
                (loadPlan.engineConfiguration as? InklingQ4LoadPlan)?.perSlotBytes ?? 0
            },
            tokenizerFolder: { directory in try tokenizerFolder(directory, host: host) })
    }

    /// The install's tokenizer as a folder the session can load: exported once
    /// from the verified GGUF header (`tokenizer.json`, its config and the chat
    /// template) into the host's cache directory, keyed by the install's
    /// identity and the header's size so a different install never reuses it.
    static func tokenizerFolder(_ directory: URL, host: HostServices) throws -> URL {
        let header = directory.appendingPathComponent(InklingQ4Install.ggufHeaderFile)
        let manifest = try SSDAIManifestReader.read(directoryURL: directory)
        let size =
            (try? FileManager.default.attributesOfItem(atPath: header.path)[.size] as? Int) ?? 0
        let snapshot = (manifest.sourceSnapshotHash ?? "unpinned")
            .replacingOccurrences(of: "sha256:", with: "").prefix(16)
        let folder =
            host.cacheDirectory
            .appendingPathComponent("gguf-tokenizers", isDirectory: true)
            .appendingPathComponent(
                cacheComponent(modelID: manifest.modelID, snapshot: String(snapshot), size: size),
                isDirectory: true)
        let files = [
            GGUFTokenizerSidecar.tokenizerFileName, GGUFTokenizerSidecar.templateFileName,
        ]
        if !files.allSatisfy({
            FileManager.default.fileExists(atPath: folder.appendingPathComponent($0).path)
        }) {
            try GGUFTokenizerSidecar.write(
                from: try GGUFModelReader.readHeader(at: header), to: folder)
        }
        return folder
    }

    /// One filesystem-safe cache component for an install.
    ///
    /// `manifest.modelID` is not path text: it is whatever a checkpoint says,
    /// and this component decides where the host writes three files — reached
    /// from `--catalog` alone, before any load. A `modelID` containing `/` or
    /// `..` therefore wrote outside the cache (a `createDirectory` and two
    /// writes at an attacker-chosen path). Every character outside
    /// `[A-Za-z0-9._-]` becomes `-`, a `.`/`..` result is replaced, and a hash
    /// of the raw id keeps two ids that sanitise alike from colliding — a
    /// collision would reuse the wrong tokenizer, which is worse than a
    /// cosmetic name.
    static func cacheComponent(modelID: String, snapshot: String, size: Int) -> String {
        let allowed = modelID.map { character -> Character in
            guard character.isASCII else { return "-" }
            let keep =
                character.isLetter || character.isNumber || character == "-"
                || character == "_" || character == "."
            return keep ? character : "-"
        }
        let clipped = String(String(allowed).prefix(64))
        let usable = (clipped.isEmpty || clipped == "." || clipped == "..") ? "model" : clipped
        return "\(usable)-\(stableHash(modelID))-\(snapshot)-\(size)"
    }

    /// FNV-1a over the raw id, as eight hex digits: stable across processes
    /// and runs, which is what a cache key needs, and no dependency.
    static func stableHash(_ text: String) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in text.utf8 {
            hash = (hash ^ UInt64(byte)) &* 0x0000_0100_0000_01B3
        }
        return String(hash & 0xffff_ffff, radix: 16)
    }
}
