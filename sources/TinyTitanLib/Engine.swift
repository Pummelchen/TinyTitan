// The engine an embedder holds: one install, one Metal device, one resident
// model.
//
// Part of the supported facade (phase A1 of `docs/plan-embedded-library.md`,
// §4). It is a thin, honest layer over the moved orchestrator: loading is
// `ServerModelSession.load`, which is the same path the server takes, and the
// descriptor is read from the install's manifest. No second implementation of
// anything lives here.
//
// Phase A1 gaps, all reported rather than papered over:
//
//   * The loading path builds its own `MetalContext`, which takes the system
//     default device, so the caller's device is honoured only when it is that
//     one; anything else is refused with `metalUnavailable`.
//   * `ServerModelSession.load` derives its streaming mode from the expert
//     cache slot count and resolves its integrity policy from the install, so
//     `EngineConfiguration` cannot expose those two knobs yet.
import Foundation
import Metal
import TinyTitan

/// How an engine loads and runs its model.
public struct EngineConfiguration: Sendable {
    /// The context window the session is loaded at.
    public var contextWindow: Int
    /// Bytes the routed-expert cache may use (`nil` = the install's own
    /// profile, which the loader derives from the manifest).
    public var expertCacheBudgetBytes: Int?
    /// KV-cache width.
    public var cachePrecision: CachePrecision
    /// How many generations the session may run at once. The loader clamps it
    /// to what the per-slot stores can hold beside the expert cache.
    public var maxConcurrentGenerations: Int
    /// Prefill chunk in tokens (`nil` = the install's own profile row, then the
    /// loader's family default). A larger chunk amortizes routed-expert reads
    /// over more tokens; a smaller one uses less GPU scratch.
    public var prefillChunkTokens: Int?
    /// Routed-expert cache slots per layer (`nil` = derive from
    /// `expertCacheBudgetBytes` or the install's tuned budget). The two knobs
    /// are alternatives, exactly as the loader's precedence states: an explicit
    /// count wins over a budget target.
    public var expertCacheSlots: Int?
    /// Context scaling. `.yarn` needs a context window from
    /// `RuntimeConfiguration.supportedYaRNContextTokens`.
    public var ropeScaling: RoPEScaling
    /// The reasoning switch the model is loaded at. It changes the rendered
    /// chat template, so it is a load-time decision, not a request one.
    public var thinkingMode: ThinkingMode
    /// Reasoning-effort level for a family whose template defines levels
    /// (`nil` = the template's own default).
    public var reasoningEffort: ReasoningEffort?
    /// Expert read-ahead advice. `nil` keeps `TINYTITAN_RDADVISE_POLICY` (or
    /// the loader's default); a value is an explicit override.
    public var readAhead: ReadAheadAdvice?
    /// Force the logits head instead of letting a pure-greedy generation use
    /// the fused greedy head. The fused head is faster but only exists for a
    /// 4-bit lm_head/attention install whose family has no hyper-connections;
    /// a caller that needs the logits buffer (diagnostics, or matching a run
    /// made against it) sets this. Sampling always forces the logits head.
    public var forceLogitsHead: Bool
    /// How much of the install to re-read before running it.
    ///
    /// There is no streaming-mode knob beside this one on purpose: the runtime's
    /// `ExpertStreamingMode` has a single case whose only parameter is the slot
    /// count, and `expertCacheSlots` and `expertCacheBudgetBytes` above are how
    /// a caller chooses it. A second name for the same decision would be a knob
    /// that lies about being a choice.
    public var integrityPolicy: InstallIntegrity
    /// Where this library's diagnostics go.
    ///
    /// `nil` — the default — writes them to stderr, which is what the server
    /// wants and what every line did before this knob existed. An embedder that
    /// owns its output passes its own sink, and `{ _ in }` silences the library
    /// entirely (which is what the CLI's `--quiet` does).
    ///
    /// The destination is **process-wide**: the orchestrator logs statically
    /// from deep inside the engine, so the most recently created `Engine` sets
    /// it for the process. Two engines with different sinks do not each get
    /// their own; the second one wins.
    public var logSink: (@Sendable (String) -> Void)?

    public init(
        contextWindow: Int = 262_144,
        expertCacheBudgetBytes: Int? = nil,
        cachePrecision: CachePrecision = .eightBit,
        maxConcurrentGenerations: Int = 1,
        prefillChunkTokens: Int? = nil,
        expertCacheSlots: Int? = nil,
        ropeScaling: RoPEScaling = .none,
        thinkingMode: ThinkingMode = .off,
        reasoningEffort: ReasoningEffort? = nil,
        readAhead: ReadAheadAdvice? = nil,
        forceLogitsHead: Bool = true,
        integrityPolicy: InstallIntegrity = .automatic,
        logSink: (@Sendable (String) -> Void)? = nil
    ) {
        self.contextWindow = contextWindow
        self.expertCacheBudgetBytes = expertCacheBudgetBytes
        self.cachePrecision = cachePrecision
        self.maxConcurrentGenerations = maxConcurrentGenerations
        self.prefillChunkTokens = prefillChunkTokens
        self.expertCacheSlots = expertCacheSlots
        self.ropeScaling = ropeScaling
        self.thinkingMode = thinkingMode
        self.reasoningEffort = reasoningEffort
        self.readAhead = readAhead
        self.forceLogitsHead = forceLogitsHead
        self.integrityPolicy = integrityPolicy
        self.logSink = logSink
    }
}

/// How much of an install the engine re-reads before it runs it.
///
/// The trade is startup time against trust: a 125B install is a quarter of a
/// terabyte, and hashing all of it is minutes, which is what the installer's
/// receipt exists to avoid.
public enum InstallIntegrity: Sendable, Equatable {
    /// Today's rule, and the default: trust the installer's receipt when the
    /// directory carries one — the check is then against the manifest and the
    /// install's path — and hash the payload when it does not.
    case automatic
    /// Hash the payload regardless of any receipt. Slow on a large install, and
    /// the point of the knob: a caller who does not trust the directory.
    case verifyEveryFile
    /// Trust the receipt's recorded digests. Fast, and strict: an install
    /// without a valid receipt is refused rather than quietly re-hashed, which
    /// would mask a moved directory or a tampered receipt and defeat the point.
    case trustInstallerReceipt

    var engineValue: ModelIntegrityPolicy? {
        switch self {
        case .automatic: nil  // the loader resolves it from the directory
        case .verifyEveryFile: .fullSha256
        case .trustInstallerReceipt: .sizeCheckTrustedReceipt
        }
    }
}

/// KV-cache precision in the kit's own vocabulary, so no engine enum appears
/// in a public signature.
public enum CachePrecision: Sendable, Equatable {
    case fourBit
    case eightBit
    case sixteenBit

    var engineValue: KVCachePrecision {
        switch self {
        case .fourBit: .int4
        case .eightBit: .int8
        case .sixteenBit: .fp16
        }
    }
}

/// Context scaling, in the kit's own vocabulary.
public enum RoPEScaling: Sendable, Equatable {
    case none
    case yarn

    var engineValue: RuntimeRoPEScalingMode {
        switch self {
        case .none: .none
        case .yarn: .yarn
        }
    }
}

/// The reasoning switch a model is loaded at.
public enum ThinkingMode: Sendable, Equatable {
    case off
    case on

    var engineValue: ModelThinkingMode { self == .on ? .on : .off }
}

/// A reasoning-effort level, for the families whose chat template defines one.
public enum ReasoningEffort: String, Sendable, Equatable, CaseIterable {
    case low
    case medium
    case xhigh

    var engineValue: ModelReasoningEffort {
        switch self {
        case .low: .low
        case .medium: .medium
        case .xhigh: .xhigh
        }
    }
}

/// Expert read-ahead advice, in the kit's own vocabulary.
public enum ReadAheadAdvice: String, Sendable, Equatable, CaseIterable {
    case off
    case `default`
    case bounded
    case adaptive

    var engineValue: RDAdvicePolicyMode {
        switch self {
        case .off: .off
        case .default: .default
        case .bounded: .bounded
        case .adaptive: .adaptive
        }
    }
}

/// One model's sampling row, in the kit's own vocabulary.
///
/// The facade has no engine enum in a public signature, so a caller that wants
/// to fill what it did not name asks the engine for this instead of reaching
/// into `GenerationDefaults` or `ModelProfile`.
public struct SamplingDefaults: Sendable, Equatable {
    public var temperature: Double
    public var topK: Int
    public var topP: Double
    /// OpenAI presence penalty: subtracted once per distinct id already in the
    /// history. Zero is neutral.
    public var presencePenalty: Double

    public init(
        temperature: Double,
        topK: Int,
        topP: Double,
        presencePenalty: Double = 0
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.presencePenalty = presencePenalty
    }

    init(_ sampling: GenerationDefaults.Sampling) {
        self.init(
            temperature: Double(sampling.temperature),
            topK: sampling.topK,
            topP: Double(sampling.topP),
            presencePenalty: Double(sampling.presencePenalty))
    }

    /// The row an install declares, read from `manifest.json` alone — no
    /// weights, no Metal, no load.
    ///
    /// A front end that has to fill `GenerationOptions` before the engine
    /// exists (the CLI, whose head selection depends on whether its plan is
    /// pure greedy) asks this; the value equals `Engine.samplingDefaults`
    /// afterwards. An unreadable install falls back to the house row, which is
    /// what the pre-facade CLI did.
    public static func forInstall(
        at directory: URL,
        thinkingMode: ThinkingMode = .off
    ) -> SamplingDefaults {
        let identity = try? ManifestReader.peekIdentity(directoryURL: directory)
        let profile =
            identity.map { ModelProfile.resolve(identity: $0).sampling }
            ?? GenerationDefaults.forFamily(.qwen36)
        return Engine.effectiveSampling(
            profile: profile,
            family: identity?.family,
            thinking: thinkingMode == .on)
    }
}

/// One loaded install and the device it runs on.
///
/// The engine owns the resident weights, the expert cache and the Metal
/// pipelines; a `Session` owns one conversation's state. Serialising through
/// the actor means a caller cannot touch the engine while it loads or unloads.
public actor Engine {
    /// What was loaded. Readable without awaiting the actor: it is fixed at
    /// init and survives `unload()`.
    public nonisolated let descriptor: ModelDescriptor

    /// The sampling row this install declares, resolved for the thinking mode
    /// the engine was loaded at. A caller that leaves a `GenerationOptions`
    /// field at its house value can fill it from here instead of reaching into
    /// the runtime.
    public nonisolated let samplingDefaults: SamplingDefaults

    private var modelSession: ServerModelSession?
    private let reasoningProfile: ServerReasoningProfile

    /// Loads `directory` on `device`.
    ///
    /// Throws `TinyTitanError` for the failures the facade can classify
    /// (`modelNotFound`, `notAnInstall`, `unsupportedFamily`,
    /// `metalUnavailable`, `cancelled`). A failure the engine reports with an
    /// internal error type is rethrown as it is, rather than flattened into a
    /// case that would claim more than the facade knows.
    public init(
        directory: URL,
        device: MTLDevice,
        configuration: EngineConfiguration = .init()
    ) async throws {
        // Before anything that can log: the loader's first line is the RAM
        // profile, and an embedder that installed a sink should not miss it.
        ServerLog.useSink(configuration.logSink)
        var isDirectory: ObjCBool = false
        guard
            FileManager.default.fileExists(
                atPath: directory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw TinyTitanError.modelNotFound(directory)
        }
        let family: ModelFamily
        do {
            family = try ManifestReader.peekFamily(directoryURL: directory)
        } catch {
            throw TinyTitanError.notAnInstall(directory)
        }
        // The loading path builds its own `MetalContext`, which takes the
        // system default device. Refusing a different one is the honest
        // option: ignoring it would run on a device the caller did not choose.
        guard let systemDevice = MTLCreateSystemDefaultDevice(),
            systemDevice.registryID == device.registryID
        else {
            throw TinyTitanError.metalUnavailable(
                reason: "the engine loads on the system default Metal device")
        }
        let arch = try Self.resolveArch(family: family, directory: directory)
        let manifest = try Self.readManifest(directory: directory, arch: arch)
        let loaded: ServerModelSession
        do {
            loaded = try await ServerModelSession.load(
                modelDirectory: directory,
                maxContext: configuration.contextWindow,
                slots: configuration.maxConcurrentGenerations,
                prefillChunkTokens: configuration.prefillChunkTokens,
                kvCachePrecision: configuration.cachePrecision.engineValue,
                ropeScalingMode: configuration.ropeScaling.engineValue,
                thinkingMode: configuration.thinkingMode.engineValue,
                reasoningEffort: configuration.reasoningEffort?.engineValue,
                integrityPolicy: configuration.integrityPolicy.engineValue,
                expertCacheSlots: configuration.expertCacheSlots,
                expertCacheBudgetBytes: configuration.expertCacheBudgetBytes,
                rdadvisePolicy: configuration.readAhead?.engineValue,
                forceLogitsHead: configuration.forceLogitsHead)
        } catch is ServerInferenceError {
            // The loader's only typed refusal is "this build cannot run it".
            throw TinyTitanError.unsupportedFamily(family: family.rawValue)
        } catch let error as GFTokenizerError {
            switch error {
            case .modelDirectoryNotFound:
                throw TinyTitanError.modelNotFound(directory)
            case .missingToolTemplate, .missingSpecialToken, .invalidChatTemplate:
                throw TinyTitanError.notAnInstall(directory)
            case .unsupportedForDialect:
                throw error
            }
        } catch is CancellationError {
            throw TinyTitanError.cancelled
        } catch let error as ModelError {
            guard let classified = Self.classify(error, directory: directory, family: family)
            else { throw error }
            throw classified
        }
        self.descriptor = Self.describe(
            session: loaded, manifest: manifest)
        self.reasoningProfile = ServerReasoningProfile(
            family: loaded.modelFamily,
            thinkingMode: loaded.loadedReasoning.thinkingMode,
            reasoningEffort: loaded.loadedReasoning.effort)
        self.samplingDefaults = Self.effectiveSampling(
            profile: loaded.samplingDefaults,
            family: loaded.modelFamily,
            thinking: loaded.loadedReasoning.thinkingMode.isEnabled)
        self.modelSession = loaded
    }

    /// A new conversation over the loaded model, with an optional system
    /// prompt prepended to every request.
    ///
    /// Cheap and non-throwing: the model is already resident, so the session
    /// only carries the system prompt and its own generation state. A session
    /// made after `unload()` is inert and throws `engineShutDown` on use.
    public func session(system: String? = nil) -> Session {
        Session(
            system: system,
            modelSession: modelSession,
            descriptor: descriptor,
            reasoningProfile: reasoningProfile)
    }

    /// Drops the engine's reference to the model.
    ///
    /// Sessions hold the model weakly, so once the last in-flight generation
    /// has drained the resident buffers and the expert cache go away with it;
    /// any session still alive then reports `engineShutDown`.
    public func unload() async {
        modelSession = nil
    }

    /// The sampling row an install declares, resolved for `thinkingMode`.
    ///
    /// Read from `manifest.json` alone — no weights, no Metal, no load — so a
    /// caller that must decide something before the engine exists (whether a
    /// pure-greedy plan may take the fused head) can still ask. An unreadable
    /// install falls back to the house row, exactly as the CLI did.
    static func effectiveSampling(
        profile: GenerationDefaults.Sampling,
        family: ModelFamily?,
        thinking: Bool
    ) -> SamplingDefaults {
        // Qwen3.8 publishes a row per mode and the profile carries only the
        // thinking one, so the mode picks between them here — the same choice
        // the server makes during validation.
        if let family, family == .qwen38flash || family == .qwen38flashMTP {
            return SamplingDefaults(
                GenerationDefaults.forFamily(family, thinking: thinking))
        }
        return SamplingDefaults(profile)
    }

    /// The smallest prefill chunk the engine accepts that covers `prompt`, or
    /// the largest allowed chunk when none does.
    ///
    /// `EngineConfiguration.prefillChunkTokens` is settled at load, so a front
    /// end whose flag means "size the chunk to the prompt" has to know the
    /// prompt's length before the engine exists. This answers that with the
    /// install's tokenizer alone — no weights, no Metal — so the flag keeps its
    /// meaning without the front end owning a tokenizer.
    public static func prefillChunk(
        covering prompt: Prompt,
        directory: URL,
        thinkingMode: ThinkingMode = .off,
        reasoningEffort: ReasoningEffort? = nil
    ) async throws -> Int {
        let tokenizer = try await GFTokenizer.load(
            forModelDirectory: directory,
            thinkingMode: thinkingMode.engineValue,
            reasoningEffort: reasoningEffort?.engineValue)
        let tokenCount: Int
        switch prompt {
        case .raw(let text):
            tokenCount = tokenizer.encode(text, addBOS: true).count
        case .messages(let messages):
            let rendered = try tokenizer.applyChatTemplate(
                messages.map {
                    GFTokenizer.Message(
                        role: $0.role.tokenizerRole, content: $0.content)
                })
            tokenCount = tokenizer.encode(rendered, addBOS: false).count
        }
        return RuntimeConfiguration.allowedPrefillChunkTokens
            .first { $0 >= tokenCount }
            ?? PrefillRuntimeConfig.maxChunkTokens
    }

    /// Turn the loader's own failure into the facade's, or `nil` to rethrow it.
    ///
    /// The taxonomy a caller acts on is three-way: a wrong format is the
    /// caller's to fix, a corrupt install is a re-download, and an architecture
    /// this build cannot run is a support question. Failures outside that
    /// taxonomy — a Metal command buffer that errored, a runtime invariant —
    /// say something about the machine rather than about the install, and are
    /// deliberately **not** flattened into a case that would claim more than the
    /// facade knows.
    ///
    /// `package` rather than private so the library's own tests can pin the
    /// mapping without a model on disk, and exhaustive rather than defaulted so
    /// a new loader failure has to be classified rather than inherited.
    package static func classify(
        _ error: ModelError,
        directory: URL,
        family: ModelFamily
    ) -> TinyTitanError? {
        switch error {
        case .notASSDAIDirectory, .unsupportedVersion, .unknownFlag:
            return .unsupportedFormat(detail: error.description)
        case .checksumMismatch(let file):
            return .integrityFailure(path: file, detail: error.description)
        case .trustedReceiptInvalid(let detail), .indexCorrupt(let detail):
            return .integrityFailure(path: directory.path, detail: detail)
        case .archMismatch, .unsupportedArchitecture:
            return .unsupportedFamily(family: family.rawValue)
        case .partialInstall, .missingFile, .tensorNotFound, .tensorSizeMismatch,
            .expertStrideNotPageAligned:
            return .notAnInstall(directory)
        case .residentBufferWrapFailed, .posixFailed, .expertCacheUnplaceable,
            .commandBufferFailed, .internalInconsistency:
            return nil
        }
    }

    private static func resolveArch(family: ModelFamily, directory: URL) throws -> ArchConfig {
        do {
            return try ArchConfig.resolved(forFamily: family, directoryURL: directory)
        } catch let error as ArchResolutionError {
            switch error {
            case .unsupportedFamily(let name):
                throw TinyTitanError.unsupportedFamily(family: name)
            case .manifestOmitsField:
                throw TinyTitanError.notAnInstall(directory)
            }
        } catch {
            throw TinyTitanError.notAnInstall(directory)
        }
    }

    private static func readManifest(directory: URL, arch: ArchConfig) throws -> Manifest {
        do {
            return try ManifestReader.load(directoryURL: directory, expecting: arch)
        } catch {
            throw TinyTitanError.notAnInstall(directory)
        }
    }

    private static func describe(
        session: ServerModelSession,
        manifest: Manifest
    ) -> ModelDescriptor {
        ModelDescriptor(
            id: session.defaultModelID,
            family: session.modelFamily.rawValue,
            contextWindow: session.maximumContext,
            weightBytes: manifest.files.values.reduce(UInt64(0)) { $0 + $1.size },
            expertCacheBytes: UInt64(session.expertCacheSlots) * manifest.expertStride
                * UInt64(manifest.arch.numLayers))
    }
}
