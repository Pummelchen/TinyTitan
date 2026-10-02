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

    public init(
        contextWindow: Int = 262_144,
        expertCacheBudgetBytes: Int? = nil,
        cachePrecision: CachePrecision = .eightBit,
        maxConcurrentGenerations: Int = 1
    ) {
        self.contextWindow = contextWindow
        self.expertCacheBudgetBytes = expertCacheBudgetBytes
        self.cachePrecision = cachePrecision
        self.maxConcurrentGenerations = maxConcurrentGenerations
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

/// One loaded install and the device it runs on.
///
/// The engine owns the resident weights, the expert cache and the Metal
/// pipelines; a `Session` owns one conversation's state. Serialising through
/// the actor means a caller cannot touch the engine while it loads or unloads.
public actor Engine {
    /// What was loaded. Readable without awaiting the actor: it is fixed at
    /// init and survives `unload()`.
    public nonisolated let descriptor: ModelDescriptor

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
                kvCachePrecision: configuration.cachePrecision.engineValue,
                expertCacheBudgetBytes: configuration.expertCacheBudgetBytes)
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
        }
        self.descriptor = Self.describe(
            session: loaded, manifest: manifest)
        self.reasoningProfile = ServerReasoningProfile(
            family: loaded.modelFamily,
            thinkingMode: loaded.loadedReasoning.thinkingMode,
            reasoningEffort: loaded.loadedReasoning.effort)
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
