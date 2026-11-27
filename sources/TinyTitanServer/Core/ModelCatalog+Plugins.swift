import Foundation
import TinyTitanCore
import TinyTitanLib
import TinyTitanPluginAPI
import TinyTitanVariants

/// Install probing from the resolved plugin's identity, and on-disk sizing.
/// Split out of `ModelCatalog.swift` under the 500-line-per-file rule (TT-126);
/// the identity read moved here in TT-134 group (f).
extension ModelCatalog {
    static func probeInstall(_ directory: URL) -> Result<Entry, ProbeFailure> {
        let resolved: ResolvedPlugin
        do {
            resolved = try EngineFactories.pluginRegistry.resolve(directory)
        } catch let error as PluginRegistryError {
            switch error {
            case .noPlugin:
                return .failure(
                    ProbeFailure(
                        reason: "no engine plugin serves this install (\(directory.path))"))
            case .ambiguous, .incompatible:
                return .failure(ProbeFailure(reason: "unreadable manifest.json: \(error)"))
            }
        } catch {
            return .failure(ProbeFailure(reason: "unreadable manifest.json: \(error)"))
        }
        let identity: ModelIdentity
        do {
            guard let declared = try resolved.modelIdentity() else {
                return .failure(
                    ProbeFailure(
                        reason: "the engine plugin declares no identity for this install"))
            }
            identity = declared
        } catch {
            return .failure(ProbeFailure(reason: "unreadable manifest.json: \(error)"))
        }
        // A family no front end can serve is skipped here rather than offered
        // and refused on first use. The plugin/adapter owns that judgement.
        if let reason = identity.unsupportedReason {
            return .failure(ProbeFailure(reason: reason))
        }
        let factory = resolved.engineFactory()
        // A plugin's hook is a closure, not proof the install can tokenize. The
        // catalog must ask the cheapest question that is still true: the
        // plugin's own source predicate when it has one (the Inkling 2 Q4
        // install's GGUF header), otherwise the hook itself -- calling it is
        // what distinguishes "declares a hook" from "can produce a tokenizer".
        let hasTokenizer: Bool
        if let sourceIsPresent = factory.tokenizerSourceIsPresent {
            guard sourceIsPresent(directory) else {
                return .failure(
                    ProbeFailure(
                        reason:
                            "plugin tokenizer source is missing; the install is incomplete"))
            }
            hasTokenizer = true
        } else if let hook = factory.tokenizerFolder {
            do {
                _ = try hook(directory)
                hasTokenizer = true
            } catch {
                return .failure(
                    ProbeFailure(reason: "plugin tokenizer source is unreadable: \(error)"))
            }
        } else {
            hasTokenizer =
                GFTokenizer.tokenizerFolder(forModelDirectory: directory) != nil
        }
        guard hasTokenizer else {
            return .failure(
                ProbeFailure(reason: "no tokenizer/tokenizer.json; the install is incomplete"))
        }
        let preferred = identity.engines.first ?? .gpu
        let engines = identity.engines.isEmpty ? [preferred] : identity.engines
        let kind: Kind
        switch preferred {
        case .gpu:
            kind = .gpu(family: identity.family, reasoning: identity.reasoningControl)
        case .cpu:
            kind = .cpu(family: identity.family, reasoning: identity.reasoningControl)
        }
        return .success(
            Entry(
                id: identity.catalogID,
                name: identity.displayName,
                kind: kind,
                quant: identity.weightBits,
                path: directory,
                sampling: identity.sampling,
                contextLimit: identity.contextLimit,
                sizeBytes: sizeOnDisk(directory),
                engines: engines.map(backend(for:))))
    }

    package static func backend(for engine: ModelEngine) -> Backend {
        switch engine {
        case .gpu: .gpu
        case .cpu: .cpu
        }
    }

    /// Allocated bytes, so an APFS clone or a sparse file reports what it
    /// really occupies rather than its logical length.
    static func sizeOnDisk(_ directory: URL) -> Int64 {
        let keys: [URLResourceKey] = [.totalFileAllocatedSizeKey, .fileSizeKey, .isRegularFileKey]
        guard
            let walker = FileManager.default.enumerator(
                at: directory, includingPropertiesForKeys: keys)
        else {
            return 0
        }
        var total: Int64 = 0
        for case let file as URL in walker {
            guard let values = try? file.resourceValues(forKeys: Set(keys)),
                values.isRegularFile == true
            else { continue }
            total += Int64(values.totalFileAllocatedSize ?? values.fileSize ?? 0)
        }
        return total
    }
}
