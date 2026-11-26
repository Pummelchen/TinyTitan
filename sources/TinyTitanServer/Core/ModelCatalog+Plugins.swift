import Foundation
import TinyTitanCore
import TinyTitanPluginAPI

/// Plugin-backed install probing and on-disk sizing, split out of
/// `ModelCatalog.swift` under the 500-line-per-file rule (TT-126).
extension ModelCatalog {
    static func probePluginInstall(
        _ directory: URL,
        resolved: ResolvedPlugin
    ) -> Result<Entry, ProbeFailure> {
        let factory = resolved.engineFactory()
        let metadata: EngineModelMetadata
        do {
            metadata = try factory.metadata(directory)
        } catch {
            return .failure(ProbeFailure(reason: "unreadable plugin metadata: \(error)"))
        }
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
        return .success(
            Entry(
                id: metadata.modelID,
                name: resolved.plugin.displayName,
                kind: .plugin(family: metadata.family, reasoning: metadata.reasoningControl),
                quant: metadata.weightBits,
                path: directory,
                sampling: metadata.sampling,
                contextLimit: nil,
                sizeBytes: sizeOnDisk(directory)))
    }

    /// Allocated bytes, so an APFS clone or a sparse file reports what it
    /// really occupies rather than its logical length.
    static func sizeOnDisk(_ directory: URL) -> Int64 {
        let keys: [URLResourceKey] = [.totalFileAllocatedSizeKey, .fileSizeKey, .isRegularFileKey]
        guard
            let walker = FileManager.default.enumerator(
                at: directory, includingPropertiesForKeys: keys)
        else { return 0 }
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
