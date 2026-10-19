import Darwin
import Foundation
import TinyTitanFormat

// Payload copying and manifest rewriting for `--compress-experts`.
//
// Split out of `CompressExpertsTool.swift` so the orchestration reads in one
// screen; these are the two steps that touch the filesystem on both sides.

extension CompressExpertsTool {

    /// Copy every non-layer entry of the input install into the output.
    ///
    /// Files are hard-linked or APFS-cloned, never byte-copied: a 125B install
    /// carries a 95 GB n-gram table, and duplicating that to test a 4% decode
    /// win would make the tool unusable on exactly the models it is for. A
    /// volume change is refused rather than silently falling back to a copy.
    static func copyPayload(from input: String, to output: String) throws {
        let names = try FileManager.default.contentsOfDirectory(atPath: input).sorted()
        for name in names where !skippedRootEntries.contains(name) {
            try linkTree(from: input + "/" + name, to: output + "/" + name)
        }
    }

    static func linkTree(from source: String, to destination: String) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: source, isDirectory: &isDirectory) else {
            return
        }
        if isDirectory.boolValue {
            try FileManager.default.createDirectory(
                atPath: destination, withIntermediateDirectories: true)
            for name in try FileManager.default.contentsOfDirectory(atPath: source) {
                if name == ".DS_Store" { continue }
                try linkTree(from: source + "/" + name, to: destination + "/" + name)
            }
            return
        }
        if Darwin.link(source, destination) == 0 { return }
        let linkErrno = errno
        if clonefile(source, destination, 0) == 0 { return }
        throw RepackError.configurationInvalid(
            detail: "cannot link or clone \(source) into \(destination) "
                + "(link errno \(linkErrno), clone errno \(errno)); the output must be on "
                + "the input's volume, and this tool does not copy multi-gigabyte payloads")
    }

    /// Rewrite `manifest.json` with the compressed sizes and the index entries.
    ///
    /// The input manifest is edited as a JSON object rather than re-encoded from
    /// a typed plan: this tool has no plan (the source checkpoint is long gone),
    /// and every field it does not understand -- quant widths, arch geometry,
    /// the snapshot hash -- has to survive untouched for the install to load.
    static func writeManifest(
        input: Input,
        output: String,
        layoutBytes: Data,
        layerResults: [(file: String, result: CompressedExpertLayerResult)]
    ) throws {
        var manifest = input.manifest
        guard var files = manifest["files"] as? [String: Any] else {
            throw RepackError.configurationInvalid(detail: "manifest.json has no files map")
        }
        for (file, result) in layerResults {
            files["packed_experts/\(file)"] = entry(
                size: result.compressedBytes, sha256: result.sha256)
            let indexName =
                input.codec == .rans
                ? ExpertRANSIndex.sidecarName(forLayerFile: file)
                : ExpertBlobIndex.sidecarName(forLayerFile: file)
            files["packed_experts/\(indexName)"] = entry(
                size: result.indexBytes, sha256: result.indexSha256)
        }
        files["packed_experts/layout.json"] = entry(
            size: UInt64(layoutBytes.count),
            sha256: try sha256(of: layoutBytes, labelled: "packed_experts/layout.json"))
        manifest["files"] = files
        let data = try JSONSerialization.data(
            withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try Posix.atomicWrite(data, to: output + "/manifest.json", durableIn: output)
    }

    static func entry(size: UInt64, sha256: String) -> [String: Any] {
        ["size": NSNumber(value: size), "sha256": sha256]
    }

    static func sha256(of data: Data, labelled: String) throws -> String {
        var hasher = Sha256Stream()
        data.withUnsafeBytes { hasher.update($0) }
        let hex = hasher.finalizeHexString()
        guard !hex.isEmpty else {
            throw RepackError.configurationInvalid(
                detail: "\(labelled) hashed to nothing")
        }
        return hex
    }
}
