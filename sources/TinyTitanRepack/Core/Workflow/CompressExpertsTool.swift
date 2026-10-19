import Foundation
import TinyTitanFormat

/// Compress an installed model's routed-expert layers in place of a re-download.
///
/// The read side has always been able to open a compressed layer through its
/// `.idx` sidecar; nothing could *produce* one, because the installer writes
/// experts as range copies and a compressed frame is not a byte range of the
/// source. This tool is the writing half: it takes an installed raw model (the
/// only thing every existing install has) and emits a second install whose
/// layer files are LZFSE frames plus one index per layer.
///
/// It never touches the input. That matters twice over: the raw install stays
/// runnable, which is what a paired A/B needs, and a failed conversion cannot
/// leave the only copy of a 17 GB model half-rewritten.
///
/// **It is not a speed win, and that is measured.** Ornith 4-bit converts at
/// 91.1% of the expert bytes (18.12 -> 16.50 GB, 94.6 s for 40 layers) and
/// decodes at 7.91 tok/s against the raw install's 13.76 -- a 42.5% loss, because
/// the read path decodes both frames before returning and serialises read with
/// decode. That is a pipeline result, not a codec one: the same frames decode at
/// 5.61 GB/s on eight workers, so the design note's overlapped wave has a 4-8%
/// ceiling. The tool exists so the result stays reproducible; do not convert an
/// install expecting it to get faster until that pipeline exists.
/// `docs/llm-compression-design.md` has the budget and the fixes.
public struct CompressExpertsOptions: Sendable {
    public let inputSSDAI: String
    public let output: String
    public let overwrite: Bool

    public init(inputSSDAI: String, output: String, overwrite: Bool) {
        self.inputSSDAI = inputSSDAI
        self.output = output
        self.overwrite = overwrite
    }
}

public struct CompressExpertsProgress: Sendable {
    public let layer: Int
    public let layerCount: Int
    public let file: String
    public let rawBytes: UInt64
    public let compressedBytes: UInt64
    public let elapsedSeconds: Double
}

public struct CompressExpertsResult: Sendable {
    public let outputDirectory: String
    public let layerCount: Int
    public let rawBytes: UInt64
    public let compressedBytes: UInt64
    public let indexBytes: UInt64
    public let receiptPath: String
}

public enum CompressExpertsTool {
    /// Everything the conversion reads from the input install.
    struct Input {
        let root: String
        let manifest: [String: Any]
        let files: [String: Any]
        let layout: SSDAIPackedExpertsLayoutV1
        let layoutData: Data
        let layers: [LayerPlan]
        let rawLayerBytes: UInt64
    }

    struct LayerPlan {
        let file: String
        let geometry: CompressedExpertLayerGeometry
    }

    /// Files an install may carry that are not payload of the copy: the two
    /// documents this tool rewrites, plus Finder and lock droppings.
    static let skippedRootEntries: Set<String> = [
        "packed_experts", "manifest.json", "verified-install.json", ".DS_Store",
        ".install.lock",
    ]

    public static func run(
        options: CompressExpertsOptions,
        progress: (CompressExpertsProgress) -> Void = { _ in }
    ) throws -> CompressExpertsResult {
        try prepareOutput(options)
        do {
            let input = try loadInput(options.inputSSDAI)
            try copyPayload(from: input.root, to: options.output)
            let layerResults = try compressLayers(
                input: input, output: options.output, progress: progress)
            let layoutBytes = try SSDAIJSON.markCompressed(layoutData: input.layoutData)
            try writeLayout(layoutBytes, to: options.output)
            try writeManifest(
                input: input, output: options.output,
                layoutBytes: layoutBytes, layerResults: layerResults)
            let receipt = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputSSDAI: options.output))
            return CompressExpertsResult(
                outputDirectory: options.output,
                layerCount: input.layers.count,
                rawBytes: input.rawLayerBytes * UInt64(input.layers.count),
                compressedBytes: layerResults.reduce(0) { $0 + $1.result.compressedBytes },
                indexBytes: layerResults.reduce(0) { $0 + $1.result.indexBytes },
                receiptPath: receipt.receiptPath)
        } catch {
            // A failed conversion must not leave a directory that looks like an
            // install. The receipt above is the last thing written, so a
            // directory without one was never a usable output.
            try? FileManager.default.removeItem(atPath: options.output)
            throw error
        }
    }

    /// Validate the input/output pairing and create a clean output directory.
    static func prepareOutput(_ options: CompressExpertsOptions) throws {
        let input = URL(fileURLWithPath: options.inputSSDAI).standardizedFileURL
        let output = URL(fileURLWithPath: options.output).standardizedFileURL
        guard input.path != output.path else {
            throw RepackError.configurationInvalid(
                detail: "--compress-experts rewrites a *copy*; --input-ssdai and "
                    + "--output must differ")
        }
        guard !output.path.hasPrefix(input.path + "/"),
            !input.path.hasPrefix(output.path + "/")
        else {
            throw RepackError.configurationInvalid(
                detail: "the compressed install must not sit inside the input "
                    + "install (or the other way round)")
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: input.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw RepackError.configurationInvalid(
                detail: "\(input.path) is not a directory")
        }
        if FileManager.default.fileExists(atPath: output.path) {
            guard options.overwrite else {
                throw RepackError.configurationInvalid(
                    detail: "\(output.path) already exists; pass --overwrite to replace it")
            }
            try FileManager.default.removeItem(atPath: output.path)
        }
        try FileManager.default.createDirectory(
            atPath: output.path, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            atPath: output.appendingPathComponent("packed_experts").path,
            withIntermediateDirectories: true)
    }

    /// Read the manifest and layout, and check the raw contract they claim.
    static func loadInput(_ root: String) throws -> Input {
        let manifestData = try Posix.readBoundedData(
            root + "/manifest.json",
            maximumBytes: VerifiedInstallTool.metadataMaxBytes)
        guard let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any]
        else {
            throw RepackError.configurationInvalid(detail: "manifest.json is not an object")
        }
        guard let files = manifest["files"] as? [String: Any] else {
            throw RepackError.configurationInvalid(detail: "manifest.json has no files map")
        }
        let layoutData = try Posix.readBoundedData(
            root + "/packed_experts/layout.json",
            maximumBytes: VerifiedInstallTool.metadataMaxBytes)
        let layout = try SSDAIPackedExpertsLayoutCodec.decode(layoutData)
        guard layout.compression == nil else {
            throw RepackError.configurationInvalid(
                detail: "\(root) is already compressed (layout.json says "
                    + "\(layout.compression?.rawValue ?? ""))")
        }
        guard layout.expertsPerLayer > 0, layout.expertStride > 0 else {
            throw RepackError.configurationInvalid(
                detail: "\(root) packs no routed experts; there is nothing to compress")
        }
        let (rawLayerBytes, overflow) = UInt64(layout.expertsPerLayer)
            .multipliedReportingOverflow(by: layout.expertStride)
        guard !overflow else {
            throw RepackError.configurationInvalid(
                detail: "an expert layer of \(layout.expertsPerLayer) experts of "
                    + "\(layout.expertStride) bytes overflows")
        }
        var layers: [LayerPlan] = []
        layers.reserveCapacity(layout.layers.count)
        for layer in layout.layers {
            let relativePath = "packed_experts/\(layer.file)"
            guard let entry = files[relativePath] as? [String: Any],
                let size = (entry["size"] as? NSNumber)?.uint64Value
            else {
                throw RepackError.configurationInvalid(
                    detail: "manifest.json has no entry for \(relativePath)")
            }
            guard size == rawLayerBytes else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) is \(size) bytes; a raw layer is "
                        + "\(rawLayerBytes)")
            }
            let actual =
                try FileManager.default
                .attributesOfItem(atPath: root + "/" + relativePath)[.size] as? NSNumber
            guard actual?.uint64Value == size else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) on disk is \(actual?.uint64Value ?? 0) bytes; "
                        + "the manifest says \(size)")
            }
            let split = ExpertBlobIndex.phase1Bytes(
                downOffset: layer.experts.first?.tensors["down"]?.offset,
                expertStride: layout.expertStride)
            layers.append(
                LayerPlan(
                    file: layer.file,
                    geometry: CompressedExpertLayerGeometry(
                        expertStride: layout.expertStride,
                        expertsPerLayer: layout.expertsPerLayer,
                        splitBytes: split)))
        }
        return Input(
            root: root, manifest: manifest, files: files, layout: layout,
            layoutData: layoutData, layers: layers, rawLayerBytes: rawLayerBytes)
    }

    /// Write the layer files, in layout order.
    static func compressLayers(
        input: Input,
        output: String,
        progress: (CompressExpertsProgress) -> Void
    ) throws -> [(file: String, result: CompressedExpertLayerResult)] {
        var results: [(file: String, result: CompressedExpertLayerResult)] = []
        results.reserveCapacity(input.layers.count)
        let started = Date()
        for (index, layer) in input.layers.enumerated() {
            let source = input.root + "/packed_experts/" + layer.file
            let destination = output + "/packed_experts/" + layer.file
            let result = try CompressedExpertLayerWriter.write(
                sourcePath: source, destinationPath: destination, geometry: layer.geometry)
            results.append((layer.file, result))
            progress(
                CompressExpertsProgress(
                    layer: index, layerCount: input.layers.count, file: layer.file,
                    rawBytes: layer.geometry.rawBytes,
                    compressedBytes: result.compressedBytes,
                    elapsedSeconds: Date().timeIntervalSince(started)))
        }
        return results
    }

    static func writeLayout(_ data: Data, to output: String) throws {
        try Posix.atomicWrite(
            data, to: output + "/packed_experts/layout.json",
            durableIn: output + "/packed_experts")
    }
}
