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
/// Which codec the converted install carries.
public enum CompressExpertsCodec: String, Sendable {
    /// LZFSE frames, decoded on the CPU (the first format).
    case frames
    /// Static rANS blocks, decoded by a Metal kernel (the GPU format).
    case rans
}

public struct CompressExpertsOptions: Sendable {
    public let inputSSDAI: String
    public let output: String
    public let overwrite: Bool
    public let codec: CompressExpertsCodec

    public init(
        inputSSDAI: String, output: String, overwrite: Bool,
        codec: CompressExpertsCodec = .frames
    ) {
        self.inputSSDAI = inputSSDAI
        self.output = output
        self.overwrite = overwrite
        self.codec = codec
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
        let codec: CompressExpertsCodec
    }

    struct LayerPlan {
        let file: String
        let geometry: CompressedExpertLayerGeometry
        let ransBlocks: [RANSBlockSpec]
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
        // Built beside the destination and renamed onto it only once the
        // conversion, the manifest and the receipt are all complete. With
        // `--overwrite` that is what makes the flag mean "replace on success":
        // until the rename, the previous install keeps every byte (AUD-0076).
        let staging = try prepareOutput(options)
        do {
            let input = try loadInput(options.inputSSDAI, codec: options.codec)
            let rawBytes = try totalRawBytes(
                rawLayerBytes: input.rawLayerBytes, layerCount: input.layers.count)
            try copyPayload(from: input.root, to: staging)
            let layerResults = try compressLayers(
                input: input, output: staging, progress: progress)
            // The result's byte totals are reported to the operator. Summing
            // them before the receipt makes an (unreachable for real files)
            // overflow a failed conversion rather than a trap after success.
            let compressedBytes = try totalBytes(
                layerResults.map(\.result.compressedBytes), field: "compressed bytes")
            let indexBytes = try totalBytes(
                layerResults.map(\.result.indexBytes), field: "index bytes")
            let layoutBytes = try SSDAIJSON.markCompressed(
                layoutData: input.layoutData,
                compression: input.codec == .rans ? .rans : .lzfse)
            try writeLayout(layoutBytes, to: staging)
            try writeManifest(
                input: input, output: staging,
                layoutBytes: layoutBytes, layerResults: layerResults)
            let receipt = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputSSDAI: staging),
                recordedDirectory: options.output)
            try install(staging, at: options.output, overwrite: options.overwrite)
            return CompressExpertsResult(
                outputDirectory: options.output,
                layerCount: input.layers.count,
                rawBytes: rawBytes,
                compressedBytes: compressedBytes,
                indexBytes: indexBytes,
                receiptPath: receipt.receiptPath)
        } catch {
            // A failed conversion must not leave a directory that looks like an
            // install, and must not have disturbed the previous one. The
            // receipt above is the last thing written into the staging
            // directory, so a staging directory without one was never usable.
            try? FileManager.default.removeItem(atPath: staging)
            throw error
        }
    }

    /// The total raw bytes of a plan: every layer file has the same size.
    ///
    /// `rawLayerBytes` is `expertsPerLayer * expertStride`, already checked by
    /// `loadInput`; this product is the remaining unchecked one. It is refused
    /// before any copy or compression starts.
    static func totalRawBytes(rawLayerBytes: UInt64, layerCount: Int) throws -> UInt64 {
        guard layerCount >= 0 else {
            throw RepackError.configurationInvalid(
                detail: "a compression plan cannot have \(layerCount) layers")
        }
        let (total, overflow) = rawLayerBytes.multipliedReportingOverflow(
            by: UInt64(layerCount))
        guard !overflow else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(layerCount) layers of \(rawLayerBytes) raw bytes overflow the total")
        }
        return total
    }

    /// A checked sum of per-layer byte counts for the result summary.
    static func totalBytes(_ values: [UInt64], field: String) throws -> UInt64 {
        var total: UInt64 = 0
        for value in values {
            let (next, overflow) = total.addingReportingOverflow(value)
            guard !overflow else {
                throw RepackError.configurationInvalid(
                    detail: "\(field) total overflows at \(value)")
            }
            total = next
        }
        return total
    }

    /// Validate the input/output pairing and create the staging sibling.
    ///
    /// The existing destination is checked but never removed here: removing it
    /// up front is what lost both copies when a conversion failed (AUD-0076).
    /// Returns the directory the conversion writes into.
    static func prepareOutput(_ options: CompressExpertsOptions) throws -> String {
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
        }
        let staging = output.path + ".partial"
        guard staging != input.path, !staging.hasPrefix(input.path + "/"),
            !input.path.hasPrefix(staging + "/")
        else {
            throw RepackError.configurationInvalid(
                detail: "the staging directory \(staging) overlaps the input install")
        }
        // A killed conversion leaves this behind, and it is never a finished
        // install -- the rename onto `--output` is the last step -- so it is
        // this run's to reclaim.
        if FileManager.default.fileExists(atPath: staging) {
            try FileManager.default.removeItem(atPath: staging)
        }
        try FileManager.default.createDirectory(
            atPath: staging, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            atPath: staging + "/packed_experts", withIntermediateDirectories: true)
        return staging
    }

    /// Rename the finished build onto `--output`, replacing an existing install
    /// only now that the new one is complete.
    static func install(_ staging: String, at output: String, overwrite: Bool) throws {
        switch try Posix.entryKind(output) {
        case .absent:
            try Posix.rename(from: staging, to: output)
        default:
            guard overwrite else {
                throw RepackError.configurationInvalid(
                    detail: "\(output) appeared during the conversion; pass --overwrite "
                        + "to replace it")
            }
            // One atomic exchange, so a kill cannot leave the destination
            // missing: either the old install or the new one is there.
            try Posix.renameSwap(staging, output)
            // After the exchange the staging path holds the previous install.
            try? FileManager.default.removeItem(atPath: staging)
        }
        try Posix.fsyncDirectory((output as NSString).deletingLastPathComponent)
    }

    /// Read the manifest and layout, and check the raw contract they claim.
    static func loadInput(
        _ root: String, codec: CompressExpertsCodec = .frames
    ) throws -> Input {
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
                        splitBytes: split),
                    ransBlocks: RANSExpertLayerGeometry.blocks(for: layer)))
        }
        return Input(
            root: root, manifest: manifest, files: files, layout: layout,
            layoutData: layoutData, layers: layers, rawLayerBytes: rawLayerBytes,
            codec: codec)
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
            let result: CompressedExpertLayerResult
            switch input.codec {
            case .frames:
                result = try CompressedExpertLayerWriter.write(
                    sourcePath: source, destinationPath: destination,
                    geometry: layer.geometry)
            case .rans:
                let index = (destination as NSString).deletingPathExtension + ".ridx"
                let rans = try RANSExpertLayerWriter.write(
                    sourcePath: source, destinationPath: destination,
                    indexPath: index,
                    geometry: RANSExpertLayerGeometry(
                        expertStride: layer.geometry.expertStride,
                        expertsPerLayer: layer.geometry.expertsPerLayer,
                        blocks: layer.ransBlocks))
                result = CompressedExpertLayerResult(
                    compressedBytes: rans.compressedBytes,
                    indexBytes: rans.indexBytes,
                    sha256: rans.sha256,
                    indexSha256: rans.indexSha256,
                    blocksPerExpert: rans.blockCount,
                    blockBytes: 0)
            }
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
