import Foundation
import TinyTitanFormat

/// Plans an Inkling 2 Mini `.ssdai` install from its GGUF (TT-121), keeping
/// every tensor in the encoding the GGUF stores it in: the routed experts stay
/// NVFP4 (Q8_0 in the last four layers), everything else stays Q8_0, F32 or
/// BF16. Nothing is requantized, so the install is the source's bytes
/// rearranged for streaming, and the plan is what the range-copy machinery
/// executes chunk by chunk.
///
/// - Resident: every non-expert tensor, in file order, each at a 64-byte
///   aligned offset in `model_weights.bin`, under its GGUF name.
/// - Experts: one blob per expert per layer — `gate_up`, `down`, and their
///   per-expert F32 scales when the layer has them — at a 16 KiB stride. NVFP4
///   and Q8_0 layers have different strides, so the layers are compact inside
///   the largest one.
/// - The GGUF header (metadata, tokenizer, chat template, tensor table) is
///   copied verbatim as `inkling2_gguf_header.bin`, so the variant builds its
///   tokenizer from the same bytes llama.cpp does.
/// - `arch.variantGeometry` carries the loop, relative-attention, controller and
///   activation constants, read from the GGUF's own metadata.
enum Inkling2GGUFPlanner {
    struct Source {
        /// The GGUF's file name, the shard id the copy plan reads.
        let fileName: String
        let header: GGUFHeaderV3
        let fileSize: UInt64
    }

    static let headerSidecarName = "inkling2_gguf_header.bin"
    static let residentAlignment: UInt64 = 64
    static let blobAlignment: UInt64 = 64

    /// How a ggml tensor type is kept: its index dtype code, and its block
    /// encoding when it has one (`nil` for plain element types).
    struct Kept {
        let code: UInt8
        let block: SSDAIBlockEncodingV1?
        let elementBytes: UInt64
        let bits: Int
    }

    static func kept(ggmlType: UInt32, tensor name: String) throws -> Kept {
        switch ggmlType {
        case 0: return Kept(code: 3, block: nil, elementBytes: 4, bits: 32)
        case 1: return Kept(code: 2, block: nil, elementBytes: 2, bits: 16)
        case 30: return Kept(code: 1, block: nil, elementBytes: 2, bits: 16)
        case 8: return Kept(code: 4, block: .ggufQ8_0, elementBytes: 0, bits: 8)
        case 40: return Kept(code: 5, block: .ggufNVFP4, elementBytes: 0, bits: 4)
        default:
            throw RepackError.configurationInvalid(
                detail:
                    "\(name): ggml type \(ggmlType) is not an encoding the Inkling install keeps")
        }
    }

    /// Bytes of a `rows x columns` matrix (ggml order: `dimensions[0]` is the
    /// row length; every later dimension multiplies into the rows).
    ///
    /// Every product is checked: the dimensions come from the GGUF header, so a
    /// hostile tensor can name values whose product wraps `UInt64`. The typed
    /// error is raised before the result can size a buffer or an offset.
    static func bytes(of dimensions: [UInt64], kept: Kept, tensor name: String) throws -> UInt64 {
        guard let columns = dimensions.first else {
            throw RepackError.configurationInvalid(detail: "\(name): no dimensions")
        }
        let rows = try dimensions.dropFirst().reduce(1) { total, dimension in
            try checkedProduct(
                total, dimension, tensor: name, field: "dimensions \(dimensions)")
        }
        if let block = kept.block {
            return try block.payloadBytes(rows: rows, columns: columns, field: name)
        }
        let values = try checkedProduct(
            rows, columns, tensor: name, field: "dimensions \(dimensions)")
        return try checkedProduct(
            values, kept.elementBytes, tensor: name,
            field: "dimensions \(dimensions) element size")
    }

    /// A product of two untrusted sizes, or a typed planner error naming the
    /// tensor and the field.
    private static func checkedProduct(
        _ lhs: UInt64, _ rhs: UInt64, tensor name: String, field: String
    ) throws -> UInt64 {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw RepackError.configurationInvalid(detail: "\(name): \(field) overflows")
        }
        return value
    }

    /// A sum of two untrusted sizes, or a typed planner error.
    private static func checkedAdd(
        _ lhs: UInt64, _ rhs: UInt64, tensor name: String, field: String
    ) throws -> UInt64 {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else {
            throw RepackError.configurationInvalid(detail: "\(name): \(field) overflows")
        }
        return value
    }

    /// A resident string-table offset: `count` bytes into the table.
    ///
    /// The resident index stores each name's offset as a `UInt32`, so a table
    /// that grows past `UInt32.max` cannot be represented. The accepted GGUF
    /// header budget (128 MiB) and the checkpoint index cap keep this
    /// unreachable for any accepted input, but the conversion is checked rather
    /// than trusting that bound to stay true.
    static func stringTableOffset(
        _ count: Int, field: String = "resident string table"
    ) throws -> UInt32 {
        guard count >= 0, let offset = UInt32(exactly: count) else {
            throw RepackError.configurationInvalid(
                detail: "\(field) offset \(count) exceeds UInt32")
        }
        return offset
    }

    /// `blk.<layer>.ffn_{gate_up,down}_exps.{weight,scale}` → (layer, sub-tensor key).
    static func expertKey(_ name: String) -> (layer: Int, key: String)? {
        let parts = name.split(separator: ".")
        guard parts.count == 4, parts[0] == "blk", let layer = Int(parts[1]) else { return nil }
        let role: String
        switch parts[2] {
        case "ffn_gate_up_exps": role = "gate_up"
        case "ffn_down_exps": role = "down"
        default: return nil
        }
        switch parts[3] {
        case "weight": return (layer, role)
        case "scale": return (layer, role + "_scale")
        default: return nil
        }
    }

    static func plan(source: Source, outputDir: String) throws -> RepackPlan {
        let header = source.header
        guard header.string("general.architecture") == "inkling2" else {
            throw RepackError.configurationInvalid(
                detail: "\(source.fileName) is not an Inkling 2 GGUF (architecture "
                    + "\(header.string("general.architecture") ?? "none"))")
        }
        let arch = try Inkling2GGUFArch.archInfo(header)
        guard arch.numLayers > 0, arch.numExperts > 0 else {
            throw RepackError.configurationInvalid(
                detail:
                    "Inkling 2 GGUF declares \(arch.numLayers) layers and "
                    + "\(arch.numExperts) experts; both must be positive")
        }
        var residentTensors: [GGUFHeaderV3.Tensor] = []
        var expertTensors: [Int: [String: GGUFHeaderV3.Tensor]] = [:]
        for tensor in header.tensors {
            if let key = expertKey(tensor.name) {
                expertTensors[key.layer, default: [:]][key.key] = tensor
            } else {
                residentTensors.append(tensor)
            }
        }
        let resident = try planResident(
            residentTensors.sorted { $0.absoluteOffset < $1.absoluteOffset },
            source: source, outputDir: outputDir)
        var layers: [LayerFilePlan] = []
        for layer in 0..<arch.numLayers {
            guard let tensors = expertTensors[layer] else {
                throw RepackError.configurationInvalid(
                    detail: "layer \(layer) has no routed experts")
            }
            layers.append(
                try planLayer(
                    layer, tensors: tensors, experts: arch.numExperts,
                    source: source, outputDir: outputDir))
        }
        var plan = RepackPlan(
            arch: arch, baseMode: SSDAIBlockEncodingV1.ggufQ8_0.rawValue, baseGroupSize: 32,
            resident: resident, layers: layers, matchedModelID: nil,
            excludedMultimodalTensorNames: [],
            passthroughFiles: [
                PassthroughFile(
                    sourceName: source.fileName, destinationName: headerSidecarName,
                    size: header.dataStart, required: true)
            ])
        plan.slotSchemes = ["routedExpert": SSDAIBlockEncodingV1.ggufNVFP4.rawValue]
        return plan
    }

    private static func sourceTensor(
        _ tensor: GGUFHeaderV3.Tensor, size: UInt64, source: Source
    ) throws -> SourceTensor {
        let (end, overflow) = tensor.absoluteOffset.addingReportingOverflow(size)
        guard !overflow, end <= source.fileSize else {
            throw RepackError.configurationInvalid(
                detail: "\(tensor.name) ends at \(end), past the \(source.fileSize)-byte file")
        }
        return SourceTensor(
            name: tensor.name, shardPath: source.fileName, dtype: .opaque,
            shape: tensor.dimensions.reversed(), absoluteOffset: tensor.absoluteOffset,
            sizeBytes: size)
    }

    private static func planResident(
        _ tensors: [GGUFHeaderV3.Tensor], source: Source, outputDir: String
    ) throws -> ResidentFilePlan {
        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        for tensor in tensors {
            offsets.append(try stringTableOffset(stringTable.count))
            stringTable.append(contentsOf: tensor.name.utf8)
        }
        let (indexEntries, entriesOverflow) = tensors.count.multipliedReportingOverflow(
            by: SSDAIBinary.indexEntryBytes)
        let (indexBody, headerOverflow) = SSDAIBinary.indexHeaderBytes.addingReportingOverflow(
            indexEntries)
        let (rawIndexValue, rawOverflow) = indexBody.addingReportingOverflow(stringTable.count)
        guard !entriesOverflow, !headerOverflow, !rawOverflow, rawIndexValue >= 0 else {
            throw RepackError.configurationInvalid(
                detail: "the resident index of \(tensors.count) tensors overflows")
        }
        let rawIndex = UInt64(rawIndexValue)
        let indexSize = try roundUp(
            rawIndex, to: Layout.pageBytes, field: "resident index size")
        var cursor = indexSize
        var entries: [ResidentEntry] = []
        for tensor in tensors {
            let kept = try kept(ggmlType: tensor.ggmlType, tensor: tensor.name)
            let size = try bytes(of: tensor.dimensions, kept: kept, tensor: tensor.name)
            guard tensor.dimensions.count <= 4 else {
                throw RepackError.configurationInvalid(detail: "\(tensor.name): rank above 4")
            }
            cursor = try roundUp(
                cursor, to: residentAlignment, field: "\(tensor.name) resident offset")
            var shape: [UInt32] = []
            for dimension in tensor.dimensions.reversed() {
                guard let narrowed = UInt32(exactly: dimension) else {
                    throw RepackError.configurationInvalid(
                        detail: "\(tensor.name): dimension \(dimension) exceeds UInt32")
                }
                shape.append(narrowed)
            }
            entries.append(
                ResidentEntry(
                    name: tensor.name, dtype: kept.code,
                    logicalShape4: shape + [UInt32](repeating: 0, count: 4 - shape.count),
                    fileOffset: cursor, sizeBytes: size,
                    scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: try sourceTensor(tensor, size: size, source: source),
                    sourceScales: nil, sourceBiases: nil))
            cursor = try checkedAdd(
                cursor, size, tensor: tensor.name, field: "resident offset")
        }
        return ResidentFilePlan(
            path: (outputDir as NSString).appendingPathComponent("model_weights.bin"),
            entries: entries, stringTable: stringTable, stringTableOffsets: offsets,
            indexSize: indexSize, residentSize: cursor - indexSize)
    }

    private static func planLayer(
        _ layer: Int, tensors: [String: GGUFHeaderV3.Tensor], experts: Int,
        source: Source, outputDir: String
    ) throws -> LayerFilePlan {
        guard experts > 0 else {
            throw RepackError.configurationInvalid(
                detail: "layer \(layer): \(experts) experts")
        }
        var slices: [PerExpertTensorSlice] = []
        var cursor: UInt64 = 0
        // Weights first, then their scales, each 64-byte aligned in the blob.
        for key in ["gate_up", "down", "gate_up_scale", "down_scale"] {
            guard let tensor = tensors[key] else {
                if key.hasSuffix("_scale") { continue }
                throw RepackError.configurationInvalid(detail: "layer \(layer) has no \(key)")
            }
            guard tensor.dimensions.last == UInt64(experts) else {
                throw RepackError.configurationInvalid(
                    detail:
                        "\(tensor.name): \(tensor.dimensions) does not end in \(experts) experts")
            }
            let kept = try kept(ggmlType: tensor.ggmlType, tensor: tensor.name)
            // A per-expert scale is one value per expert: `[experts]` leaves a
            // one-element tensor, not an empty shape.
            let perExpertDimensions =
                tensor.dimensions.count == 1 ? [UInt64(1)] : Array(tensor.dimensions.dropLast())
            let perExpert = try bytes(of: perExpertDimensions, kept: kept, tensor: tensor.name)
            let isScale = key.hasSuffix("_scale")
            cursor = try roundUp(
                cursor, to: blobAlignment, field: "layer \(layer) \(key) offset")
            slices.append(
                PerExpertTensorSlice(
                    role: isScale ? String(key.dropLast("_scale".count)) : key,
                    component: isScale ? "scale" : "weights",
                    dtype: kept.code,
                    logicalShape: perExpertDimensions.reversed(),
                    offsetInExpertBlob: cursor, sizeInExpertBlob: perExpert,
                    sourceOffsetPerExpert: perExpert,
                    sourceTensor: try sourceTensor(
                        tensor,
                        size: try checkedProduct(
                            perExpert, UInt64(experts), tensor: tensor.name,
                            field: "per-expert bytes"),
                        source: source),
                    bitsForWeights: isScale ? nil : kept.bits))
            cursor = try checkedAdd(
                cursor, perExpert, tensor: tensor.name, field: "layer \(layer) \(key) end")
        }
        return LayerFilePlan(
            layerIndex: layer,
            path: (outputDir as NSString).appendingPathComponent(
                String(format: "packed_experts/layer_%02d.bin", layer)),
            expertsPerLayer: experts,
            expertStride: try roundUp(
                cursor, to: Layout.pageBytes, field: "layer \(layer) expert stride"),
            subTensors: slices)
    }

    /// Rounds `value` up to `alignment`, refusing a value that cannot be
    /// represented instead of trapping on the `+ alignment - 1` intermediate.
    static func roundUp(
        _ value: UInt64, to alignment: UInt64, field: String = "offset"
    ) throws -> UInt64 {
        guard alignment > 0 else {
            throw RepackError.configurationInvalid(detail: "\(field): alignment is zero")
        }
        let (padded, overflow) = value.addingReportingOverflow(alignment - 1)
        guard !overflow else {
            throw RepackError.configurationInvalid(
                detail: "\(field): \(value) cannot be rounded to \(alignment)")
        }
        return (padded / alignment) * alignment
    }
}
