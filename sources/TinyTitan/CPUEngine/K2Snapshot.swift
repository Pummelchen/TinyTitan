import Foundation

/// The K2-Horizon snapshot: `config.json`, a quantized safetensors shard set,
/// and the accessors the CPU model reads.
///
/// The converter keeps the checkpoint's own tensor names, so this reader is
/// written against them: `model.layers.N.self_attn.q_proj.weight`,
/// `model.layers.N.mlp.experts.E.gate_proj.weight`, and so on. Quantized
/// matrices are packed `uint32` weights with BF16 `.scales`/`.biases` beside
/// them, group 64, the same affine layout the Qwen snapshots use.
///
/// The file is mapped, not read: the 4-bit snapshot is ~22 GB and the 8-bit
/// one ~40 GB, and only the active experts of a token are touched, so the page
/// cache is the right owner of the pages.
public struct K2Snapshot: Sendable {

    public struct Configuration: Sendable {
        public let hiddenSize: Int
        public let layers: Int
        public let heads: Int
        public let keyValueHeads: Int
        public let headDim: Int
        public let ropeDim: Int
        public let vocabSize: Int
        public let intermediateSize: Int
        public let moeIntermediateSize: Int
        public let numExperts: Int
        public let topKExperts: Int
        public let movaExperts: Int
        public let movaTopK: Int
        public let sharedExperts: Int
        public let normGroups: Int
        public let epsilon: Float
        public let ropeTheta: Float
        public let tieEmbeddings: Bool
        public let denseLayers: [Int]
        public let attentionGate: String
        public let routerScaling: Float
        public let routerBias: Bool
        public let normTopK: Bool
        public let eosTokens: [Int]
        public let maxPositions: Int
    }

    public let directory: URL
    public let configuration: Configuration
    public let baseBits: Int
    public let groupSize: Int

    private let shards: [String: SafeTensorsFile]
    private let placement: [String: String]
    private let widths: [String: Int]

    public init(directory: URL, maxBytes: UInt64 = ManifestReader.defaultMaxBytes) throws {
        self.directory = directory
        let configData = try BoundedMetadataRead.read(
            fileAt: directory.appendingPathComponent("config.json"), maxBytes: maxBytes)
        guard
            let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
        else {
            throw SafeTensorsFile.Failure.malformed("config.json is not an object")
        }
        func integer(_ key: String, _ fallback: Int? = nil) throws -> Int {
            if let value = config[key] as? Int { return value }
            if let fallback { return fallback }
            throw SafeTensorsFile.Failure.malformed("config.json lacks \(key)")
        }
        func double(_ key: String, _ fallback: Double) -> Double {
            (config[key] as? NSNumber)?.doubleValue ?? fallback
        }
        let rope = (config["rope_parameters"] as? [String: Any]) ?? [:]
        let theta =
            (rope["rope_theta"] as? NSNumber)?.doubleValue
            ?? (config["rope_theta"] as? NSNumber)?.doubleValue ?? 10_000_000
        let headDim = try integer("head_dim")
        let eos: [Int]
        if let one = config["eos_token_id"] as? Int {
            eos = [one]
        } else if let many = config["eos_token_id"] as? [Int] {
            eos = many
        } else {
            eos = [1]
        }
        configuration = Configuration(
            hiddenSize: try integer("hidden_size"),
            layers: try integer("num_hidden_layers"),
            heads: try integer("num_attention_heads"),
            keyValueHeads: try integer("num_key_value_heads"),
            headDim: headDim,
            ropeDim: try integer("rope_head_dim", headDim),
            vocabSize: try integer("vocab_size"),
            intermediateSize: try integer("intermediate_size"),
            moeIntermediateSize: try integer("moe_intermediate_size"),
            numExperts: try integer("num_experts"),
            topKExperts: try integer("num_experts_per_tok"),
            movaExperts: try integer("mova_num_experts"),
            movaTopK: try integer("mova_num_experts_per_tok"),
            sharedExperts: try integer("num_shared_experts", 0),
            normGroups: try integer("layernorm_num_groups", 1),
            epsilon: Float(double("rms_norm_eps", 1e-6)),
            ropeTheta: Float(theta),
            tieEmbeddings: (config["tie_word_embeddings"] as? Bool) ?? false,
            denseLayers: (config["mlp_only_layers"] as? [Int]) ?? [],
            attentionGate: (config["attention_gate_func"] as? String) ?? "none",
            routerScaling: Float(double("router_scaling_factor", 1)),
            routerBias: (config["moe_gate_bias"] as? Bool) ?? false,
            normTopK: (config["norm_topk_prob"] as? Bool) ?? false,
            eosTokens: eos,
            maxPositions: try integer("max_position_embeddings", 32_768))

        guard let quantization = config["quantization"] as? [String: Any],
            let bits = quantization["bits"] as? Int,
            let group = quantization["group_size"] as? Int
        else {
            throw SafeTensorsFile.Failure.malformed("config.json lacks a quantization block")
        }
        baseBits = bits
        groupSize = group
        var overrides: [String: Int] = [:]
        for (stem, value) in quantization {
            if let entry = value as? [String: Any], let width = entry["bits"] as? Int {
                overrides[stem] = width
            }
        }
        widths = overrides

        let indexData = try BoundedMetadataRead.read(
            fileAt: directory.appendingPathComponent("model.safetensors.index.json"),
            maxBytes: maxBytes)
        guard let index = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
            let map = index["weight_map"] as? [String: String]
        else {
            throw SafeTensorsFile.Failure.malformed("index has no weight_map")
        }
        placement = map
        var opened: [String: SafeTensorsFile] = [:]
        for file in Set(map.values) {
            opened[file] = try SafeTensorsFile(url: directory.appendingPathComponent(file))
        }
        shards = opened
    }

    private func shard(_ name: String) throws -> SafeTensorsFile {
        guard let file = placement[name], let shard = shards[file] else {
            throw SafeTensorsFile.Failure.missing(name)
        }
        return shard
    }

    public func has(_ name: String) -> Bool { placement[name] != nil }

    public func floats(_ name: String) throws -> [Float] {
        try shard(name).floats(name)
    }

    /// The width a matrix's stem is quantized at: the base width, or the
    /// per-stem override the converter wrote for the embedding, the head and
    /// the bf16 keeps (which are not matrices and never reach this).
    public func bits(forStem stem: String) -> Int {
        widths[stem] ?? baseBits
    }

    /// A quantized matrix by its weight name.
    public func matrix(_ name: String) throws -> AffineSnapshot.Matrix {
        let stem = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
        let file = try shard(name)
        let entry = try file.entry(name)
        guard entry.shape.count == 2 else {
            throw SafeTensorsFile.Failure.malformed("\(name) is not a matrix")
        }
        let width = bits(forStem: stem)
        guard width == 4 || width == 8 else {
            throw SafeTensorsFile.Failure.malformed("\(stem): no CPU kernel for \(width) bits")
        }
        let lanes = 32 / width
        let rows = entry.shape[0]
        let columns = entry.shape[1] * lanes
        guard columns % groupSize == 0 else {
            throw SafeTensorsFile.Failure.malformed(
                "\(stem): width \(columns) at \(width) bits is not a whole number "
                    + "of \(groupSize)-element groups")
        }
        let weights = try file.bytes(name)
        guard weights.count == rows * columns * width / 8 else {
            throw SafeTensorsFile.Failure.malformed(
                "\(stem): \(weights.count) packed bytes for \(rows)x\(columns) at \(width) bits")
        }
        // A writer may flush between a weight and its companions, so they are
        // looked up on their own shard rather than assumed to sit beside it.
        let scales = try shard(stem + ".scales").bytes(stem + ".scales")
        let biases = try shard(stem + ".biases").bytes(stem + ".biases")
        let groups = rows * (columns / groupSize)
        guard scales.count == groups * 2, biases.count == groups * 2,
            let scalesBase = scales.baseAddress?.assumingMemoryBound(to: UInt16.self),
            let biasesBase = biases.baseAddress?.assumingMemoryBound(to: UInt16.self)
        else {
            throw SafeTensorsFile.Failure.malformed("\(stem): scales or biases do not match")
        }
        return AffineSnapshot.Matrix(
            weights: weights, scales: scalesBase, biases: biasesBase,
            rows: rows, columns: columns, bits: width, groupSize: groupSize)
    }

    /// One row of a quantized matrix, dequantized to FP32.
    ///
    /// The embedding is a gather, not a GEMV: one row of 2,560 values, read
    /// with its own group scales. This is the same arithmetic the kernels
    /// apply, on one row.
    public func row(_ name: String, _ index: Int) throws -> [Float] {
        let stem = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
        let file = try shard(name)
        let entry = try file.entry(name)
        let width = bits(forStem: stem)
        let lanes = 32 / width
        let columns = entry.shape[1] * lanes
        guard index >= 0, index < entry.shape[0] else {
            throw SafeTensorsFile.Failure.malformed("\(name): row \(index) out of range")
        }
        let weights = try file.bytes(name)
        let bytesPerRow = columns * width / 8
        let groups = columns / groupSize
        let scales = try shard(stem + ".scales").bytes(stem + ".scales")
        let biases = try shard(stem + ".biases").bytes(stem + ".biases")
        guard let weightsBase = weights.baseAddress?.assumingMemoryBound(to: UInt8.self),
            let scalesBase = scales.baseAddress?.assumingMemoryBound(to: UInt16.self),
            let biasesBase = biases.baseAddress?.assumingMemoryBound(to: UInt16.self)
        else {
            throw SafeTensorsFile.Failure.malformed("\(name): no storage")
        }
        func bf16(_ bits: UInt16) -> Float {
            Float(bitPattern: UInt32(bits) << 16)
        }
        var values = [Float](repeating: 0, count: columns)
        for group in 0..<groups {
            let scale = bf16(scalesBase[index * groups + group])
            let bias = bf16(biasesBase[index * groups + group])
            for offset in 0..<groupSize {
                let column = group * groupSize + offset
                let byte: UInt8
                if width == 8 {
                    byte = weightsBase[index * bytesPerRow + column]
                    values[column] = Float(byte) * scale + bias
                } else {
                    byte = weightsBase[index * bytesPerRow + column / 2]
                    let nibble = column % 2 == 0 ? byte & 0x0F : byte >> 4
                    values[column] = Float(nibble) * scale + bias
                }
            }
        }
        return values
    }
}
