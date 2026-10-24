import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanFormat

/// The `lowmem` install profile: routed experts stored at 2 bits and read at
/// 2 bits, with the resident tiers unchanged.
///
/// The format half is exercised through the same codec the repacker and the
/// runtime use. The layout half goes through `PackedExpertsLayoutReader` with a
/// real `Manifest`, because the recipe is a *pair* of declarations — the
/// manifest's profile and every tensor's `bits` — and either one alone can
/// describe an install the kernels would read wrongly.
@Suite struct LowMemoryProfileTests {

    private static func lowMemoryQuant(routedBits: Int = 2) -> [String: Any] {
        var quant = ManifestReaderTests.quant()
        quant["routedExpert"] = ManifestReaderTests.quantSlot(routedBits)
        return quant
    }

    /// A layer whose nine tensors describe `bits`-bit weights. Shape and byte
    /// extent agree, so the only thing under test is the profile contract.
    private static func layerJSON(
        layer: Int, experts: Int, stride: Int, bits: Int
    ) -> [String: Any] {
        let weightBytes = 64 * 64 * bits / 8
        let auxBytes = 64 * 2
        var expertList: [[String: Any]] = []
        for expert in 0..<experts {
            var tensors: [String: Any] = [:]
            var cursor = 0
            for role in ["gate", "up", "down"] {
                tensors[role] = [
                    "offset": cursor, "size": weightBytes,
                    "dtype": "U32", "shape": [64, 64], "bits": bits,
                ]
                cursor += weightBytes
                tensors["\(role)_scales"] = [
                    "offset": cursor, "size": auxBytes,
                    "dtype": "BF16", "shape": [64, 1],
                ]
                cursor += auxBytes
                tensors["\(role)_biases"] = [
                    "offset": cursor, "size": auxBytes,
                    "dtype": "BF16", "shape": [64, 1],
                ]
                cursor += auxBytes
            }
            expertList.append([
                "expert": expert,
                "offset": expert * stride,
                "size": stride,
                "tensors": tensors,
            ])
        }
        return [
            "layer": layer,
            "file": "layer_\(layer).bin",
            "experts": expertList,
        ]
    }

    static func layoutJSON(
        layers: Int, experts: Int, stride: Int, bits: Int,
        compression: Bool = false
    ) throws -> Data {
        var layersArray: [[String: Any]] = []
        for layer in 0..<layers {
            layersArray.append(
                layerJSON(
                    layer: layer, experts: experts, stride: stride, bits: bits))
        }
        var root: [String: Any] = [
            "expertStride": stride,
            "numLayers": layers,
            "expertsPerLayer": experts,
            "layers": layersArray,
        ]
        if compression { root["compression"] = "lzfse" }
        return try JSONSerialization.data(withJSONObject: root)
    }

    private static func wireManifest(
        _ overrides: [String: Any] = [:]
    ) throws -> SSDAIManifestV1 {
        let (dir, _) = try ManifestReaderTests.writeToyManifest(overrides)
        defer { try? FileManager.default.removeItem(at: dir) }
        let data = try Data(contentsOf: dir.appendingPathComponent("manifest.json"))
        return try SSDAIManifestCodec.decode(data)
    }

    /// A toy lowmem manifest with layer files sized for the uniform layout the
    /// tests below decode. `crossValidate` reads those manifest sizes, so a
    /// layout the reader would otherwise accept is refused if they disagree.
    private static func manifestDirectory(
        quant: [String: Any], memoryProfile: String?
    ) throws -> (URL, ArchConfig) {
        let toy = ArchConfig.qwenToy()
        var files: [String: [String: Any]] = [
            "model_weights.bin": [
                "size": 1_024, "sha256": String(repeating: "0", count: 64),
            ],
            "packed_experts/layout.json": [
                "size": 1_024, "sha256": String(repeating: "0", count: 64),
            ],
        ]
        for layer in 0..<toy.numLayers {
            files["packed_experts/layer_\(layer).bin"] = [
                "size": toy.numExperts * 16_384,
                "sha256": String(repeating: "0", count: 64),
            ]
        }
        var overrides: [String: Any] = ["quant": quant]
        if let memoryProfile { overrides["memoryProfile"] = memoryProfile }
        return try ManifestReaderTests.writeToyManifest(
            overrides, filesOverride: files)
    }

    private static func lowMemoryManifestDirectory() throws -> (URL, ArchConfig) {
        try manifestDirectory(quant: lowMemoryQuant(), memoryProfile: "lowmem")
    }

    // MARK: - Format

    @Test func lowMemoryProfileRoundTripsThroughTheManifestCodec() throws {
        let wire = try Self.wireManifest([
            "quant": Self.lowMemoryQuant(),
            "memoryProfile": "lowmem",
        ])
        #expect(wire.resolvedMemoryProfile == .lowmem)
        #expect(wire.quant?.routedExpert.weightBits == 2)

        // Encoding then decoding must keep the profile and the 2-bit slot, or
        // a repack would silently drop the marker and the next reader would
        // treat the 2-bit blobs as an expandable cold tier.
        let encoded = try JSONEncoder().encode(wire)
        let decoded = try SSDAIManifestCodec.decode(encoded)
        #expect(decoded.resolvedMemoryProfile == .lowmem)
        #expect(decoded.quant?.routedExpert.weightBits == 2)
    }

    @Test func absentProfileMeansStandard() throws {
        let wire = try Self.wireManifest()
        #expect(wire.memoryProfile == nil)
        #expect(wire.resolvedMemoryProfile == .standard)
    }

    @Test func anUnknownProfileIsNotSilentlyStandard() throws {
        let (dir, _) = try ManifestReaderTests.writeToyManifest([
            "quant": Self.lowMemoryQuant(),
            "memoryProfile": "someday",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        let data = try Data(contentsOf: dir.appendingPathComponent("manifest.json"))
        #expect(throws: (any Error).self) {
            _ = try SSDAIManifestCodec.decode(data)
        }
    }

    @Test func twoBitExpertsRequireTheLowMemoryMarker() throws {
        let (dir, _) = try ManifestReaderTests.writeToyManifest([
            "quant": Self.lowMemoryQuant()
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .qwenToy())
            Issue.record("a standard manifest with 2-bit routed experts must be refused")
        } catch let error as ModelError {
            let text = "\(error)"
            #expect(text.contains("lowmem"), "the refusal must name the profile, got \(text)")
        }
    }

    @Test func lowMemoryMustActuallyBeTwoBit() throws {
        let (dir, _) = try ManifestReaderTests.writeToyManifest([
            "quant": Self.lowMemoryQuant(routedBits: 4),
            "memoryProfile": "lowmem",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: (any Error).self) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .qwenToy())
        }
    }

    @Test func lowMemoryDoesNotNarrowTheResidentTiers() throws {
        var quant = Self.lowMemoryQuant()
        quant["attention"] = ManifestReaderTests.quantSlot(6)
        let (dir, _) = try ManifestReaderTests.writeToyManifest([
            "quant": quant, "memoryProfile": "lowmem",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: (any Error).self) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .qwenToy())
        }
    }

    @Test func theExperimentalSwitchFailsClosed() {
        #expect(SSDAIMemoryProfile.standard.isServable(environment: [:]))
        #expect(!SSDAIMemoryProfile.lowmem.isServable(environment: [:]))
        #expect(
            !SSDAIMemoryProfile.lowmem.isServable(
                environment: [SSDAIMemoryProfile.experimentalOptInVariable: "true"]))
        #expect(
            SSDAIMemoryProfile.lowmem.isServable(
                environment: [SSDAIMemoryProfile.experimentalOptInVariable: "1"]))
    }

    // MARK: - Loader width acceptance

    @Test func loaderAcceptsTheLowMemoryProfileAndReportsTwoBits() throws {
        let (dir, toy) = try ManifestReaderTests.writeToyManifest([
            "quant": Self.lowMemoryQuant(),
            "memoryProfile": "lowmem",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(manifest.memoryProfile == .lowmem)
        #expect(manifest.quant?.routedExpert.weightBits == 2)

        let identity = try ManifestReader.peekIdentity(directoryURL: dir)
        #expect(identity.weightBits == 2)
        #expect(identity.memoryProfile == .lowmem)
    }

    // MARK: - Layout recipe

    @Test func lowMemoryAcceptsAUniformTwoBitLayout() throws {
        let (dir, toy) = try Self.lowMemoryManifestDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: toy)
        let data = try Self.layoutJSON(
            layers: toy.numLayers, experts: toy.numExperts, stride: 16_384, bits: 2)
        let layout = try PackedExpertsLayoutReader.decode(data: data, manifest: manifest)
        #expect(layout.expertStride == 16_384)
        #expect(layout.layers[0].experts[0].subTensors["gate"]?.bits == 2)
        #expect(layout.layers[0].experts[0].size == 16_384)
    }

    @Test func aUniformTwoBitLayoutWithoutTheMarkerIsRefused() throws {
        // This is the dangerous shape: standard plus all-cold would run the
        // 2-bit-to-4-bit expansion against itself.
        let (dir, toy) = try Self.manifestDirectory(
            quant: ManifestReaderTests.quant(), memoryProfile: nil)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: toy)
        let data = try Self.layoutJSON(
            layers: toy.numLayers, experts: toy.numExperts, stride: 16_384, bits: 2)
        do {
            _ = try PackedExpertsLayoutReader.decode(data: data, manifest: manifest)
            Issue.record("a uniform cold layer under the standard profile must be refused")
        } catch {
            #expect("\(error)".contains("expert 0"), "got \(error)")
        }
    }

    @Test func lowMemoryRejectsAHotExpertInItsMidst() throws {
        // A 2-bit layer with one 4-bit expert is a tiered layer, not the
        // uniform profile; the native 2-bit kernel would read the odd blob at
        // the wrong stride.
        let (dir, toy) = try Self.lowMemoryManifestDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: toy)

        var root = try #require(
            try JSONSerialization.jsonObject(
                with: try Self.layoutJSON(
                    layers: toy.numLayers, experts: toy.numExperts,
                    stride: 16_384, bits: 2)) as? [String: Any])
        var layers = try #require(root["layers"] as? [[String: Any]])
        var experts = try #require(layers[0]["experts"] as? [[String: Any]])
        var tensors = try #require(experts[1]["tensors"] as? [String: Any])
        var gate = try #require(tensors["gate"] as? [String: Any])
        gate["bits"] = 4
        tensors["gate"] = gate
        experts[1]["tensors"] = tensors
        layers[0]["experts"] = experts
        root["layers"] = layers
        let data = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: (any Error).self) {
            _ = try PackedExpertsLayoutReader.decode(data: data, manifest: manifest)
        }
    }

    @Test func lowMemoryRejectsCompression() throws {
        // The compressed readers never run the expert width path, so a
        // compressed lowmem layer would hand 2-bit frames to the 4-bit slot.
        let (dir, toy) = try Self.lowMemoryManifestDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: toy)
        let data = try Self.layoutJSON(
            layers: toy.numLayers, experts: toy.numExperts,
            stride: 16_384, bits: 2, compression: true)
        #expect(throws: (any Error).self) {
            _ = try PackedExpertsLayoutReader.decode(data: data, manifest: manifest)
        }
    }

    @Test func lowMemoryRejectsACompactUnevenOffset() throws {
        let (dir, toy) = try Self.lowMemoryManifestDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: toy)
        var root = try #require(
            try JSONSerialization.jsonObject(
                with: try Self.layoutJSON(
                    layers: toy.numLayers, experts: toy.numExperts,
                    stride: 16_384, bits: 2)) as? [String: Any])
        var layers = try #require(root["layers"] as? [[String: Any]])
        var experts = try #require(layers[0]["experts"] as? [[String: Any]])
        experts[1]["offset"] = 1_024
        layers[0]["experts"] = experts
        root["layers"] = layers
        let data = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: (any Error).self) {
            _ = try PackedExpertsLayoutReader.decode(data: data, manifest: manifest)
        }
    }
}
