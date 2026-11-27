import Foundation
import Testing
@testable import TinyTitanCore

@testable import TinyTitanLib
@testable import TinyTitanServerCore
import TinyTitanVariants

/// Builds a models directory by hand: the shapes the catalog must recognise,
/// and the ones it must skip without failing.
private enum CatalogFixture {
    static func root() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-catalog-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func write(_ object: Any, to url: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: url)
    }

    static func directory(_ root: URL, _ name: String) throws -> URL {
        let url = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func slot(_ bits: Int) -> [String: Any] {
        [
            "weightBits": bits, "scheme": "affine", "scaleType": "bf16", "biasType": "bf16",
            "groupSize": 64,
        ]
    }

    /// A manifest the real reader accepts, with the family and width under test.
    static func install(
        _ root: URL, _ name: String, modelID: String, family: String,
        bits: Int, tokenizer: Bool = true
    ) throws {
        let url = try directory(root, name)
        let arch: [String: Any] = [
            "hiddenSize": 64, "ffnIntermediate": 128, "moeIntermediateSize": 128,
            "numHeads": 4, "numKVHeads": 2, "numFullKVHeads": 2, "headDim": 32,
            "fullHeadDim": 32, "vocabSize": 1024, "slidingWindow": 1024,
            "finalLogitSoftcap": 0.0, "ropeTheta": 10_000_000.0, "fullRopeTheta": 10_000_000.0,
            "partialRotaryFactor": 0.25, "numLayers": 4, "numExperts": 8, "topKExperts": 8,
            "tieWordEmbeddings": false, "attentionKEqV": false, "hiddenActivation": "silu",
            "fullAttentionLayerMask": [2, 1, 2, 1], "family": family,
        ]
        try write(
            [
                "magic": "SSDAI", "versionMajor": 1, "versionMinor": 0,
                "flags": ["streamingPresent": true], "modelID": modelID, "arch": arch,
                "files": [
                    "model_weights.bin": [
                        "size": 1024, "sha256": String(repeating: "0", count: 64),
                    ]
                ],
                "expertsPerLayer": 8, "numLayers": 4, "expertStride": 16384,
                "quant": [
                    "embedding": slot(8), "attention": slot(8), "router": slot(8),
                    "sharedExpert": slot(bits), "routedExpert": slot(bits),
                ],
            ] as [String: Any], to: url.appendingPathComponent("manifest.json"))
        if tokenizer {
            let folder = try directory(url, "tokenizer")
            try Data("{}".utf8).write(to: folder.appendingPathComponent("tokenizer.json"))
        }
    }

    /// The Ornith install the shipped plugin claims, so the catalog can probe
    /// it without running a model.
    static func ornith(
        _ root: URL, _ name: String, tokenizer: Bool = true
    ) throws {
        try install(
            root, name, modelID: "ornith-1.5-35b-a3b", family: "qwen36",
            bits: 4, tokenizer: tokenizer)
    }
}

@Suite("Model catalog")
struct ModelCatalogTests {
    @Test func recognisesInstallsAndSkipsTheRest() throws {
        let root = try CatalogFixture.root()
        defer { try? FileManager.default.removeItem(at: root) }
        try CatalogFixture.ornith(root, "ornith-1.5_35B_A3B_4Bit")
        try CatalogFixture.install(
            root, "qwen3.6_35B_A3B_MTP_4Bit", modelID: "qwen3.6-35b-a3b-mtp-4bit",
            family: "qwen36_mtp", bits: 4)
        try CatalogFixture.ornith(root, "ornith-no-tokenizer", tokenizer: false)
        try CatalogFixture.install(
            root, "qwen3.6_35B_A3B_8Bit", modelID: "qwen3.6-35b-a3b",
            family: "qwen36", bits: 8)
        // A safetensors snapshot has no manifest and is not an install: the
        // CPU engine that served it left with TT-142 slice (b).
        let snapshot = try CatalogFixture.directory(root, "qwen3.5_2B_4Bit")
        try CatalogFixture.write(
            ["model_type": "qwen3_5_dense", "hidden_size": 2048],
            to: snapshot.appendingPathComponent("config.json"))
        let broken = try CatalogFixture.directory(root, "broken")
        try Data("not json".utf8).write(to: broken.appendingPathComponent("manifest.json"))
        _ = try CatalogFixture.directory(root, "gguf")
        _ = try CatalogFixture.directory(root, ".cache")
        try Data().write(to: root.appendingPathComponent("qwen3.6_35B_A3B_8Bit.install.lock"))

        let catalog = ModelCatalog.scan(directory: root)

        #expect(catalog.entries.map(\.id) == ["ornith-1.5-35b-a3b_4-Bit"])
        let gpu = try #require(catalog.entry(id: "ornith-1.5-35b-a3b_4-Bit"))
        #expect(gpu.name == "Ornith 1.5 35B-A3B")
        #expect(gpu.kind == .gpu(family: "qwen36", reasoning: .binaryThinking))
        #expect(gpu.quant == 4)
        #expect(gpu.sampling == .house)
        #expect(gpu.sizeBytes > 0)

        let skipped = Dictionary(
            uniqueKeysWithValues: catalog.skipped.map {
                ($0.path.lastPathComponent, $0.reason)
            })
        #expect(
            Set(skipped.keys) == [
                "qwen3.6_35B_A3B_8Bit", "qwen3.6_35B_A3B_MTP_4Bit",
                "ornith-no-tokenizer", "qwen3.5_2B_4Bit", "broken", "gguf",
            ])
        #expect(skipped["qwen3.6_35B_A3B_8Bit"]?.contains("no engine plugin serves") == true)
        #expect(skipped["qwen3.6_35B_A3B_MTP_4Bit"]?.contains("no engine plugin serves") == true)
        #expect(skipped["ornith-no-tokenizer"]?.contains("no tokenizer") == true)
        #expect(skipped["qwen3.5_2B_4Bit"]?.contains("no manifest.json") == true)
    }

    @Test func skippedDirectoriesAreReportedOnceInOneNotice() throws {
        let root = try CatalogFixture.root()
        defer { try? FileManager.default.removeItem(at: root) }
        try CatalogFixture.ornith(root, "good")
        _ = try CatalogFixture.directory(root, "gguf")
        let catalog = ModelCatalog.scan(directory: root)
        let pipe = Pipe()
        catalog.reportSkipped(to: pipe.fileHandleForWriting)
        try pipe.fileHandleForWriting.close()
        let text = pipe.fileHandleForReading.readDataToEndOfFile().lossyUTF8String
        #expect(text.hasPrefix("catalog: skipped 1 of 2 directories"))
        #expect(text.contains("gguf: no manifest.json"))
    }

    @Test func aModelLinkedInFromElsewhereIsServed() throws {
        let root = try CatalogFixture.root()
        let elsewhere = try CatalogFixture.root()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: elsewhere)
        }
        try CatalogFixture.ornith(elsewhere, "ornith-1.5_35B_A3B_4Bit")
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("linked"),
            withDestinationURL: elsewhere.appendingPathComponent("ornith-1.5_35B_A3B_4Bit"))
        let catalog = ModelCatalog.scan(directory: root)
        #expect(catalog.entries.map(\.id) == ["ornith-1.5-35b-a3b_4-Bit"])
        #expect(catalog.skipped.isEmpty)
    }

    @Test func aMissingDirectoryIsReportedNotFatal() {
        let catalog = ModelCatalog.scan(
            directory: URL(fileURLWithPath: "/nonexistent/models-\(UUID())"))
        #expect(catalog.entries.isEmpty)
        #expect(catalog.skipped.count == 1)
    }

    @Test func aDuplicateIdIsServedOnce() throws {
        let root = try CatalogFixture.root()
        defer { try? FileManager.default.removeItem(at: root) }
        try CatalogFixture.ornith(root, "a")
        try CatalogFixture.ornith(root, "b")
        let catalog = ModelCatalog.scan(directory: root)
        #expect(catalog.entries.map(\.id) == ["ornith-1.5-35b-a3b_4-Bit"])
        #expect(catalog.skipped.first?.reason.hasPrefix("duplicate id") == true)
    }

    @Test func theModelFlagFindsAnEntryByIDOrPath() throws {
        let root = try CatalogFixture.root()
        defer { try? FileManager.default.removeItem(at: root) }
        try CatalogFixture.ornith(root, "ornith-1.5_35B_A3B_4Bit")
        let catalog = ModelCatalog.scan(directory: root)
        #expect(
            catalog.entry(idOrPath: "ornith-1.5-35b-a3b_4-Bit")?.id == "ornith-1.5-35b-a3b_4-Bit")
        let path = root.appendingPathComponent("ornith-1.5_35B_A3B_4Bit").path + "/"
        #expect(catalog.entry(idOrPath: path)?.id == "ornith-1.5-35b-a3b_4-Bit")
        #expect(catalog.entry(idOrPath: "nothing") == nil)
    }

    /// An engine suffix is not a model id any more: it existed because a dense
    /// install had two engines, and the CPU engine left with TT-142 slice (b).
    @Test func theModelFlagDoesNotResolveAnEngineSuffix() {
        let ornith = ModelCatalog.Entry(
            id: "ornith-1.5-35b-a3b_4-Bit", name: "Ornith 1.5 35B-A3B",
            kind: .gpu(family: "qwen36", reasoning: .binaryThinking), quant: 4,
            path: URL(fileURLWithPath: "/models/ornith"), sampling: .house)
        let catalog = ModelCatalog(
            directory: URL(fileURLWithPath: "/models"), entries: [ornith])
        #expect(catalog.entry(idOrPath: "ornith-1.5-35b-a3b_4-Bit@gpu") == nil)
        #expect(catalog.entry(idOrPath: "ornith-1.5-35b-a3b_4-Bit@cpu") == nil)
    }

    /// The exact document a launcher script parses, field for field.
    @Test func theCatalogJSONIsTheLaunchersShape() throws {
        let catalog = ModelCatalog(
            directory: URL(fileURLWithPath: "/abs"),
            entries: [
                ModelCatalog.Entry(
                    id: "qwen3.6-35b-a3b_8-Bit", name: "Qwen 3.6 35B-A3B",
                    kind: .gpu(family: "qwen36", reasoning: .binaryThinking), quant: 8,
                    path: URL(fileURLWithPath: "/abs/path"),
                    sampling: .house, sizeBytes: 36_200_000_000)
            ])
        let text = try catalog.jsonData().lossyUTF8String
        #expect(
            text
                == #"{"models":[{"id":"qwen3.6-35b-a3b_8-Bit","name":"Qwen 3.6 35B-A3B","family":"qwen36","quant":8,"backend":"gpu","engines":"gpu","path":"/abs/path","thinking":["off","on"],"sampling":{"temperature":0.6,"top_p":0.95,"top_k":20},"size_gb":36.2}]}"#
        )
    }

    @Test func anEffortModelReportsWhatItWillUse() throws {
        let catalog = ModelCatalog(
            directory: URL(fileURLWithPath: "/abs"),
            entries: [
                ModelCatalog.Entry(
                    id: "qwen3.6-35b-a3b_4-Bit", name: "Qwen 3.6 35B-A3B",
                    kind: .gpu(family: "qwen38flash", reasoning: .binaryThinking), quant: 4,
                    path: URL(fileURLWithPath: "/c"),
                    sampling: SamplingProfile.house),
                ModelCatalog.Entry(
                    id: "qwen3.8-flash-next_4-Bit", name: "Qwen 3.8 Flash Next 125B-A6B",
                    kind: .gpu(
                        family: "qwen38flash",
                        reasoning: .thinkingWithEffortLevels(
                            defaultEffort: .xhigh, supported: [.low, .medium, .xhigh])), quant: 4,
                    path: URL(fileURLWithPath: "/f"),
                    sampling: SamplingProfile(
                        temperature: 1.0, topK: 20, topP: 0.95, presencePenalty: 0)),
            ])
        let document = try #require(
            JSONSerialization.jsonObject(with: try catalog.jsonData()) as? [String: Any])
        let models = try #require(document["models"] as? [[String: Any]])
        #expect(models[0]["thinking"] as? [String] == ["off", "on"])
        #expect(models[1]["thinking"] as? [String] == ["off", "low", "medium", "xhigh"])
        #expect((models[1]["sampling"] as? [String: Any])?["temperature"] as? Double == 1.0)
    }
}
