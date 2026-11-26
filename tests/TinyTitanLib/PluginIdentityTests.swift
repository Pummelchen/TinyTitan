import Foundation
import Testing
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanPluginAPI
import TinyTitanVariants

/// TT-134 group (f): the host-facing identity a resolved plugin declares for an
/// install. Model-free: every install here is a synthetic manifest and no
/// weight file is created or read.
@Suite("Plugin install identity")
struct PluginIdentityTests {
    private enum Fixture {
        static func write(
            _ family: String, _ modelID: String, bits: Int
        ) throws -> URL {
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("plugin-identity-\(UUID().uuidString)")
            let install = root.appendingPathComponent("install")
            try FileManager.default.createDirectory(
                at: install, withIntermediateDirectories: true)
            func slot(_ bits: Int) -> [String: Any] {
                [
                    "weightBits": bits, "scheme": "affine", "scaleType": "bf16",
                    "biasType": "bf16", "groupSize": 64,
                ]
            }
            let dense = family == "qwen3_5_dense"
            let arch: [String: Any] = [
                "hiddenSize": 64, "ffnIntermediate": 128, "moeIntermediateSize": 128,
                "numHeads": 4, "numKVHeads": 2, "numFullKVHeads": 2, "headDim": 32,
                "fullHeadDim": 32, "vocabSize": 1024, "slidingWindow": 1024,
                "finalLogitSoftcap": 0.0, "ropeTheta": 10_000_000.0,
                "fullRopeTheta": 10_000_000.0, "partialRotaryFactor": 0.25,
                "numLayers": 4, "numExperts": dense ? 0 : 8, "topKExperts": dense ? 0 : 8,
                "tieWordEmbeddings": false, "attentionKEqV": false,
                "hiddenActivation": "silu",
                "fullAttentionLayerMask": [2, 1, 2, 1], "family": family,
            ]
            let manifest: [String: Any] = [
                "magic": "SSDAI", "versionMajor": 1, "versionMinor": 0,
                "flags": [String: Bool](), "modelID": modelID, "arch": arch,
                "files": [
                    "model_weights.bin": [
                        "size": 1024, "sha256": String(repeating: "0", count: 64),
                    ]
                ],
                "expertsPerLayer": dense ? 0 : 8, "numLayers": 4,
                "expertStride": dense ? 0 : 16384,
                "quant": [
                    "embedding": slot(8), "attention": slot(8), "router": slot(8),
                    "sharedExpert": slot(bits), "routedExpert": slot(bits),
                ],
            ]
            try JSONSerialization.data(
                withJSONObject: manifest, options: [.sortedKeys]
            ).write(to: install.appendingPathComponent("manifest.json"))
            return install
        }
    }

    private func identity(
        family: String, modelID: String, bits: Int
    ) throws -> (ResolvedPlugin, ModelIdentity) {
        let install = try Fixture.write(family, modelID, bits: bits)
        defer { try? FileManager.default.removeItem(at: install.deletingLastPathComponent()) }
        let resolved = try EngineFactories.pluginRegistry.resolve(install)
        let identity = try #require(try resolved.modelIdentity(), "no identity declared")
        return (resolved, identity)
    }

    @Test func aStandalonePluginNamesItsOwnInstall() throws {
        let (resolved, identity) = try identity(
            family: "qwen36", modelID: "ornith-1.5-35b-a3b", bits: 4)
        #expect(resolved.pluginID == "ai.tinytitan.ornith-1.5-35b-a3b-4bit")
        #expect(identity.modelID == "ornith-1.5-35b-a3b")
        #expect(identity.catalogID == "ornith-1.5-35b-a3b_4-Bit")
        #expect(identity.displayName == "Ornith 1.5 35B-A3B")
        #expect(identity.family == "qwen36")
        #expect(identity.weightBits == 4)
        #expect(identity.sampling == .house)
        #expect(identity.reasoningControl == .binaryThinking)
        #expect(identity.unsupportedReason == nil)
    }

    @Test func theDensePluginNamesTheInstallBySize() throws {
        let (resolved, identity) = try identity(
            family: "qwen3_5_dense", modelID: "qwen3.5-2b", bits: 4)
        #expect(resolved.pluginID == "ai.tinytitan.qwen3.5-dense")
        #expect(identity.modelID == "qwen3.5-2b")
        #expect(identity.catalogID == "qwen3.5-2b_4-Bit")
        #expect(identity.displayName == "Qwen 3.5 2B")
        #expect(identity.family == "qwen3_5_dense")
        #expect(identity.weightBits == 4)
        #expect(identity.reasoningControl == .binaryThinking)
        #expect(identity.unsupportedReason == nil)
    }

    /// The 6.x adapter owns what it still serves: it reads the manifest itself
    /// and reports the same id, name and sampling the catalog published before
    /// the identity moved onto it.
    @Test func theAdapterNamesTheFamiliesItStillServes() throws {
        let (resolved, identity) = try identity(
            family: "qwen36", modelID: "qwen3.6-35b-a3b", bits: 8)
        #expect(resolved.pluginID == "ai.tinytitan.adapter.legacy-6x")
        #expect(identity.modelID == "qwen3.6-35b-a3b")
        #expect(identity.catalogID == "qwen3.6-35b-a3b_8-Bit")
        #expect(identity.displayName == "Qwen 3.6 35B-A3B")
        #expect(identity.family == "qwen36")
        #expect(identity.weightBits == 8)
        #expect(identity.unsupportedReason == nil)
    }

    /// A family the adapter's engine cannot serve says so in its identity, so a
    /// front end does not need its own family table to refuse it.
    @Test func theAdapterReportsWhyAFamilyIsNotServed() throws {
        let (_, k2) = try identity(
            family: "k2_horizon", modelID: "k2-horizon-4bit", bits: 4)
        #expect(k2.modelID == "k2-horizon-4bit")
        #expect(k2.catalogID == "k2-horizon_4-Bit")
        #expect(k2.unsupportedReason?.contains("server's probe backend") == true)

        let (_, mtp) = try identity(
            family: "qwen36_mtp", modelID: "qwen3.6-35b-a3b-mtp-4bit", bits: 4)
        #expect(mtp.modelID == "qwen3.6-35b-a3b-mtp-4bit")
        #expect(mtp.catalogID == "qwen3.6-35b-a3b-mtp_4-Bit")
        #expect(mtp.unsupportedReason?.contains("MTP draft head") == true)

        let (_, laguna) = try identity(
            family: "laguna", modelID: "laguna-s-2.1-4bit", bits: 4)
        #expect(laguna.unsupportedReason?.contains("not served") == true)
    }
}
