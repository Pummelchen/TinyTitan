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
        #expect(identity.engines == [.gpu])
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
        #expect(identity.engines == [.gpu, .cpu])
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
        let (_, mtp) = try identity(
            family: "qwen36_mtp", modelID: "qwen3.6-35b-a3b-mtp-4bit", bits: 4)
        #expect(mtp.modelID == "qwen3.6-35b-a3b-mtp-4bit")
        #expect(mtp.catalogID == "qwen3.6-35b-a3b-mtp_4-Bit")
        #expect(mtp.unsupportedReason?.contains("MTP draft head") == true)

        let (_, laguna) = try identity(
            family: "laguna", modelID: "laguna-s-2.1-4bit", bits: 4)
        #expect(laguna.unsupportedReason?.contains("not served") == true)
    }

    /// A dense install whose manifest carries the placeholder id is named by
    /// its family, exactly as the old adapter table named it.
    @Test func aDenseInstallWithoutAnIDUsesTheFamilyName() throws {
        let (_, identity) = try identity(
            family: "qwen3_5_dense", modelID: "unknown/snapshot", bits: 4)
        #expect(identity.catalogID == "qwen3.5-dense_4-Bit")
        #expect(identity.displayName == "qwen3.5-dense")
    }

    /// A manifest carries an id, not a card name. The shipped set's names are
    /// asserted here, through the identity the front ends read, so adding a
    /// model without a name is a deliberate edit rather than a silent fallback
    /// to its id.
    @Test func everyShippedInstallKeepsItsDisplayName() throws {
        let shipped: [(family: String, modelID: String, bits: Int, name: String)] = [
            ("qwen36", "qwen3.6-35b-a3b", 8, "Qwen 3.6 35B-A3B"),
            ("qwen36", "ornith-1.5-35b-a3b", 4, "Ornith 1.5 35B-A3B"),
            (
                "qwen36", "ornith-1.5-35b-a3b-bf16dense", 4,
                "Ornith 1.5 35B-A3B (bf16 dense)"
            ),
            ("qwen36", "qwen-agentworld", 4, "Qwen AgentWorld 35B-A3B"),
            ("qwen36", "kat-coder-v2.5", 4, "KAT-Coder-V2.5-Dev 35B-A3B"),
            ("qwen38flash", "qwen3.8-flash-next", 4, "Qwen 3.8 Flash Next 125B-A6B"),
            (
                "qwen4flash", "qwen4-omni-120b-think", 4,
                "Qwen 4.0 Flash (Qwen4-Omni-120B-Think)"
            ),
            ("qwen3_5_dense", "qwen3.5-2b", 4, "Qwen 3.5 2B"),
            ("qwen3_5_dense", "qwen3.5-4b", 4, "Qwen 3.5 4B"),
            ("qwen3_5_dense", "qwen3.5-9b", 4, "Qwen 3.5 9B"),
        ]
        var names: Set<String> = []
        for install in shipped {
            let (_, identity) = try identity(
                family: install.family, modelID: install.modelID, bits: install.bits)
            #expect(identity.displayName == install.name, "\(install.modelID)")
            names.insert(identity.displayName)
        }
        #expect(names.count == shipped.count, "two shipped models share a display name")
    }
}
