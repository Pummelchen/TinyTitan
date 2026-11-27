import Foundation
import Testing
import TinyTitanCore
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
            let arch: [String: Any] = [
                "hiddenSize": 64, "ffnIntermediate": 128, "moeIntermediateSize": 128,
                "numHeads": 4, "numKVHeads": 2, "numFullKVHeads": 2, "headDim": 32,
                "fullHeadDim": 32, "vocabSize": 1024, "slidingWindow": 1024,
                "finalLogitSoftcap": 0.0, "ropeTheta": 10_000_000.0,
                "fullRopeTheta": 10_000_000.0, "partialRotaryFactor": 0.25,
                "numLayers": 4, "numExperts": 8, "topKExperts": 8,
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
                "expertsPerLayer": 8, "numLayers": 4,
                "expertStride": 16384,
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

    /// A manifest carries an id, not a card name. The identity a front end
    /// reads names the install from the plugin, not from the id: this is the
    /// Ornith install, the only shipped family whose probe accepts a synthetic
    /// manifest. The two Inkling installs need their real payload layout, so
    /// their identity is covered by their own plugin suites.
    @Test func theOrnithInstallKeepsItsDisplayName() throws {
        let (_, identity) = try identity(
            family: "qwen36", modelID: "ornith-1.5-35b-a3b", bits: 4)
        #expect(identity.displayName == "Ornith 1.5 35B-A3B")
    }
}
