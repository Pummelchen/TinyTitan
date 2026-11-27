import Foundation
import Testing
import TinyTitanCore
import TinyTitanVariants
import Ornith35B4Bit

@testable import TinyTitanServerCore

/// TT-097 Phase 3 Task 10: the server's factory selection. Model-free by
/// construction — the selection is decided from the manifest identity a
/// catalog entry carries, and the factories are compared as selection
/// outcomes, never by running a model. The actual routing flips are Task 11+,
/// each gated on a real install.
@Suite struct VariantFactorySelectionTests {

    @Test("the shipped ornith 4-bit variant selects its dedicated factory")
    func ornith4BitSelectsTheVariant() {
        #expect(
            EngineFactories.gpuSelection(
                family: "qwen36",
                manifestModelID: "ornith-1.5-35b-a3b",
                routedExpertWeightBits: 4) == .ornith35B4Bit)
    }

    @Test("every other GPU identity has no shipped variant")
    func everythingElseHasNoVariant() {
        // The 8-bit ornith variant is not shipped.
        #expect(
            EngineFactories.gpuSelection(
                family: "qwen36", manifestModelID: "ornith-1.5-35b-a3b",
                routedExpertWeightBits: 8) == nil)
        // A different qwen36 install is not the ornith variant.
        #expect(
            EngineFactories.gpuSelection(
                family: "qwen36", manifestModelID: "some-private-checkpoint",
                routedExpertWeightBits: 4) == nil)
        // Other families have no shipped GPU variant.
        #expect(
            EngineFactories.gpuSelection(
                family: "qwen38flash", manifestModelID: "qwen3.8-flash-next",
                routedExpertWeightBits: 4) == nil)
        #expect(
            EngineFactories.gpuSelection(
                family: "laguna", manifestModelID: "laguna-s-2-1",
                routedExpertWeightBits: 4) == nil)
    }

    @Test("the ornith plugin factory composes its own planner stages")
    func variantFactoryIsComplete() throws {
        let variant = try EngineFactories.gpuFactory(
            family: "qwen36", manifestModelID: "ornith-1.5-35b-a3b-4bit",
            routedExpertWeightBits: 4)
        // The planner and architecture stages are the family-generic
        // production ones; the engine stages are the variant's own. A default
        // EngineFactory refuses with a sentinel ("no load planner" / "no
        // architecture resolver"), so probing a bogus directory distinguishes
        // composed stages from defaults without running a model: a composed
        // stage gets far enough to fail on the read itself.
        let sentinels = ["no load planner", "no architecture resolver"]
        func isSentinel(_ error: Error) -> Bool {
            sentinels.contains { "\(error).".contains($0) }
        }
        do {
            _ = try variant.resolveArchitecture(URL(fileURLWithPath: "/nonexistent"))
            Issue.record("a bogus directory must fail, not resolve")
        } catch {
            #expect(!isSentinel(error), "the variant factory carries the default resolver")
        }
        do {
            _ = try variant.prepareLoad(
                SessionBuildOptions(
                    modelDirectory: URL(fileURLWithPath: "/nonexistent"),
                    context: nil, maxContext: 64, slots: 1,
                    promptCacheMaximumEntries: 0, promptCacheMemoryLimitBytes: 0,
                    promptCacheDiskDirectory: nil, promptCacheDiskLimitBytes: 0,
                    prefillChunkTokens: nil, kvCachePrecision: .int8,
                    hostKVWindowTokens: nil, ropeScalingMode: .none,
                    thinkingMode: .off, reasoningEffort: nil,
                    expertCacheSlots: nil, expertCacheBudgetBytes: nil,
                    mtpModelDirectory: nil, mtpMemoryMiB: 0,
                    integrityPolicy: nil, rdAdvicePolicy: nil, forceLogitsHead: true))
            Issue.record("a bogus directory must fail, not plan")
        } catch {
            #expect(!isSentinel(error), "the variant factory carries the default planner")
        }
    }

    @Test("the router dispatch hands ornith its variant factory, others a refusal")
    func routerDispatchSelectsPerIdentity() throws {
        // With no Metal context the engine stage is where the two factories
        // differ, and the error names the engine: a model-free probe of
        // which factory the dispatch returned.
        func engineStageError(_ factory: EngineFactory) -> String {
            do {
                _ = try factory.buildModel(
                    URL(fileURLWithPath: "/nonexistent"),
                    ArchConfig.qwen36_35B_A3B,
                    nil)
                return "no error"
            } catch {
                return "\(error)"
            }
        }

        let ornith = try EngineFactories.gpuFactory(
            family: "qwen36", manifestModelID: "ornith-1.5-35b-a3b",
            routedExpertWeightBits: 4)
        #expect(
            engineStageError(ornith).contains("Ornith engine"),
            "the shipped ornith identity must dispatch to the variant factory")

        // Every identity no shipped variant claims is refused by name: there
        // is no fallback engine (TT-142).
        #expect(throws: EngineFactoryError.self) {
            _ = try EngineFactories.gpuFactory(
                family: "qwen36", manifestModelID: "some-private-checkpoint",
                routedExpertWeightBits: 4)
        }

        // A family that used to have a dense engine is refused by name now:
        // no install no plugin claims is handed to a fallback (TT-142 slice
        // (b) deleted the dense family, 2026-11-28).
        #expect(
            EngineFactories.gpuSelection(
                family: "qwen3_5_dense", manifestModelID: "qwen3.5-4b",
                routedExpertWeightBits: 4) == nil)
        #expect(throws: EngineFactoryError.self) {
            _ = try EngineFactories.gpuFactory(
                family: "qwen3_5_dense", manifestModelID: "qwen3.5-4b",
                routedExpertWeightBits: 4)
        }
    }

    @Test("the single-model directory dispatch reads the install identity")
    func directoryDispatchSelectsPerIdentity() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("variant-selection-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let arch: [String: Any] = [
            "hiddenSize": 64, "ffnIntermediate": 128, "moeIntermediateSize": 128,
            "numHeads": 4, "numKVHeads": 2, "numFullKVHeads": 2, "headDim": 32,
            "fullHeadDim": 32, "vocabSize": 1024, "slidingWindow": 1024,
            "finalLogitSoftcap": 0.0, "ropeTheta": 10_000_000.0,
            "fullRopeTheta": 10_000_000.0, "partialRotaryFactor": 0.25,
            "numLayers": 4, "numExperts": 8, "topKExperts": 8,
            "tieWordEmbeddings": false, "attentionKEqV": false,
            "hiddenActivation": "silu", "fullAttentionLayerMask": [2, 1, 2, 1],
            "family": "qwen36",
        ]
        let slot: [String: Any] = [
            "weightBits": 4, "scheme": "affine", "scaleType": "bf16",
            "biasType": "bf16", "groupSize": 64,
        ]
        try JSONSerialization.data(withJSONObject: [
            "magic": "SSDAI", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true], "modelID": "ornith-1.5-35b-a3b",
            "arch": arch, "files": [:], "expertsPerLayer": 8, "numLayers": 4,
            "expertStride": 16384,
            "quant": [
                "embedding": slot, "attention": slot, "router": slot,
                "sharedExpert": slot, "routedExpert": slot,
            ],
        ]).write(to: directory.appendingPathComponent("manifest.json"))
        let factory = try EngineFactories.gpuFactory(directoryURL: directory)
        do {
            _ = try factory.buildModel(directory, ArchConfig.qwen36_35B_A3B, nil)
            Issue.record("the variant requires a Metal context")
        } catch {
            #expect("\(error)".contains("Ornith engine"))
        }
        // The CLI and an embedder call `routingFactory`, which must take the
        // same plugin resolution: a directory naming the Ornith install gets
        // the variant, not some other engine.
        let routed = try EngineFactories.routingFactory(directoryURL: directory)
        do {
            _ = try routed.buildModel(directory, ArchConfig.qwen36_35B_A3B, nil)
            Issue.record("the routed factory must be the variant too")
        } catch {
            #expect("\(error)".contains("Ornith engine"))
        }
    }

}
