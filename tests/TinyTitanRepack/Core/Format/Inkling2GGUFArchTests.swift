import Testing
import TinyTitanFormat

@testable import TinyTitanRepackCore

/// The Inkling 2 GGUF architecture reader: every `inkling2.*` integer comes
/// from untrusted metadata and must become an `Int` without trapping.
@Suite("Inkling 2 GGUF architecture")
struct Inkling2GGUFArchTests {
    @Test("Non-finite and out-of-range integer metadata is a typed error")
    func nonFiniteIntegerMetadataIsRefused() throws {
        for value in [Double.nan, .infinity, -.infinity, 1e300] {
            var fixture = Inkling2GGUFFixture.valid()
            fixture.set("inkling2.rope.dimension_count", .f64(value))
            let header = try GGUFHeaderV3(bytes: fixture.bytes())
            #expect(throws: RepackError.self, "\(value)") {
                _ = try Inkling2GGUFArch.archInfo(header)
            }
        }
    }

    @Test("The integer conversion is exact and total")
    func integerConversionIsChecked() throws {
        let exact = 9_007_199_254_740_992  // 2^53, the largest exact Double unit
        #expect(try Inkling2GGUFArch.integer(Double(exact), field: "x") == exact)
        #expect(try Inkling2GGUFArch.integer(Double(-exact), field: "x") == -exact)
        #expect(try Inkling2GGUFArch.integer(0, field: "x") == 0)
        for value in [24.5, Double(Int.max), Double.nan, .infinity, -.infinity, 1e300] {
            #expect(throws: RepackError.self, "\(value)") {
                _ = try Inkling2GGUFArch.integer(value, field: "x")
            }
        }
    }

    @Test("A complete fixture reads as the architecture it describes")
    func aValidFixtureReadsTheArchitecture() throws {
        let header = try GGUFHeaderV3(bytes: Inkling2GGUFFixture.valid().bytes())
        let arch = try Inkling2GGUFArch.archInfo(header)
        #expect(arch.family == .inkling2)
        #expect(arch.numLayers == 2)
        #expect(arch.numExperts == 2)
        #expect(arch.topKExperts == 2)
        #expect(arch.headDim == 8)
        #expect(arch.attentionScale.isFinite)
        #expect(arch.attentionScale == 1 / 8.0.squareRoot())
    }

    /// `attention.head_count` may be a metadata list; the first entry is the
    /// value, and it goes through the same exact conversion.
    @Test("A head-count list uses its first value, checked")
    func aHeadCountListUsesItsFirstValue() throws {
        var fixture = Inkling2GGUFFixture.valid()
        fixture.set("inkling2.attention.head_count", .f64Array([3, 1]))
        let header = try GGUFHeaderV3(bytes: fixture.bytes())
        #expect(try Inkling2GGUFArch.archInfo(header).numHeads == 3)

        fixture.set("inkling2.attention.head_count", .f64Array([.nan]))
        let badHeader = try GGUFHeaderV3(bytes: fixture.bytes())
        #expect(throws: RepackError.self) {
            _ = try Inkling2GGUFArch.archInfo(badHeader)
        }
    }

    /// `(passes - 1) * (loopEnd - loopStart)` came from metadata and was
    /// unchecked; 2^62 passes over a four-layer core overflows `Int`.
    @Test("An overflowing loop product is refused")
    func overflowingLoopProductIsRefused() throws {
        var fixture = Inkling2GGUFFixture.valid()
        fixture.set("inkling2.num_loops", .f64(4_611_686_018_427_387_904))  // 2^62
        fixture.set("inkling2.loop.core_end", .u32(4))
        fixture.set("inkling2.block_count", .u32(4))
        let header = try GGUFHeaderV3(bytes: fixture.bytes())
        #expect(throws: RepackError.self) {
            _ = try Inkling2GGUFArch.archInfo(header)
        }
    }

    /// `Parameters.unrolledLayers` repeated the product when the shared
    /// `Inkling2Arch.archInfo` built the manifest scalars.
    @Test("An overflowing unrolled-layer count is refused")
    func overflowingUnrolledLayersIsRefused() {
        let parameters = Self.parameters(stored: Int.max, loopEnd: 1)
        #expect(throws: RepackError.self) {
            _ = try Inkling2Arch.archInfo(parameters)
        }
    }

    @Test("The loop arithmetic is checked at every step")
    func loopArithmeticIsChecked() throws {
        #expect(try Inkling2Arch.extraLoopLayers(loopStart: 0, loopEnd: 2, passes: 2) == 2)
        #expect(
            try Inkling2Arch.storedLayers(
                unrolled: 4, loopStart: 0, loopEnd: 2, passes: 2) == 2)
        #expect(throws: RepackError.self) {
            _ = try Inkling2Arch.extraLoopLayers(
                loopStart: 0, loopEnd: 4, passes: 4_611_686_018_427_387_904)
        }
        #expect(throws: RepackError.self) {
            _ = try Inkling2Arch.storedLayers(
                unrolled: Int.min, loopStart: 0, loopEnd: 1, passes: 2)
        }
        #expect(throws: RepackError.self) {
            _ = try Inkling2Arch.checkedAdd(Int.max, 1, field: "x")
        }
        #expect(throws: RepackError.self) {
            _ = try Inkling2Arch.checkedSubtract(Int.min, 1, field: "x")
        }
        #expect(throws: RepackError.self) {
            _ = try Inkling2Arch.checkedMultiply(Int.max, 2, field: "x")
        }
    }

    @Test("Sliding-window visits are mapped with checked arithmetic")
    func slidingWindowVisitsAreMapped() throws {
        let mapped = try Inkling2GGUFArch.slidingLayers(
            pattern: [0, 1, 0, 1, 1, 0], stored: 4, loopStart: 0, loopEnd: 2,
            passes: 2)
        #expect(mapped == [1, 2])
        #expect(throws: RepackError.self) {
            _ = try Inkling2GGUFArch.slidingLayers(
                pattern: [0, 0], stored: 2, loopStart: 0, loopEnd: 1,
                passes: Int.max)
        }
    }

    @Test("A valid parameter set still reads the same manifest values")
    func validParametersAreUnchanged() throws {
        let arch = try Inkling2Arch.archInfo(Self.parameters())
        #expect(arch.numLayers == 2)
        #expect(arch.variantGeometry?.scalars["unrolledLayers"] == 4)
    }

    private static func parameters(
        stored: Int = 2, loopStart: Int = 0, loopEnd: Int = 2, passes: Int = 2,
        headDim: Int = 8
    ) -> Inkling2Arch.Parameters {
        Inkling2Arch.Parameters(
            hiddenSize: 64, intermediateSize: 32, expertIntermediateSize: 32,
            heads: 2, kvHeads: 2, headDim: headDim,
            paddedVocabulary: 128, vocabulary: 128, slidingWindow: 16,
            finalLogitSoftcap: 30, storedLayers: stored, experts: 2, expertsPerToken: 2,
            localLayers: [1], tiedEmbeddings: true,
            loopStart: loopStart, loopEnd: loopEnd, loopPasses: passes,
            loopGateRank: 4, attentionResidualBlock: 2, relativeLength: 8,
            relativeExtent: 16, relativeClip: 8, convolutionTaps: 4,
            rmsEpsilon: 1e-6, qkClip: 20, logScaleFloor: 8_192,
            logScaleAlpha: 0.1, expertWeightsScale: 8, situBeta: 4,
            situLinearBeta: 25, contextLength: 1_024, quantGroupSize: 32)
    }
}
