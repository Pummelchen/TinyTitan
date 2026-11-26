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
}
