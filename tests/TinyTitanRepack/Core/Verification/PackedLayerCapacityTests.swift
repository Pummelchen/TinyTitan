import Foundation
import TinyTitanFormat
import Testing

@testable import TinyTitanRepackCore

/// `--verify-install` must refuse a malformed layout instead of trapping, and
/// it must apply the loader's own structural predicate (AUD-0073/0074).
///
/// The trap was `UInt64(layer.expertsPerLayer ?? …)` on an `Int?` decoded from
/// an untrusted document: `-1` is a fatal error, which reads to an operator as a
/// broken tool rather than a broken install. The weak-predicate half was the
/// verifier's local mirror, which did not check duplicate layer files,
/// overlapping experts or the expert-count ceiling the runtime's validator
/// enforces.
@Suite struct PackedLayerCapacityTests {
    @Test func aNegativeCountIsRefusedNotTrapped() {
        #expect(throws: RepackError.self) {
            _ = try VerifiedInstallTool.packedLayerCapacity(count: -1, stride: 64)
        }
    }

    @Test func aCountThatWouldOverflowIsRefused() {
        #expect(throws: RepackError.self) {
            _ = try VerifiedInstallTool.packedLayerCapacity(count: Int.max, stride: 64)
        }
    }

    @Test func anOrdinaryLayerIsItsCountTimesStride() throws {
        #expect(try VerifiedInstallTool.packedLayerCapacity(count: 8, stride: 64) == 512)
        #expect(try VerifiedInstallTool.packedLayerCapacity(count: 0, stride: 64) == 0)
    }
}

/// The verifier decodes `layout.json` through the format layer, so the rules
/// that decide whether the runtime can load it are the rules that decide
/// whether the tool may call it verified.
@Suite struct VerifierUsesTheLoaderPredicateTests {
    /// Two layers naming one file is valid JSON, passes the verifier's mirror
    /// arithmetic, and is refused by the loader. It must be refused here too.
    @Test func aDuplicateLayerFileIsRefused() throws {
        let layout = """
            {"expertStride":64,"numLayers":2,"expertsPerLayer":1,
             "layers":[
              {"layer":0,"file":"layer_00.bin","experts":[{"expert":0,"offset":0,"size":64,
                "slices":[{"fileOffset":0,"sizeBytes":64,"dtype":0,"shape":[1]}]}]},
              {"layer":1,"file":"layer_00.bin","experts":[{"expert":0,"offset":0,"size":64,
                "slices":[{"fileOffset":0,"sizeBytes":64,"dtype":0,"shape":[1]}]}]}
             ]}
            """
        #expect(throws: (any Error).self) {
            _ = try SSDAIPackedExpertsLayoutCodec.decode(Data(layout.utf8))
        }
    }
}
