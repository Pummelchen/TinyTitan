import Testing
import TinyTitanFormat

@testable import Ornith35B4Bit

/// The rANS descriptor geometry is derived, not literal (AUD-0081).
///
/// `RANSLayerResources` sizes its descriptor buffer and
/// `PreadExpertStreamer` strides its writes through it, from
/// `ExpertRANSIndex.maximumStreamsPerBlock`. A literal 256 here would silently
/// under-size the region the moment the index's clamp rose.
@Suite("rANS descriptor capacity")
struct RANSCapacityTests {
    @Test func theStrideCoversEveryStreamOfAWave() {
        let wave = 8
        let models = 3
        let expected =
            wave * models * ExpertRANSIndex.maximumStreamsPerBlock
            * RANSStreamPlan.descriptorWords * 4
        #expect(RANSStreamPlan.descriptorStride(models: models) * wave == expected)
    }

    @Test func theCapacityIsTheIndexClamp() {
        #expect(
            ExpertRANSIndex.maximumStreamsPerBlock
                == ExpertRANSIndex.streams(symbolCount: 2048 * 1024))
    }
}
