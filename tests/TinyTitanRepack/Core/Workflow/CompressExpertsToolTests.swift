import Testing

@testable import TinyTitanRepackCore

/// The accounting `--compress-experts` reports: the raw plan total and the
/// per-layer sums. The inputs are plan declarations, so an overflow is refused
/// rather than wrapping or trapping; the boundary is tested directly because a
/// full `run` would need a layer file of the same impossible size.
@Suite("Compress experts accounting")
struct CompressExpertsToolTests {
    @Test("The raw plan total is checked")
    func rawTotalIsChecked() throws {
        #expect(
            try CompressExpertsTool.totalRawBytes(
                rawLayerBytes: 1 << 40, layerCount: 4) == 1 << 42)
        #expect(
            try CompressExpertsTool.totalRawBytes(
                rawLayerBytes: UInt64.max / 2, layerCount: 2) == UInt64.max - 1)
        #expect(throws: RepackError.self) {
            _ = try CompressExpertsTool.totalRawBytes(
                rawLayerBytes: UInt64.max / 2 + 1, layerCount: 2)
        }
        #expect(throws: RepackError.self) {
            _ = try CompressExpertsTool.totalRawBytes(
                rawLayerBytes: 1, layerCount: -1)
        }
    }

    @Test("The summary sums are checked")
    func summarySumsAreChecked() throws {
        #expect(try CompressExpertsTool.totalBytes([1, 2, 3], field: "bytes") == 6)
        #expect(try CompressExpertsTool.totalBytes([], field: "bytes") == 0)
        #expect(throws: RepackError.self) {
            _ = try CompressExpertsTool.totalBytes([UInt64.max, 1], field: "bytes")
        }
    }
}
