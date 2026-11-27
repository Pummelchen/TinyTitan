import Testing

@testable import TinyTitanCore

/// A resident entry's byte range is manifest input, so the engines convert it
/// to machine-size offsets through one checked helper instead of an unchecked
/// `fileOffset + sizeBytes` and `Int(...)` (AUD-0084).
@Suite("resident entry byte range")
struct ResidentIndexEntryRangeTests {
    private func entry(offset: UInt64, size: UInt64) -> ResidentIndexEntry {
        ResidentIndexEntry(
            name: "blk.0.weight", dtype: 3, fileOffset: offset, sizeBytes: size,
            shape: (16, 1, 1, 1), scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0)
    }

    @Test func aRangeInsideTheFileIsConverted() throws {
        let range = try entry(offset: 4096, size: 64).checkedByteRange(within: 8192)
        #expect(range.offset == 4096)
        #expect(range.count == 64)
    }

    @Test func aRangeThatEndsPastTheFileIsRefused() {
        #expect(throws: ModelError.self) {
            _ = try entry(offset: 8192, size: 64).checkedByteRange(within: 8192)
        }
    }

    @Test func aRangeWhoseEndOverflowsIsRefused() {
        #expect(throws: ModelError.self) {
            _ = try entry(offset: UInt64(Int.max) - 1, size: 8)
                .checkedByteRange(within: Int.max)
        }
    }

    @Test func anOffsetThatDoesNotFitIntIsRefused() {
        // A file length of Int.max is not reachable, but the conversion is
        // what must not trap: an offset above Int.max is refused first.
        #expect(throws: ModelError.self) {
            _ = try entry(offset: UInt64(Int.max) + 1, size: 1)
                .checkedByteRange(within: Int.max - 1)
        }
    }
}
