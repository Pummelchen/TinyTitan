import Foundation
import Testing

@testable import TinyTitan

/// The seek index a compressed expert layer cannot do without: an uncompressed
/// layer is `n * stride`, a compressed one is wherever its blob landed.
///
/// Every case here is a failure the streamer would otherwise turn into silent
/// weight corruption -- the wrong expert's bytes decoded into a slot -- rather
/// than a read error.
@Suite("Expert blob index")
struct ExpertBlobIndexTests {
    private static func sample(_ count: Int) -> [ExpertBlobIndex.Entry] {
        var entries: [ExpertBlobIndex.Entry] = []
        var offset: UInt64 = 0
        for expert in 0..<count {
            let length = UInt32(1_000 + expert)
            entries.append(.init(offset: offset, length: length))
            offset += UInt64(length)
        }
        return entries
    }

    @Test("An index round-trips")
    func roundTrips() throws {
        let entries = Self.sample(8)
        let decoded = try ExpertBlobIndex.decode(
            ExpertBlobIndex.encode(entries), expectedCount: 8)
        #expect(decoded == entries)
    }

    @Test("A truncated index is refused, not read past its end")
    func refusesTruncation() {
        let data = ExpertBlobIndex.encode(Self.sample(8))
        #expect(throws: ExpertBlobIndex.Failure.self) {
            _ = try ExpertBlobIndex.decode(data.prefix(data.count - 4), expectedCount: 8)
        }
    }

    @Test("An index written for another layer is refused")
    func refusesWrongExpertCount() {
        let data = ExpertBlobIndex.encode(Self.sample(8))
        #expect(throws: ExpertBlobIndex.Failure.self) {
            _ = try ExpertBlobIndex.decode(data, expectedCount: 9)
        }
    }

    @Test("A file that is not an index is refused at the magic")
    func refusesForeignFile() {
        var data = ExpertBlobIndex.encode(Self.sample(2))
        data[0] = UInt8(ascii: "X")
        #expect(throws: ExpertBlobIndex.Failure.self) {
            _ = try ExpertBlobIndex.decode(data, expectedCount: 2)
        }
    }

    /// Overlapping blobs would decode one expert's weights into another expert's
    /// slot; the index cannot say so at read time, so it says so at decode.
    @Test("Overlapping entries are refused")
    func refusesOverlap() {
        let entries = [
            ExpertBlobIndex.Entry(offset: 0, length: 100),
            ExpertBlobIndex.Entry(offset: 50, length: 100),
        ]
        #expect(throws: ExpertBlobIndex.Failure.self) {
            _ = try ExpertBlobIndex.decode(
                ExpertBlobIndex.encode(entries), expectedCount: 2)
        }
    }

    @Test("A zero-length entry is refused")
    func refusesEmptyEntry() {
        let entries = [ExpertBlobIndex.Entry(offset: 0, length: 0)]
        #expect(throws: ExpertBlobIndex.Failure.self) {
            _ = try ExpertBlobIndex.decode(
                ExpertBlobIndex.encode(entries), expectedCount: 1)
        }
    }
}
