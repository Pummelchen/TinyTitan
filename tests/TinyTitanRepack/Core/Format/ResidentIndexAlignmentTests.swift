import Foundation
import Testing
import TinyTitanFormat

/// The resident index's alignment and element-size contract (AUD-0083).
///
/// Consumers map the payload and `bindMemory` a tensor's bytes at `fileOffset`
/// to `Float`/`UInt16`, so an offset that is not aligned for its dtype is
/// undefined behaviour on a stricter machine, and a plain entry whose
/// `sizeBytes` is not whole elements truncates its last element silently.
@Suite("resident index alignment")
struct ResidentIndexAlignmentTests {
    private static let indexSize = Int(SSDAIFormatV1.alignmentBytes)

    /// A one-entry index with the payload following it, at a chosen offset.
    private static func index(
        dtype: UInt8, shape: [UInt32], size: UInt64, offset: UInt64? = nil
    ) -> (bytes: [UInt8], header: SSDAIResidentIndexHeaderV1) {
        let fileOffset = offset ?? UInt64(indexSize)
        let residentSize = max(0, Int(fileOffset) + Int(size) - indexSize)
        var bytes = [UInt8](repeating: 0, count: indexSize + residentSize)
        func put(_ value: UInt64, _ at: Int, _ width: Int) {
            for byte in 0..<width {
                bytes[at + byte] = UInt8((value >> (8 * UInt64(byte))) & 0xff)
            }
        }
        let name = Array("blk.0.weight".utf8)
        let entry = 24
        let nameAt = entry + 72
        put(UInt64(indexSize), 0, 8)
        put(UInt64(residentSize), 8, 8)
        put(1, 16, 8)
        put(UInt64(nameAt), entry, 4)
        put(UInt64(name.count), entry + 4, 2)
        bytes[entry + 6] = dtype
        put(fileOffset, entry + 8, 8)
        put(size, entry + 16, 8)
        for (axis, dimension) in shape.enumerated() {
            put(UInt64(dimension), entry + 24 + 4 * axis, 4)
        }
        bytes.replaceSubrange(nameAt..<nameAt + name.count, with: name)
        let header = SSDAIResidentIndexHeaderV1(
            indexSize: UInt64(indexSize), residentSize: UInt64(residentSize), entryCount: 1)
        return (bytes, header)
    }

    private static func decode(
        _ built: (bytes: [UInt8], header: SSDAIResidentIndexHeaderV1)
    ) throws -> [SSDAIResidentIndexEntryV1] {
        try built.bytes.withUnsafeBytes {
            try SSDAIResidentIndexCodec.decodeRegion($0, header: built.header)
        }
    }

    @Test("An F32 tensor at an offset that is not 4-byte aligned is refused")
    func refusesMisalignedF32() {
        #expect(throws: TinyTitanFormatError.self) {
            _ = try Self.decode(
                Self.index(
                    dtype: 3, shape: [16, 0, 0, 0], size: 64,
                    offset: UInt64(Self.indexSize) + 2))
        }
    }

    @Test("A packed u32 tensor at an offset that is not 4-byte aligned is refused")
    func refusesMisalignedU32() {
        #expect(throws: TinyTitanFormatError.self) {
            _ = try Self.decode(
                Self.index(
                    dtype: 0, shape: [64, 0, 0, 0], size: 32,
                    offset: UInt64(Self.indexSize) + 2))
        }
    }

    @Test("A BF16 tensor at an odd offset is refused, at an even one it decodes")
    func bf16NeedsTwoByteAlignment() throws {
        #expect(throws: TinyTitanFormatError.self) {
            _ = try Self.decode(
                Self.index(
                    dtype: 1, shape: [16, 0, 0, 0], size: 32,
                    offset: UInt64(Self.indexSize) + 1))
        }
        let built = Self.index(
            dtype: 1, shape: [16, 0, 0, 0], size: 32,
            offset: UInt64(Self.indexSize) + 2)
        #expect(try Self.decode(built).first?.sizeBytes == 32)
    }

    @Test("An F32 size of 4n+2 bytes is refused instead of truncating")
    func refusesPartialF32Element() {
        // 66 bytes at 4 bytes an element is 16 elements and a half: the old
        // reader accepted it and `bindMemory(to: Float.self)` read 16 values,
        // dropping the last two bytes.
        #expect(throws: TinyTitanFormatError.self) {
            _ = try Self.decode(
                Self.index(dtype: 3, shape: [16, 0, 0, 0], size: 66))
        }
    }

    @Test("A plain size that disagrees with its shape is refused")
    func refusesSizeThatDisagreesWithShape() {
        // 68 bytes is whole F32 elements but the shape says 16, i.e. 64 bytes.
        #expect(throws: TinyTitanFormatError.self) {
            _ = try Self.decode(
                Self.index(dtype: 3, shape: [16, 0, 0, 0], size: 68))
        }
        #expect(throws: TinyTitanFormatError.self) {
            _ = try Self.decode(
                Self.index(dtype: 1, shape: [16, 0, 0, 0], size: 34))
        }
    }

    @Test("Packed u32 weights are sized by their bytes, not the logical shape")
    func packedU32KeepsItsPackedSize() throws {
        // A 4-bit row of 64 values is 32 bytes: `sizeBytes` is not
        // `product(shape) * 4`, and dtype 0 is also the packed-weight code.
        let built = Self.index(dtype: 0, shape: [64, 0, 0, 0], size: 32)
        #expect(try Self.decode(built).first?.dtype == 0)
        // A size that is not a whole number of words is still refused.
        #expect(throws: TinyTitanFormatError.self) {
            _ = try Self.decode(Self.index(dtype: 0, shape: [15, 0, 0, 0], size: 30))
        }
    }

    @Test("A correctly sized F32 entry still decodes")
    func acceptsAlignedF32() throws {
        let built = Self.index(dtype: 3, shape: [16, 0, 0, 0], size: 64)
        let entry = try #require(try Self.decode(built).first)
        #expect(entry.dtype == 3)
        #expect(entry.sizeBytes == 64)
    }
}
