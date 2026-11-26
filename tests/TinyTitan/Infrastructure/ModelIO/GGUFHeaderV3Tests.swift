import Foundation
import Testing

@testable import TinyTitanFormat

/// Builds synthetic GGUF headers byte by byte, so a malformed document is a
/// few lines and never depends on a model file.
private struct GGUFBuilder {
    var bytes: [UInt8] = []

    mutating func u32(_ value: UInt32) {
        bytes += withUnsafeBytes(of: value.littleEndian, Array.init)
    }

    mutating func u64(_ value: UInt64) {
        bytes += withUnsafeBytes(of: value.littleEndian, Array.init)
    }

    mutating func string(_ value: String) {
        u64(UInt64(value.utf8.count))
        bytes += Array(value.utf8)
    }

    mutating func raw(_ value: [UInt8]) {
        bytes += value
    }

    static func header(
        version: UInt32 = 3, tensors: UInt64 = 0, keys: UInt64 = 0
    ) -> GGUFBuilder {
        var builder = GGUFBuilder()
        builder.raw(Array("GGUF".utf8))
        builder.u32(version)
        builder.u64(tensors)
        builder.u64(keys)
        return builder
    }

    /// `general.alignment` as the uint64 it may legally be read from.
    mutating func alignment(_ value: UInt64) {
        string("general.alignment")
        u32(10)
        u64(value)
    }

    mutating func tensor(name: String, rank: UInt32, dimensions: [UInt64], offset: UInt64) {
        string(name)
        u32(rank)
        for dimension in dimensions { u64(dimension) }
        u32(0)  // ggml F32; the header reader does not interpret it
        u64(offset)
    }
}

/// The installer's GGUF header reader (TT-121 phase 1).
@Suite struct GGUFHeaderV3Tests {
    /// A minimal v3 file: two metadata keys, one 2 x 64 F32 tensor at data offset 0.
    static func tinyFile() -> [UInt8] {
        var out: [UInt8] = []
        func u32(_ v: UInt32) { out += withUnsafeBytes(of: v.littleEndian, Array.init) }
        func u64(_ v: UInt64) { out += withUnsafeBytes(of: v.littleEndian, Array.init) }
        func str(_ s: String) {
            u64(UInt64(s.utf8.count))
            out += Array(s.utf8)
        }
        out += Array("GGUF".utf8)
        u32(3)
        u64(1)  // tensors
        u64(3)  // metadata keys
        str("general.architecture")
        u32(8)
        str("inkling2")
        str("inkling2.block_count")
        u32(4)
        u32(24)
        str("inkling2.layers")
        u32(9)
        u32(5)  // int32 array
        u64(3)
        for v: UInt32 in [0, 1, 2] { u32(v) }
        str("w")
        u32(2)
        u64(64)
        u64(2)
        u32(0)  // F32
        u64(0)
        while out.count % 32 != 0 { out.append(0) }
        out += [UInt8](repeating: 0, count: 2 * 64 * 4)
        return out
    }

    @Test func aSmallFileParses() throws {
        let bytes = Self.tinyFile()
        let header = try GGUFHeaderV3(bytes: bytes)
        #expect(header.version == 3)
        #expect(header.string("general.architecture") == "inkling2")
        #expect(header.number("inkling2.block_count") == 24)
        #expect(header.numbers("inkling2.layers") == [0, 1, 2])
        #expect(header.alignment == 32)
        #expect(header.dataStart % 32 == 0)
        #expect(
            header.tensors == [
                .init(
                    name: "w", dimensions: [64, 2], ggmlType: 0, absoluteOffset: header.dataStart)
            ])
    }

    @Test func aTruncatedHeaderAsksForMoreBytes() throws {
        let bytes = Array(Self.tinyFile().prefix(60))
        #expect {
            _ = try GGUFHeaderV3(bytes: bytes)
        } throws: { error in
            guard
                case GGUFHeaderV3.ReadError.needMoreBytes(let count)? = error
                    as? GGUFHeaderV3.ReadError
            else { return false }
            return count > 60
        }
    }

    @Test func aWrongMagicIsRefused() {
        var bytes = Self.tinyFile()
        bytes[0] = 0x58
        #expect(throws: GGUFHeaderV3.ReadError.self) { _ = try GGUFHeaderV3(bytes: bytes) }
    }

    private static var inklingGGUF: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_GGUF"]
    }

    /// The real Q4_K_XL header: 734 tensors, and the first expert tensor at the
    /// absolute offset llama.cpp's gguf-py reports for it.
    @Test(.enabled(if: inklingGGUF != nil)) func theInklingHeaderMatchesGGUFPy() throws {
        let path = try #require(Self.inklingGGUF)
        let handle = try #require(FileHandle(forReadingAtPath: path))
        defer { try? handle.close() }
        let prefix = [UInt8](try handle.read(upToCount: 64 << 20) ?? Data())
        let header = try GGUFHeaderV3(bytes: prefix)
        #expect(header.tensors.count == 734)
        #expect(header.string("general.architecture") == "inkling2")
        let gateUp = try #require(
            header.tensors.first { $0.name == "blk.0.ffn_gate_up_exps.weight" })
        #expect(gateUp.absoluteOffset == 1_217_729_504)
        #expect(gateUp.dimensions == [2_304, 2_304, 128])
        #expect(gateUp.ggmlType == 40)  // NVFP4
    }

    /// Every negative case asserts the typed failure and (where useful) the
    /// reason, not merely that something threw.
    private func expectInvalid(
        _ bytes: [UInt8], containing fragment: String? = nil,
        _ comment: Comment? = nil
    ) {
        do {
            let header = try GGUFHeaderV3(bytes: bytes)
            Issue.record(
                comment ?? "expected an invalid header, got dataStart \(header.dataStart)")
        } catch let error as GGUFHeaderV3.ReadError {
            guard case .invalid(let reason) = error else {
                Issue.record(comment ?? "expected .invalid, got \(error)")
                return
            }
            if let fragment {
                #expect(reason.contains(fragment), "\(reason)")
            }
        } catch {
            Issue.record(comment ?? "unexpected error \(error)")
        }
    }

    @Test func aVersionTwoHeaderParses() throws {
        let header = try GGUFHeaderV3(bytes: GGUFBuilder.header(version: 2).bytes)
        #expect(header.version == 2)
        #expect(header.metadata.isEmpty)
        #expect(header.tensors.isEmpty)
        #expect(header.alignment == 32)
        // The 24-byte header, padded to the default alignment.
        #expect(header.dataStart == 32)
    }

    /// A second metadata key would make `metadata[key]` silently choose one of
    /// two values. The parser refuses the document instead.
    @Test func duplicateMetadataKeysAreRefused() {
        var builder = GGUFBuilder.header(keys: 2)
        builder.string("a")
        builder.u32(0)
        builder.raw([1])
        builder.string("a")
        builder.u32(0)
        builder.raw([2])
        expectInvalid(builder.bytes, containing: "duplicate metadata key a")
    }

    /// The planner keys tensors by name, so a duplicate name would resolve to
    /// one of two offsets. Refused before the planner can see it.
    @Test func duplicateTensorNamesAreRefused() {
        var builder = GGUFBuilder.header(tensors: 2)
        builder.tensor(name: "w", rank: 1, dimensions: [4], offset: 0)
        builder.tensor(name: "w", rank: 1, dimensions: [8], offset: 0)
        expectInvalid(builder.bytes, containing: "duplicate tensor name w")
    }

    /// An empty array never enters the element loop, so the element type has to
    /// be validated before it.
    @Test func anEmptyArrayValidatesItsElementType() {
        var unknown = GGUFBuilder.header(keys: 1)
        unknown.string("a")
        unknown.u32(9)
        unknown.u32(13)  // no metadata value has type 13
        unknown.u64(0)
        expectInvalid(unknown.bytes, containing: "array element type 13")

        var nested = GGUFBuilder.header(keys: 1)
        nested.string("a")
        nested.u32(9)
        nested.u32(9)  // arrays of arrays are not read here
        nested.u64(0)
        expectInvalid(nested.bytes, containing: "nested array")
    }

    /// A long numeric array is carried as its length. It must not cost one
    /// iteration per element, and it must not be truncated into `.numbers`.
    @Test func aLongNumericArrayIsKeptAsItsLength() throws {
        let count = GGUFHeaderV3.maximumKeptArrayCount + 1
        var builder = GGUFBuilder.header(keys: 1)
        builder.string("a")
        builder.u32(9)
        builder.u32(0)  // uint8 array
        builder.u64(count)
        builder.raw([UInt8](repeating: 0, count: Int(count)))
        let header = try GGUFHeaderV3(bytes: builder.bytes)
        #expect(header.metadata["a"] == .longArray(count: count))
    }

    /// A tokenizer vocabulary is a string array, and it is never retained; it
    /// is still walked (so every element is validated as UTF-8 and charged to
    /// the string budget) and kept as its length.
    @Test func aValidStringArrayIsWalkedAndKeptAsItsLength() throws {
        var builder = GGUFBuilder.header(keys: 1)
        builder.string("tokenizer.ggml.tokens")
        builder.u32(9)
        builder.u32(8)
        builder.u64(3)
        for token in ["<s>", "hello", " world"] { builder.string(token) }
        let header = try GGUFHeaderV3(bytes: builder.bytes)
        #expect(header.metadata["tokenizer.ggml.tokens"] == .longArray(count: 3))
    }

    /// A declared count past the walk budget is refused before any payload is
    /// requested: the point is that the count cannot become a read or a loop.
    @Test func anOverBudgetArrayIsRefusedImmediately() {
        var builder = GGUFBuilder.header(keys: 1)
        builder.string("a")
        builder.u32(9)
        builder.u32(8)  // string elements
        builder.u64(GGUFHeaderV3.maximumArrayElements + 1)
        builder.raw([0])
        expectInvalid(builder.bytes, containing: "arrays hold more than")
    }

    @Test func aShortArrayPayloadAsksForMoreBytes() {
        var builder = GGUFBuilder.header(keys: 1)
        builder.string("a")
        builder.u32(9)
        builder.u32(4)  // uint32 elements
        builder.u64(1_000)
        builder.raw([0, 0, 0, 0])
        do {
            _ = try GGUFHeaderV3(bytes: builder.bytes)
            Issue.record("a 4,000-byte array in a short header must ask for more")
        } catch let error as GGUFHeaderV3.ReadError {
            guard case .needMoreBytes(let atLeast) = error else {
                Issue.record("expected .needMoreBytes, got \(error)")
                return
            }
            #expect(atLeast > builder.bytes.count)
            #expect(atLeast <= GGUFHeaderV3.maximumHeaderBytes)
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }

    /// The old arithmetic computed `headerEnd + alignment - 1`; a metadata
    /// alignment near `UInt64.max` made `dataStart` absurd (and the local
    /// caller then trapped converting it to `Int`).
    @Test func anAbsurdAlignmentIsRefused() {
        var builder = GGUFBuilder.header(keys: 1)
        builder.alignment(1 << 63)
        expectInvalid(builder.bytes, containing: "data section starts at")
    }

    @Test func aTensorOffsetPastTheAddressSpaceIsRefused() {
        var builder = GGUFBuilder.header(tensors: 1)
        builder.tensor(name: "w", rank: 1, dimensions: [4], offset: UInt64.max - 31)
        expectInvalid(builder.bytes, containing: "offset overflows")
    }

    @Test func anInvalidTensorRankIsRefused() {
        for rank: UInt32 in [0, 5] {
            var builder = GGUFBuilder.header(tensors: 1)
            builder.tensor(name: "w", rank: rank, dimensions: [4], offset: 0)
            expectInvalid(builder.bytes, containing: "rank \(rank)")
        }
    }

    @Test func aZeroTensorDimensionIsRefused() {
        var builder = GGUFBuilder.header(tensors: 1)
        builder.tensor(name: "w", rank: 2, dimensions: [0, 4], offset: 0)
        expectInvalid(builder.bytes, containing: "zero dimension")
    }

    @Test func anOversizedStringIsRefusedWithoutAllocatingIt() {
        var builder = GGUFBuilder.header(keys: 1)
        builder.string("a")
        builder.u32(8)
        builder.u64(1 << 30)  // no payload follows; the length alone is invalid
        expectInvalid(builder.bytes, containing: "string length")
    }

    /// GGUF v2/v3 calls a key longer than 2^16-1 bytes invalid; the limit also
    /// keeps one metadata key from consuming the whole string budget.
    @Test func anOversizedMetadataKeyIsRefused() {
        var builder = GGUFBuilder.header(keys: 1)
        builder.string(String(repeating: "a", count: 65_536))
        builder.u32(0)
        builder.raw([1])
        expectInvalid(builder.bytes, containing: "metadata key is 65536 bytes")
    }

    /// The shared caller policy: grow by doubling, never past the budget, and
    /// stop (nil) when the hint itself asks for more than a header may be.
    @Test func headerGrowthIsBounded() {
        #expect(
            GGUFHeaderV3.nextHeaderReadSize(current: 16 << 20, atLeast: 20 << 20)
                == 32 << 20)
        #expect(
            GGUFHeaderV3.nextHeaderReadSize(current: 64 << 20, atLeast: 96 << 20)
                == 128 << 20)
        #expect(
            GGUFHeaderV3.nextHeaderReadSize(current: 64 << 20, atLeast: 128 << 20)
                == 128 << 20)
        #expect(
            GGUFHeaderV3.nextHeaderReadSize(
                current: 128 << 20, atLeast: 128 << 20) == nil)
        #expect(
            GGUFHeaderV3.nextHeaderReadSize(
                current: 16 << 20, atLeast: (128 << 20) + 1) == nil)
        #expect(
            GGUFHeaderV3.nextHeaderReadSize(current: 16 << 20, atLeast: 16 << 20)
                == nil)
        #expect(
            GGUFHeaderV3.nextHeaderReadSize(current: -1, atLeast: 16) == nil)
    }

    /// Deterministic, capped fuzz: any mutation either parses to a header that
    /// honours the budget or fails with the reader's own typed error. No crash,
    /// no foreign error and no unbounded allocation from a declared count.
    @Test func boundedFuzzOnlyFailsWithTheReadersError() {
        let seed = Self.tinyFile()
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> UInt64 {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return state
        }
        for iteration in 0..<2_000 {
            var bytes = Array(seed.prefix(4_096))
            switch next() % 4 {
            case 0:
                bytes = Array(bytes.prefix(Int(next() % UInt64(bytes.count + 1))))
            case 1:
                if !bytes.isEmpty {
                    let index = Int(next() % UInt64(bytes.count))
                    bytes[index] = UInt8(truncatingIfNeeded: next())
                }
            case 2:
                bytes.removeLast(Int(next() % UInt64(bytes.count + 1)))
                bytes += (0..<Int(next() % 64)).map { _ in
                    UInt8(truncatingIfNeeded: next())
                }
            default:
                if bytes.count > 24 {
                    let index = 8 + Int(next() % UInt64(bytes.count - 8))
                    bytes[index] = UInt8(truncatingIfNeeded: next())
                }
            }
            do {
                let header = try GGUFHeaderV3(bytes: bytes)
                #expect(
                    header.dataStart <= UInt64(GGUFHeaderV3.maximumHeaderBytes),
                    "iteration \(iteration)")
            } catch let error as GGUFHeaderV3.ReadError {
                if case .needMoreBytes(let atLeast) = error {
                    #expect(
                        atLeast <= GGUFHeaderV3.maximumHeaderBytes,
                        "iteration \(iteration): \(error)")
                }
            } catch {
                Issue.record("iteration \(iteration): unexpected error \(error)")
            }
        }
    }
}
