import Foundation

/// The header of a GGUF file — metadata and tensor table — read from its first
/// bytes, so an installer can plan a conversion before fetching the data
/// section (TT-121).
///
/// It keeps what a planner needs: every scalar and string, numeric arrays up
/// to `maximumKeptArrayCount` elements, and only the length of longer arrays
/// (a 200K-entry tokenizer vocabulary is carried into an install by copying the
/// header bytes, not by parsing it here). Tensor offsets are absolute, already
/// including the data section's start. A header longer than the bytes given
/// throws `.needMoreBytes` with a lower bound to fetch, so a remote reader can
/// widen its range and retry.
///
/// **Resource budgets.** A GGUF header comes from a downloaded file, so its
/// declared sizes are untrusted. The parser never allocates from a declared
/// count before the payload is present, and it enforces four budgets: the
/// header's own bytes (`maximumHeaderBytes`), the aggregate bytes of strings
/// (`maximumStringBytes`), the aggregate elements of arrays it has to walk
/// (`maximumArrayElements`), and the per-string length. Callers grow their
/// reads through `nextHeaderReadSize`, which cannot ask past the header budget.
///
/// **Duplicates are refused.** A metadata key and a tensor name each identify
/// one value; a second occurrence would make `metadata[key]` and the planner's
/// name-keyed dictionaries silently keep one of two different values. Both are
/// rejected with `.invalid` before any lookup can see them. GGUF writers store
/// metadata in a dictionary, so a valid file carries each name once.
///
/// **Keys and values follow the format's contract.** A metadata key is ASCII
/// and dot-separated `lower_snake_case` segments (the standard keys, including
/// numeric segments such as `general.base_model.0.name`, all qualify); a BOOL
/// is exactly 0 or 1. **Arrays are one level deep**: this reader stores a
/// scalar, a flat numeric array, or a string array's length. The GGUF spec
/// permits arrays of arrays, but no model in this tree uses one, and supporting
/// it would need a recursive value type; a nested array is therefore a typed
/// `.invalid("nested array")`, never a flattened or dropped level.
package struct GGUFHeaderV3: Sendable {
    package enum Value: Sendable, Equatable {
        case integer(Int64)
        case unsigned(UInt64)
        case real(Double)
        case boolean(Bool)
        case string(String)
        case numbers([Double])
        /// An array kept only as its length.
        case longArray(count: UInt64)
    }

    package struct Tensor: Sendable, Equatable {
        package let name: String
        /// ggml order: `dimensions[0]` is the row length (fastest varying).
        package let dimensions: [UInt64]
        package let ggmlType: UInt32
        /// Absolute file offset of the tensor's first byte.
        package let absoluteOffset: UInt64
    }

    package enum ReadError: Error, Equatable, CustomStringConvertible {
        case needMoreBytes(atLeast: Int)
        case invalid(String)

        package var description: String {
            switch self {
            case .needMoreBytes(let count):
                "GGUF header continues past the bytes read; read at least \(count)"
            case .invalid(let reason): "GGUF header: \(reason)"
            }
        }
    }

    /// The largest header this reader will materialise, padding included.
    ///
    /// The real Inkling 2 Q4_K_XL header is 12 MB; 128 MiB is ten times that
    /// and still bounds what a hostile file can make an installer read into
    /// memory before the data section. Both callers cap their growth here.
    package static let maximumHeaderBytes: Int = 128 << 20
    package static let maximumKeptArrayCount: UInt64 = 65_536
    /// Aggregate elements of every array the parser has to walk. A file with a
    /// 200K-token vocabulary is far below this; the bound is what stops a
    /// declared count from turning into an iteration count.
    package static let maximumArrayElements: UInt64 = 8 << 20
    /// Aggregate bytes of metadata keys, strings and string-array elements.
    package static let maximumStringBytes: UInt64 = 64 << 20
    /// The GGUF v2/v3 key limit; the format calls a longer key invalid.
    package static let maximumMetadataKeyBytes = 65_535
    package static let maximumMetadataKeyCount: UInt64 = 1 << 20
    package static let maximumTensorCount: UInt64 = 1 << 20

    package let version: UInt32
    package let metadata: [String: Value]
    package let tensors: [Tensor]
    package let alignment: UInt64
    /// Absolute offset of the data section: the header's end, aligned.
    package let dataStart: UInt64

    package init(bytes: [UInt8]) throws {
        var cursor = Cursor(bytes: bytes, limit: Self.maximumHeaderBytes)
        guard Array(try cursor.raw(4)) == Array("GGUF".utf8) else {
            throw ReadError.invalid("bad magic")
        }
        version = try cursor.u32()
        guard version == 2 || version == 3 else { throw ReadError.invalid("version \(version)") }
        let tensorCount = try cursor.u64()
        let keyCount = try cursor.u64()
        guard tensorCount < Self.maximumTensorCount, keyCount < Self.maximumMetadataKeyCount else {
            throw ReadError.invalid("implausible counts")
        }
        var metadata: [String: Value] = [:]
        for _ in 0..<keyCount {
            let key = try cursor.string()
            try Self.validateMetadataKey(key)
            guard metadata[key] == nil else {
                throw ReadError.invalid("duplicate metadata key \(key)")
            }
            metadata[key] = try cursor.value(type: try cursor.u32())
        }
        var tensorNames: Set<String> = []
        var pending: [(String, [UInt64], UInt32, UInt64)] = []
        for _ in 0..<tensorCount {
            let name = try cursor.string()
            guard tensorNames.insert(name).inserted else {
                throw ReadError.invalid("duplicate tensor name \(name)")
            }
            let rank = try cursor.u32()
            guard rank >= 1, rank <= 4 else { throw ReadError.invalid("\(name): rank \(rank)") }
            var dimensions: [UInt64] = []
            for _ in 0..<rank { dimensions.append(try cursor.u64()) }
            guard dimensions.allSatisfy({ $0 > 0 }) else {
                throw ReadError.invalid("\(name): zero dimension")
            }
            pending.append((name, dimensions, try cursor.u32(), try cursor.u64()))
        }
        var alignment: UInt64 = 32
        switch metadata["general.alignment"] {
        case .unsigned(let value)?: alignment = value
        case .integer(let value)? where value > 0: alignment = UInt64(value)
        default: break
        }
        guard alignment > 0, alignment & (alignment - 1) == 0 else {
            throw ReadError.invalid("alignment \(alignment)")
        }
        let headerEnd = UInt64(cursor.position)
        let (padded, overflow) = headerEnd.addingReportingOverflow(alignment - 1)
        guard !overflow else {
            throw ReadError.invalid("alignment \(alignment) overflows the data start")
        }
        let start = (padded / alignment) * alignment
        guard start >= headerEnd, start <= UInt64(Self.maximumHeaderBytes) else {
            throw ReadError.invalid(
                "data section starts at \(start); a GGUF header may be at most "
                    + "\(Self.maximumHeaderBytes) bytes")
        }
        dataStart = start
        self.alignment = alignment
        self.metadata = metadata
        tensors = try pending.map { name, dimensions, type, offset in
            guard offset % alignment == 0 else {
                throw ReadError.invalid("\(name): unaligned offset")
            }
            let (absolute, overflow) = offset.addingReportingOverflow(start)
            guard !overflow else { throw ReadError.invalid("\(name): offset overflows") }
            return Tensor(
                name: name, dimensions: dimensions, ggmlType: type, absoluteOffset: absolute)
        }
    }

    /// The next read after a `.needMoreBytes` hint.
    ///
    /// Doubles the current read, never past `maximumHeaderBytes`, and returns
    /// nil when the hint asks for a header larger than any GGUF may be. A
    /// caller that gets nil reports a typed error instead of re-reading a file
    /// in ever larger pieces.
    package static func nextHeaderReadSize(
        current: Int,
        atLeast: Int,
        maximum: Int = maximumHeaderBytes
    ) -> Int? {
        guard current >= 0, atLeast > current, atLeast <= maximum else { return nil }
        let (doubled, overflow) = current.multipliedReportingOverflow(by: 2)
        let grown = max(overflow ? maximum : doubled, atLeast)
        guard grown > current else { return nil }
        return min(grown, maximum)
    }

    /// A metadata number, whatever integer or float type the file used.
    package func number(_ key: String) -> Double? {
        switch metadata[key] {
        case .integer(let value)?: Double(value)
        case .unsigned(let value)?: Double(value)
        case .real(let value)?: value
        case .boolean(let value)?: value ? 1 : 0
        default: nil
        }
    }

    package func string(_ key: String) -> String? {
        if case .string(let value)? = metadata[key] { return value }
        return nil
    }

    package func numbers(_ key: String) -> [Double]? {
        if case .numbers(let value)? = metadata[key] { return value }
        return nil
    }

    /// The GGUF key contract: at most 65 535 bytes, ASCII, and dot-separated
    /// `lower_snake_case` segments. Numeric segments are allowed because the
    /// spec itself defines `general.base_model.{id}.name`.
    private static func validateMetadataKey(_ key: String) throws {
        guard key.utf8.count <= maximumMetadataKeyBytes else {
            throw ReadError.invalid(
                "metadata key is \(key.utf8.count) bytes; the format allows "
                    + "\(maximumMetadataKeyBytes)")
        }
        var segmentBytes = 0
        for byte in key.utf8 {
            if byte == UInt8(ascii: ".") {
                guard segmentBytes > 0 else {
                    throw ReadError.invalid(
                        "metadata key \(key) has an empty segment")
                }
                segmentBytes = 0
                continue
            }
            let isLowercase = byte >= UInt8(ascii: "a") && byte <= UInt8(ascii: "z")
            let isDigit = byte >= UInt8(ascii: "0") && byte <= UInt8(ascii: "9")
            let isUnderscore = byte == UInt8(ascii: "_")
            guard isLowercase || isDigit || isUnderscore else {
                throw ReadError.invalid(
                    "metadata key \(key) is not ASCII lower_snake_case")
            }
            segmentBytes += 1
        }
        guard segmentBytes > 0 else {
            throw ReadError.invalid("metadata key \(key) has an empty segment")
        }
    }
}

private struct Cursor {
    let bytes: [UInt8]
    /// The largest position this cursor may reach: the header budget.
    let limit: Int
    var position = 0
    var arrayElements: UInt64 = 0
    var stringBytes: UInt64 = 0

    mutating func raw(_ count: Int) throws -> ArraySlice<UInt8> {
        guard count >= 0 else {
            throw GGUFHeaderV3.ReadError.invalid("negative read \(count)")
        }
        let (end, overflow) = position.addingReportingOverflow(count)
        guard !overflow else {
            throw GGUFHeaderV3.ReadError.invalid(
                "read of \(count) bytes at \(position) overflows")
        }
        guard end <= limit else {
            throw GGUFHeaderV3.ReadError.invalid(
                "header continues past the \(limit)-byte budget")
        }
        guard end <= bytes.count else {
            let (doubled, doubledOverflow) = bytes.count.multipliedReportingOverflow(by: 2)
            let growth = doubledOverflow ? limit : min(doubled, limit)
            throw GGUFHeaderV3.ReadError.needMoreBytes(atLeast: max(growth, end))
        }
        defer { position = end }
        return bytes[position..<end]
    }

    mutating func skip(_ count: Int) throws {
        _ = try raw(count)
    }

    mutating func u32() throws -> UInt32 {
        try raw(4).reversed().reduce(0) { $0 << 8 | UInt32($1) }
    }

    mutating func u64() throws -> UInt64 {
        try raw(8).reversed().reduce(0) { $0 << 8 | UInt64($1) }
    }

    mutating func string() throws -> String {
        let length = try u64()
        guard length < 1 << 30 else {
            throw GGUFHeaderV3.ReadError.invalid("string length \(length)")
        }
        let (total, overflow) = stringBytes.addingReportingOverflow(length)
        guard !overflow, total <= GGUFHeaderV3.maximumStringBytes else {
            throw GGUFHeaderV3.ReadError.invalid(
                "strings total more than \(GGUFHeaderV3.maximumStringBytes) bytes")
        }
        stringBytes = total
        guard let text = String(bytes: try raw(Int(length)), encoding: .utf8) else {
            throw GGUFHeaderV3.ReadError.invalid("string is not UTF-8")
        }
        return text
    }

    /// Reads one value of ggml metadata `type`; arrays are kept as numbers or
    /// as a length, never recursed into storage.
    mutating func value(type: UInt32) throws -> GGUFHeaderV3.Value {
        switch type {
        case 0: return .unsigned(UInt64((try raw(1).first ?? 0)))
        case 1: return .integer(Int64(Int8(bitPattern: (try raw(1).first ?? 0))))
        case 2:
            return .unsigned(
                UInt64(try raw(2).reversed().reduce(UInt16(0)) { $0 << 8 | UInt16($1) }))
        case 3:
            let raw16 = try raw(2).reversed().reduce(UInt16(0)) { $0 << 8 | UInt16($1) }
            return .integer(Int64(Int16(bitPattern: raw16)))
        case 4: return .unsigned(UInt64(try u32()))
        case 5: return .integer(Int64(Int32(bitPattern: try u32())))
        case 6: return .real(Double(Float(bitPattern: try u32())))
        case 7:
            let byte = try raw(1).first ?? 0
            guard byte <= 1 else {
                throw GGUFHeaderV3.ReadError.invalid(
                    "boolean value \(byte) is not 0 or 1")
            }
            return .boolean(byte == 1)
        case 8: return .string(try string())
        case 9: return try array()
        case 10: return .unsigned(try u64())
        case 11: return .integer(Int64(bitPattern: try u64()))
        case 12: return .real(Double(bitPattern: try u64()))
        default: throw GGUFHeaderV3.ReadError.invalid("metadata type \(type)")
        }
    }

    private mutating func array() throws -> GGUFHeaderV3.Value {
        let elementType = try u32()
        let count = try u64()
        // Validate the element type even when the array is empty: an empty
        // array of a type no value may have is still an invalid document.
        guard let elementBytes = Self.elementBytes(elementType) else {
            throw GGUFHeaderV3.ReadError.invalid(
                elementType == 9
                    ? "nested array" : "array element type \(elementType)")
        }
        let keep = count <= GGUFHeaderV3.maximumKeptArrayCount && elementType != 8
        // A fixed-width array that is not retained costs one bounds check, not
        // one iteration per element: the planner only keeps its length.
        if !keep, elementType != 8 {
            try skip(count: count, elementBytes: elementBytes)
            return .longArray(count: count)
        }
        // Arrays the parser has to walk are charged to one aggregate budget
        // before any payload is requested, so a declared count cannot turn
        // into a large read or a long loop.
        guard count <= GGUFHeaderV3.maximumArrayElements - arrayElements else {
            throw GGUFHeaderV3.ReadError.invalid(
                "arrays hold more than \(GGUFHeaderV3.maximumArrayElements) elements")
        }
        let minimumPayload = try Self.payloadBytes(count: count, elementBytes: elementBytes)
        try requirePayload(minimumPayload)
        arrayElements += count
        var numbers: [Double] = []
        for _ in 0..<count {
            let element = try value(type: elementType)
            guard keep else { continue }
            switch element {
            case .integer(let v): numbers.append(Double(v))
            case .unsigned(let v): numbers.append(Double(v))
            case .real(let v): numbers.append(v)
            case .boolean(let v): numbers.append(v ? 1 : 0)
            default: break
            }
        }
        return keep ? .numbers(numbers) : .longArray(count: count)
    }

    /// Advances past a fixed-width array without materialising its elements.
    private mutating func skip(count: UInt64, elementBytes: UInt64) throws {
        let payload = try Self.payloadBytes(count: count, elementBytes: elementBytes)
        try requirePayload(payload)
        guard payload <= UInt64(Int.max) else {
            throw GGUFHeaderV3.ReadError.invalid(
                "array payload of \(payload) bytes is not addressable")
        }
        try skip(Int(payload))
    }

    private static func payloadBytes(count: UInt64, elementBytes: UInt64) throws -> UInt64 {
        let (payload, overflow) = count.multipliedReportingOverflow(by: elementBytes)
        guard !overflow else {
            throw GGUFHeaderV3.ReadError.invalid(
                "array of \(count) elements of \(elementBytes) bytes overflows")
        }
        return payload
    }

    /// Refuses a declared payload past the header budget and asks for more
    /// bytes when it is only absent, not impossible.
    private func requirePayload(_ payload: UInt64) throws {
        let (end, overflow) = UInt64(position).addingReportingOverflow(payload)
        guard !overflow, end <= UInt64(limit) else {
            throw GGUFHeaderV3.ReadError.invalid(
                "an array of \(payload) bytes runs past the \(limit)-byte header budget")
        }
        guard end <= UInt64(bytes.count) else {
            throw GGUFHeaderV3.ReadError.needMoreBytes(atLeast: Int(end))
        }
    }

    /// The minimum bytes one array element occupies, or nil for a type that
    /// may not appear in an array. A string's minimum is its 8-byte length.
    private static func elementBytes(_ type: UInt32) -> UInt64? {
        switch type {
        case 0, 1, 7: return 1
        case 2, 3: return 2
        case 4, 5, 6: return 4
        case 8, 10, 11, 12: return 8
        default: return nil
        }
    }
}
