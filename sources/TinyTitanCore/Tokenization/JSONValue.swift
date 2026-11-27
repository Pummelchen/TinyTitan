import Foundation

/// A structural JSON value, shared by the tool-call parsers and the 7.0 server
/// backend contract so a parsed tool argument has one type across the boundary.
public indirect enum JSONValue: Codable, Equatable, Sendable {
    case object([String: JSONValue])
    case array([JSONValue])
    case string(String)
    case integer(Int64)
    case unsignedInteger(UInt64)
    case decimal(Decimal)
    case number(Double)
    case bool(Bool)
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int64.self) {
            self = .integer(value)
        } else if let value = try? container.decode(UInt64.self) {
            self = .unsignedInteger(value)
        } else if let value = try? container.decode(Decimal.self) {
            self = .decimal(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .integer(let value): try container.encode(value)
        case .unsignedInteger(let value): try container.encode(value)
        case .decimal(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }

    /// Translate to `any Sendable` containers for the Jinja tool template.
    ///
    /// Iterative on purpose: the input is client-supplied JSON, and a recursive
    /// translation makes stack use proportional to the document's depth. A
    /// schema at the validator's own maximum still nests two JSON levels per
    /// schema level, which measured out at several hundred kilobytes of stack
    /// under ASan and killed a test thread inside `mapValues`. The explicit
    /// frame stack below is bounded by heap, not by the document's shape.
    public func jinjaSendableValue() throws -> any Sendable {
        enum Frame {
            case object(
                keys: [String], values: [JSONValue], next: Int,
                built: [String: any Sendable])
            case array(values: [JSONValue], next: Int, built: [any Sendable])
        }

        var stack: [Frame] = []
        var pending: JSONValue = self
        var result: any Sendable = 0

        while true {
            switch pending {
            case .object(let object):
                let keys = object.keys.sorted()
                let values = keys.compactMap { object[$0] }
                if let first = values.first {
                    stack.append(.object(keys: keys, values: values, next: 0, built: [:]))
                    pending = first
                    continue
                }
                result = [String: any Sendable]()
            case .array(let array):
                if let first = array.first {
                    stack.append(.array(values: array, next: 0, built: []))
                    pending = first
                    continue
                }
                result = [any Sendable]()
            default:
                // A scalar is where the translation can refuse a value.
                result = try Self.scalar(pending)
            }

            // `result` is complete: hand it to its parent, and keep folding for
            // as long as each parent that receives it was itself the last child.
            var descended = false
            while let frame = stack.popLast() {
                switch frame {
                case .object(let keys, let values, var next, var built):
                    built[keys[next]] = result
                    next += 1
                    if next < values.count {
                        stack.append(
                            .object(keys: keys, values: values, next: next, built: built))
                        pending = values[next]
                        descended = true
                    } else {
                        result = built
                    }
                case .array(let values, var next, var built):
                    built.append(result)
                    next += 1
                    if next < values.count {
                        stack.append(.array(values: values, next: next, built: built))
                        pending = values[next]
                        descended = true
                    } else {
                        result = built
                    }
                }
                if descended { break }
            }
            if descended { continue }
            if stack.isEmpty { return result }
        }
    }

    /// One scalar's translation: the half of `jinjaSendableValue` that can
    /// throw, kept separate so the walk above stays flat.
    private static func scalar(_ value: JSONValue) throws -> any Sendable {
        switch value {
        case .object, .array:
            throw ToolCallParserError.malformed
        case .string(let value):
            return value
        case .integer(let value):
            guard let value = Int(exactly: value) else {
                throw ToolCallParserError.malformed
            }
            return value
        case .unsignedInteger(let value):
            guard let value = Int(exactly: value) else {
                throw ToolCallParserError.malformed
            }
            return value
        case .decimal(let value):
            let text = NSDecimalNumber(decimal: value).stringValue
            guard let double = Double(text),
                double.isFinite,
                let roundTrip = Decimal(
                    string: String(double),
                    locale: Locale(identifier: "en_US_POSIX")),
                roundTrip == value
            else {
                throw ToolCallParserError.malformed
            }
            return double
        case .number(let value):
            return value
        case .bool(let value):
            return value
        case .null:
            return Optional<String>.none as String?
        }
    }

    public var objectValue: [String: JSONValue]? {
        guard case .object(let value) = self else { return nil }
        return value
    }

    public func encoded(sortedKeys: Bool = true) throws -> String {
        let encoder = JSONEncoder()
        if sortedKeys { encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes] }
        return try encoder.encode(self).lossyUTF8String
    }
}
