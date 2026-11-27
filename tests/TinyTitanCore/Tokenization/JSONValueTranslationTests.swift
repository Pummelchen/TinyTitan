import Foundation
import Testing

@testable import TinyTitanCore

/// The Jinja translation of a tool schema must not use stack proportional to
/// the document's depth.
///
/// It used to recurse (`mapValues { $0.jinjaSendableValue() }`). A schema at
/// the validator's maximum still nests two JSON levels per schema level, and
/// under ASan that walk exhausted a test thread's stack inside `mapValues` --
/// the process died with SIGILL and no test report, which is what sent the
/// full sanitizer run red. The translation is iterative now; these pin that,
/// and pin the scalars it refused or converted before the rewrite.
@Suite struct JSONValueTranslationTests {
    /// Deep enough to be well past anything a client writes, and shallow
    /// enough that tearing the nested value down does not recurse into
    /// trouble on the test task's own stack: the dealloc of an indirect enum
    /// this shape is recursive, and measured to die (SIGBUS, signal 10) at
    /// 3,000 levels on this thread. The stack property itself is proven by
    /// the ASan run of `OpenAIValidationTests`, which is where the recursive
    /// translation used to die.
    static let levels = 200

    static func nested(to depth: Int, leaf: JSONValue = .string("leaf")) -> JSONValue {
        var value = leaf
        for _ in 0..<depth { value = .object(["a": value]) }
        return value
    }

    @Test func aDeepObjectTranslatesWithoutRecursion() throws {
        let translation = try Self.nested(to: Self.levels).jinjaSendableValue()
        var node: any Sendable = translation
        var depth = 0
        while let object = node as? [String: any Sendable], let child = object["a"] {
            node = child
            depth += 1
        }
        #expect(depth == Self.levels)
        #expect(node as? String == "leaf")
    }

    @Test func aDeepArrayTranslatesWithoutRecursion() throws {
        var value = JSONValue.string("leaf")
        for _ in 0..<Self.levels { value = .array([value]) }
        let translation = try value.jinjaSendableValue()
        var node: any Sendable = translation
        var depth = 0
        while let array = node as? [any Sendable], let first = array.first {
            node = first
            depth += 1
        }
        #expect(depth == Self.levels)
        #expect(node as? String == "leaf")
    }

    @Test func scalarsTranslateAsBefore() throws {
        #expect(try JSONValue.string("s").jinjaSendableValue() as? String == "s")
        #expect(try JSONValue.integer(7).jinjaSendableValue() as? Int == 7)
        #expect(try JSONValue.bool(true).jinjaSendableValue() as? Bool == true)
        let null = try JSONValue.null.jinjaSendableValue()
        #expect(null is String?)
        let object = try JSONValue.object([
            "a": .array([.string("x"), .integer(2)])
        ]).jinjaSendableValue()
        let dict = try #require(object as? [String: any Sendable])
        let array = try #require(dict["a"] as? [any Sendable])
        #expect(array.count == 2)
        #expect(array[0] as? String == "x")
        #expect(array[1] as? Int == 2)
    }

    @Test func aNumberThatCannotRoundTripIsStillRefused() throws {
        // A decimal with more precision than a double can carry is refused
        // rather than handed to Jinja as a rounded number.
        let precise = try #require(
            Decimal(
                string: "0.10000000000000000000000001", locale: Locale(identifier: "en_US_POSIX")))
        #expect(throws: ToolCallParserError.self) {
            try JSONValue.decimal(precise).jinjaSendableValue()
        }
        // A decimal a double represents exactly still translates.
        let exact = try #require(Decimal(string: "0.5", locale: Locale(identifier: "en_US_POSIX")))
        #expect(try JSONValue.decimal(exact).jinjaSendableValue() as? Double == 0.5)
    }
}
