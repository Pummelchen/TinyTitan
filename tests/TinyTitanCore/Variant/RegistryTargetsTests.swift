import Testing
import Foundation
@testable import TinyTitanCore

/// The registry ↔ variant-target wiring (plan task 5, tracker row TT-076). The
/// registry is the shipped matrix's inventory: three variants, each naming its
/// SwiftPM target, plus the engine targets that are in the tree with no
/// supported install. These tests pin that shape before any graph is ported.
struct RegistryTargetsTests {
    private static func registryURL() -> URL {
        let relative = "registry/registry.json"
        if FileManager.default.fileExists(atPath: relative) {
            return URL(fileURLWithPath: relative)
        }
        var dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0...6 {
            if FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("Package.swift").path)
            {
                return dir.appendingPathComponent(relative)
            }
            dir.deleteLastPathComponent()
        }
        return URL(fileURLWithPath: relative)
    }

    private static func registry() throws -> [String: Any] {
        let data = try Data(contentsOf: registryURL())
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func models() throws -> [[String: Any]] {
        let registry = try registry()
        return try #require(registry["models"] as? [[String: Any]])
    }

    private static func variants() throws -> [[String: Any]] {
        var rows: [[String: Any]] = []
        for model in try models() {
            rows += try #require(model["variants"] as? [[String: Any]])
        }
        return rows
    }

    private static func variant(_ id: String) throws -> [String: Any] {
        for variant in try variants() where variant["id"] as? String == id {
            return variant
        }
        throw RegistryTestError.missingVariant(id)
    }

    enum RegistryTestError: Error { case missingVariant(String) }

    private static func engineDirectory(_ target: String) -> URL {
        let root = registryURL().deletingLastPathComponent().deletingLastPathComponent()
        return root.appendingPathComponent("sources/Engines/\(target)")
    }

    @Test func theFirstVariantNamesItsSwiftPMTarget() throws {
        let variant = try Self.variant("ornith-1.5-35b-4bit")
        #expect(variant["target"] as? String == "Ornith35B4Bit")
        #expect(variant["shipped"] as? Bool == true)
    }

    @Test func theVariantTargetDirectoryExists() throws {
        // The target named by the registry must exist as a source directory.
        var isDirectory: ObjCBool = false
        #expect(
            FileManager.default.fileExists(
                atPath: Self.engineDirectory("Ornith35B4Bit").path, isDirectory: &isDirectory)
                && isDirectory.boolValue)
    }

    @Test func everyRegisteredVariantNamesAnExistingEngine() throws {
        for variant in try Self.variants() {
            let id = variant["id"] as? String ?? "<unnamed>"
            let target = try #require(variant["target"] as? String, "\(id) names no target")
            var isDirectory: ObjCBool = false
            #expect(
                FileManager.default.fileExists(
                    atPath: Self.engineDirectory(target).path, isDirectory: &isDirectory)
                    && isDirectory.boolValue,
                "\(id) names \(target), which is not a directory under sources/Engines")
        }
    }

    @Test func theRegistryListsOnlyTheSupportedMatrix() throws {
        // The shipped surface (AGENTS.md): Ornith 1.5 35B-A3B 4-bit and the two
        // Inkling 2 Mini installs. A family that left the matrix is not an
        // inventory row, so a re-added entry is a deliberate act.
        let ids = try Self.variants().compactMap { $0["id"] as? String }
        #expect(
            Set(ids)
                == Set([
                    "ornith-1.5-35b-4bit", "inkling-2-mini-gguf-4bit", "inkling-2-mini-nvfp4-4bit",
                ]))
    }

    @Test func theUnshippedTargetsExistAndAreNotRegistered() throws {
        // `unshippedTargets` is how the dense engines stay in the tree without a
        // supported install (TT-142 slice (b)). A row that points at nothing, or
        // that is also a registered variant, would be the exclusion hiding a
        // shipped engine.
        let registry = try Self.registry()
        let unshipped = try #require(registry["unshippedTargets"] as? [String])
        let registered = Set(try Self.variants().compactMap { $0["target"] as? String })
        #expect(!unshipped.isEmpty)
        for target in unshipped {
            var isDirectory: ObjCBool = false
            #expect(
                FileManager.default.fileExists(
                    atPath: Self.engineDirectory(target).path, isDirectory: &isDirectory)
                    && isDirectory.boolValue,
                "unshipped target \(target) is not a directory under sources/Engines")
            #expect(!registered.contains(target), "unshipped target \(target) is also registered")
        }
    }
}
