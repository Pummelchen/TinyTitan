import Testing
import Foundation
@testable import TinyTitanCore

/// The registry ↔ variant-target wiring (plan task 5, tracker row TT-076). The
/// first variant target must be named by its registry entry; this pins that the
/// registry and the SwiftPM target agree before any graph is ported.
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

    private static func variant(_ id: String) throws -> [String: Any] {
        let data = try Data(contentsOf: registryURL())
        let registry = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let models = try #require(registry["models"] as? [[String: Any]])
        for model in models {
            for variant in try #require(model["variants"] as? [[String: Any]])
            where variant["id"] as? String == id {
                return variant
            }
        }
        throw RegistryTestError.missingVariant(id)
    }

    enum RegistryTestError: Error { case missingVariant(String) }

    @Test func theFirstVariantNamesItsSwiftPMTarget() throws {
        let variant = try Self.variant("qwen35-2b-4bit")
        #expect(variant["target"] as? String == "Qwen35Dense2B4Bit")
    }

    @Test func theVariantTargetDirectoryExists() throws {
        // The target named by the registry must exist as a source directory.
        let url = Self.registryURL()
        let root = url.deletingLastPathComponent().deletingLastPathComponent()
        let targetDir = root.appendingPathComponent("sources/Engines/Qwen35Dense2B4Bit")
        #expect(FileManager.default.fileExists(atPath: targetDir.path))
    }
}
