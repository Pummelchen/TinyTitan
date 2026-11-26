import Testing
import Foundation
@testable import TinyTitanCLICore

/// `tinytitan catalog --check` (plan task 6, tracker row TT-077): the registry ↔
/// target ↔ qualification consistency gate. These pin the pure finding logic
/// with stub registries and target sets; the live command wires the real
/// providers (swift package describe + the Engines directory).
struct CatalogCheckTests {
    private func registry(_ variants: [[String: Any]]) -> [String: Any] {
        ["version": 1, "models": [["modelID": "m", "variants": variants]]]
    }

    @Test func consistentRegistryHasNoFindings() {
        let reg = registry([
            [
                "id": "m-4bit", "target": "Qwen35Dense2B4Bit", "shipped": false,
                "qualification": NSNull(),
            ]
        ])
        let findings = CatalogCommand.findings(
            registry: reg, targets: ["Qwen35Dense2B4Bit", "TinyTitan"],
            variantTargets: ["Qwen35Dense2B4Bit"])
        #expect(findings.isEmpty)
    }

    @Test func missingTargetIsNamed() {
        let reg = registry([
            [
                "id": "m-4bit", "target": "GhostTarget", "shipped": false,
                "qualification": NSNull(),
            ]
        ])
        let findings = CatalogCommand.findings(
            registry: reg, targets: ["TinyTitan"], variantTargets: [])
        #expect(findings.contains { $0.contains("GhostTarget") })
    }

    @Test func shippedVariantWithoutQualificationIsNamed() {
        let reg = registry([
            [
                "id": "m-4bit", "target": "Qwen35Dense2B4Bit", "shipped": true,
                "qualification": NSNull(),
            ]
        ])
        let findings = CatalogCommand.findings(
            registry: reg, targets: ["Qwen35Dense2B4Bit"],
            variantTargets: ["Qwen35Dense2B4Bit"])
        #expect(findings.contains { $0.contains("m-4bit") && $0.contains("qualification") })
    }

    @Test func shippedVariantWithQualificationIsAccepted() {
        let reg = registry([
            [
                "id": "m-4bit", "target": "Qwen35Dense2B4Bit", "shipped": true,
                "qualification": ["goldenSHA256": "abc"],
            ]
        ])
        let findings = CatalogCommand.findings(
            registry: reg, targets: ["Qwen35Dense2B4Bit"],
            variantTargets: ["Qwen35Dense2B4Bit"])
        #expect(findings.isEmpty)
    }

    @Test func orphanVariantTargetIsNamed() {
        let reg = registry([
            [
                "id": "m-4bit", "target": "Qwen35Dense2B4Bit", "shipped": false,
                "qualification": NSNull(),
            ]
        ])
        let findings = CatalogCommand.findings(
            registry: reg, targets: ["Qwen35Dense2B4Bit", "OrphanVariant"],
            variantTargets: ["Qwen35Dense2B4Bit", "OrphanVariant"])
        #expect(findings.contains { $0.contains("OrphanVariant") })
    }

    @Test func nullTargetIsSkippedNotFlagged() {
        // A variant whose target is not wired yet (null) is not a missing-target
        // finding; only a named-but-absent target is.
        let reg = registry([
            [
                "id": "m-8bit", "target": NSNull(), "shipped": false,
                "qualification": NSNull(),
            ]
        ])
        let findings = CatalogCommand.findings(
            registry: reg, targets: ["TinyTitan"], variantTargets: [])
        #expect(findings.isEmpty)
    }
}
