import Foundation
import TinyTitanCore
import TinyTitanLib
import TinyTitanPluginAPI
import TinyTitanVariants

/// Bridges from a catalog kind to the host's reasoning surfaces, kept in one
/// file so `ModelRouter`, `ModelRouterError` and `main.swift` need not import
/// the pieces themselves.
extension ModelCatalog.Kind {
    package var reasoningProfileTarget: (family: String, control: ReasoningControl) {
        switch self {
        case .gpu(let family, let control): (family, control)
        case .cpu(let family, let control): (family, control)
        }
    }

    package static func resolve(directory: URL, cpu: Bool) throws -> Self {
        if cpu {
            let family = try ModelCatalog.snapshotFamily(directory)
            return .cpu(family: family.rawValue, reasoning: family.reasoningControl)
        }
        let resolved = try EngineFactories.pluginRegistry.resolve(directory)
        guard let identity = try resolved.modelIdentity() else {
            throw EngineFactoryError.unsupported(
                "no engine plugin declares an identity for \(directory.path)")
        }
        return .gpu(family: identity.family, reasoning: identity.reasoningControl)
    }

    package static func cpuReasoningProfile(
        thinking: ModelThinkingMode
    ) -> ServerReasoningProfile {
        ServerReasoningProfile(
            family: VariantCPUModelFamily.qwen35Dense.rawValue,
            thinkingMode: thinking,
            reasoningEffort: nil,
            reasoningControl: VariantCPUModelFamily.qwen35Dense.reasoningControl)
    }
}
