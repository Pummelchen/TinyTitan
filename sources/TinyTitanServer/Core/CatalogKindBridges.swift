import Foundation
import Legacy6xRuntime
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanLib
import TinyTitanPluginAPI
import TinyTitanVariants

/// 6.x `ModelFamily` bridges for `ModelCatalog.Kind`, kept in one host file so
/// `ModelRouter`, `ModelRouterError` and `main.swift` need not import the 6.x
/// runtime themselves (TT-134).
extension ModelCatalog.Kind {
    package var reasoningProfileTarget: (family: String, control: ReasoningControl) {
        switch self {
        case .gpu(let family, let control): (family, control)
        case .cpu: (ModelFamily.qwen36.rawValue, ModelFamily.qwen36.engineReasoningControl)
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
            family: ModelFamily.qwen36.rawValue,
            thinkingMode: thinking,
            reasoningEffort: nil,
            reasoningControl: ModelFamily.qwen36.engineReasoningControl)
    }
}
