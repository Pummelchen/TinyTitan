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
        }
    }

    package static func resolve(directory: URL) throws -> Self {
        let resolved = try EngineFactories.pluginRegistry.resolve(directory)
        guard let identity = try resolved.modelIdentity() else {
            throw EngineFactoryError.unsupported(
                "no engine plugin declares an identity for \(directory.path)")
        }
        return .gpu(family: identity.family, reasoning: identity.reasoningControl)
    }
}
