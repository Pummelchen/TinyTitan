import Foundation
import DenseGPU
import Inkling2GGUFQ4
import Inkling2NVFP4
import Ornith35B4Bit
import TemplatePlugin
import TinyTitanCore
import TinyTitanPluginAPI

extension EngineFactories {
    /// The engine plugins compiled into this build (TPP/1, TT-130), strongest
    /// first only for readability: resolution is by probe strength.
    ///
    /// `TemplatePlugin`, `Inkling2GGUFQ4Plugin`, `Ornith35B4BitPlugin` and
    /// `DenseQwen35Plugin` are standalone plugins. There is no fallback
    /// adapter: an install no plugin claims is refused by name, because the
    /// shipped matrix is exactly what these plugins serve (TT-142).
    public static let builtInPlugins: [any EnginePlugin.Type] = [
        TemplatePlugin.self,
        Inkling2GGUFQ4Plugin.self,
        Inkling2NVFP4Plugin.self,
        Ornith35B4BitPlugin.self,
        DenseQwen35Plugin.self,
    ]

    /// The process's registry with the built-in plugins registered. An
    /// embedder registers its own plugins on the same registry.
    public static var pluginRegistry: PluginRegistry {
        _ = registerBuiltIns
        return .shared
    }

    private static let registerBuiltIns: Void = {
        for plugin in builtInPlugins {
            // Every built-in is built against this protocol version.
            try? PluginRegistry.shared.register(plugin)
        }
    }()
}
