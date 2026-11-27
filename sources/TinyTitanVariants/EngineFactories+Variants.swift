import Foundation
import Ornith35B4Bit
import TinyTitanCore
import TinyTitanLib
import TinyTitanPluginAPI

extension EngineFactories {

    // MARK: - TT-097 Phase 3 Task 10: the shipped variants as server engines

    /// Which engine a catalog identity selects where an install is named by
    /// its manifest identity rather than by a directory. The table is explicit
    /// and small: a shipped variant joins it when it qualifies (registry
    /// `shipped: true`, `catalog --check` green). An identity the table does
    /// not name has no engine here — the plugins own everything the host
    /// serves, and the host refuses the rest by name (TT-142). `Equatable` so
    /// a test can pin the selection without running a model.
    ///
    /// The two Inkling installs are selected by their own probes and carry no
    /// family/width row here; `routingFactory(directoryURL:)` and
    /// `gpuFactory(directoryURL:)` are the general path.
    public enum GPUVariantSelection: Equatable {
        case ornith35B4Bit
    }

    /// The selection for a GPU catalog identity, decided from the manifest
    /// identity the entry's directory carries: the shipped Ornith 1.5 35B-A3B
    /// install at 4 bits routes to its variant, and nil means no shipped
    /// variant claims the identity.
    public static func gpuSelection(
        family: String,
        manifestModelID: String,
        routedExpertWeightBits: Int
    ) -> GPUVariantSelection? {
        if family == "qwen36",
            manifestModelID.hasPrefix("ornith-1.5-35b"),
            routedExpertWeightBits == 4
        {
            return .ornith35B4Bit
        }
        return nil
    }

    /// The GPU factory a catalog identity selects. Resolved through the
    /// plugin registry (TPP/1, TT-130): the strongest plugin probe wins. An
    /// install no plugin claims is refused here by name; there is no
    /// fallback engine (TT-142).
    public static func gpuFactory(directoryURL: URL) throws -> EngineFactory {
        try pluginRegistry.resolve(directoryURL).engineFactory()
    }

    /// The GPU factory a single-model front end (the CLI) and an embedder
    /// use: the same plugin resolution the server's router uses. A directory
    /// no plugin serves is refused by name; there is no fallback engine.
    public static func routingFactory(directoryURL: URL) throws -> EngineFactory {
        try pluginRegistry.resolve(directoryURL).engineFactory()
    }

    /// The factory for a catalog identity, through the selection table. An
    /// identity no shipped variant claims is refused by name rather than
    /// handed to an engine for a family that is not supported (TT-142).
    public static func gpuFactory(
        family: String,
        manifestModelID: String,
        routedExpertWeightBits: Int
    ) throws -> EngineFactory {
        switch gpuSelection(
            family: family,
            manifestModelID: manifestModelID,
            routedExpertWeightBits: routedExpertWeightBits)
        {
        case .ornith35B4Bit: return Ornith35B4BitPlugin.engineFactory(host: .standard)
        case nil:
            throw EngineFactoryError.unsupported(
                "no shipped GPU engine serves family \(family) model "
                    + "\(manifestModelID) at \(routedExpertWeightBits) bits")
        }
    }
}
