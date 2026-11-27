import Foundation
import Legacy6xRuntime
import TinyTitanCore

package enum ServerModelIdentity {
    package static func base(manifestModelID: String, family: ModelFamily) -> String {
        var id = manifestModelID
        for suffix in ["-4bit", "-8bit", "-6bit"] where id.hasSuffix(suffix) {
            id = String(id.dropLast(suffix.count))
        }
        guard id == "unknown/snapshot" else { return id }
        switch family {
        case .qwen36: return "qwen3.6-35b-a3b"
        case .qwen36MTP: return "qwen3.6-35b-a3b-mtp"
        case .qwen38flash: return "qwen3.8-flash-next"
        case .qwen38flashMTP: return "qwen3.8-flash-next-mtp"
        case .qwen4flash: return "qwen4-omni-120b-think"
        case .qwen4flashMTP: return "qwen4-omni-120b-think-mtp"
        // Reached only for an install whose manifest carries no model id; the
        // dense converter always writes one.
        case .qwen35Dense: return "qwen3.5-dense"
        case .laguna: return "laguna-s-2-1"
        }
    }

    package static func apiModelID(
        manifestModelID: String, family: ModelFamily, weightBits: Int
    ) -> String {
        base(manifestModelID: manifestModelID, family: family) + "_\(weightBits)-Bit"
    }

    /// Human names for the installs this project ships. A manifest carries an
    /// id, not a name; an install outside this table is listed by its id.
    ///
    /// This lives with the 6.x adapter because the adapter is what still serves
    /// these families; a family that moves to its own plugin takes its name
    /// with it (`Ornith35B4BitPlugin` and `DenseQwen35Plugin` already do).
    package static let displayNames: [String: String] = [
        "qwen3.6-35b-a3b": "Qwen 3.6 35B-A3B",
        "ornith-1.5-35b-a3b": "Ornith 1.5 35B-A3B",
        "ornith-1.5-35b-a3b-bf16dense": "Ornith 1.5 35B-A3B (bf16 dense)",
        "qwen-agentworld": "Qwen AgentWorld 35B-A3B",
        "kat-coder-v2.5": "KAT-Coder-V2.5-Dev 35B-A3B",
        "qwen3.8-flash-next": "Qwen 3.8 Flash Next 125B-A6B",
        "qwen4-omni-120b-think": "Qwen 4.0 Flash (Qwen4-Omni-120B-Think)",
        // The dense CPU models. Their snapshots carried these names in
        // `config.json -> display_name`, which only the snapshot probe reads;
        // the install probe has a manifest, and a manifest carries an id, not
        // a name. Without these entries repacking them would silently rename
        // them from "Qwen 3.5 9B" to "qwen3.5-9b" in the app and the server's
        // /v1/models listing.
        "qwen3.5-2b": "Qwen 3.5 2B",
        "qwen3.5-4b": "Qwen 3.5 4B",
        "qwen3.5-9b": "Qwen 3.5 9B",
    ]

    /// Why no front end can serve an installed family, or nil when one can.
    ///
    /// A memory fit is not a capability claim: Laguna's decode probe exists but
    /// its serving graph does not, and a draft head has no tokenizer or layers
    /// of its own. The catalog skips these rather than offering a model that
    /// stops at the loader's refusal.
    package static func unsupportedReason(_ family: ModelFamily) -> String? {
        switch family {
        case .qwen36MTP, .qwen38flashMTP, .qwen4flashMTP:
            return "MTP draft head, served only beside its target"
        case .laguna:
            return "Laguna-S-2.1's decode probe is implemented; the serving "
                + "graph is not, so the install validates but is not served "
                + "(docs/plan-laguna-s-2.1.md)"
        case .qwen36, .qwen38flash, .qwen4flash, .qwen35Dense:
            return nil
        }
    }

    /// Why this CLI cannot launch a family the server still serves. Today no
    /// family is server-only, so it is the server's own answer.
    package static func cliUnsupportedReason(_ family: ModelFamily) -> String? {
        unsupportedReason(family)
    }

    /// The host-facing identity of a 6.x install, read from its manifest alone.
    package static func hostIdentity(directory: URL) throws -> ModelIdentity {
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        let base = base(manifestModelID: identity.modelID, family: identity.family)
        return ModelIdentity(
            modelID: identity.modelID,
            catalogID: apiModelID(
                manifestModelID: identity.modelID, family: identity.family,
                weightBits: identity.weightBits),
            displayName: displayNames[base] ?? base,
            family: identity.family.rawValue,
            weightBits: identity.weightBits,
            sampling: identity.family.engineSamplingProfile(identity: identity),
            reasoningControl: identity.family.engineReasoningControl,
            engines: identity.family == .qwen35Dense ? [.gpu, .cpu] : [.gpu],
            contextLimit: nil,
            unsupportedReason: unsupportedReason(identity.family),
            cliUnsupportedReason: cliUnsupportedReason(identity.family))
    }
}
