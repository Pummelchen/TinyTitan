import Foundation

/// How the host names an install before it loads a model.
///
/// `EngineModelMetadata` is the session's identity: it is read by the loader and
/// travels with the loaded model. This is the *host-facing* identity a model
/// list, a log line and a refusal use, and it is deliberately separate because a
/// catalog id carries the width suffix (`..._4-Bit`) while the session's id does
/// not. A plugin (or the 6.x adapter while it still serves a family) answers it
/// from the install, so the host never derives a family, a name or a sampling row
/// from the directory.
public struct ModelIdentity: Sendable, Equatable {
    /// The install's own model id, as the manifest or snapshot declares it.
    public let modelID: String
    /// The API/catalog id, width suffix included where the model list adds one
    /// (`..._4-Bit`). Equal to `modelID` for an install whose id already says
    /// everything.
    public let catalogID: String
    /// The human name a model list shows.
    public let displayName: String
    /// The family label the install's manifest declares.
    public let family: String
    /// Routed-expert width for a GPU install, the base affine width for a
    /// snapshot.
    public let weightBits: Int
    /// The sampling this model uses for anything a request omits.
    public let sampling: SamplingProfile
    /// The thinking and effort levels its template implements.
    public let reasoningControl: ReasoningControl
    /// nil when the host can serve it. A family whose loader refuses it (a
    /// draft head, or a model with no serving graph yet) says so here rather
    /// than making every front end keep its own family table.
    public let unsupportedReason: String?

    public init(
        modelID: String,
        catalogID: String,
        displayName: String,
        family: String,
        weightBits: Int,
        sampling: SamplingProfile,
        reasoningControl: ReasoningControl,
        unsupportedReason: String? = nil
    ) {
        self.modelID = modelID
        self.catalogID = catalogID
        self.displayName = displayName
        self.family = family
        self.weightBits = weightBits
        self.sampling = sampling
        self.reasoningControl = reasoningControl
        self.unsupportedReason = unsupportedReason
    }
}
