// Interior-checkpoint capture planning for one generation.
//
// Split out of ServerModelSession+Generation.swift (2026-10-22) under the
// 500-line-per-file rule. The generation orchestrator asks for a plan; this
// file owns the boundary selection, the snapshot capture and the detached
// store writes.
import Foundation
import TinyTitan

extension ServerModelSession {
    /// What one generation should capture, and the callback that captures it.
    struct InteriorCheckpointCapture: Sendable {
        let positions: [Int]
        let onCheckpoint: (@Sendable (Int) -> Void)?

        static let none = InteriorCheckpointCapture(
            positions: [], onCheckpoint: nil)
    }

    /// Plan the interior checkpoints for this prompt.
    ///
    /// The positions are message boundaries on the ids that were actually
    /// rendered. Capture is skipped when a restore could never succeed (a QSA
    /// or PLE family), when the tier is off, and when another producer (MTP or
    /// a batched head) owns the prefill.
    func interiorCheckpointCapture(
        promptIDs: [Int32],
        messages: [GFTokenizer.Message],
        imStartID: Int32,
        enabled: Bool
    ) -> InteriorCheckpointCapture {
        guard enabled,
            promptCacheMode == .multiPrefix,
            let checkpointStore = promptCheckpointStore,
            runner.supportsInferenceStateRestore
        else { return .none }
        let boundaries = PromptCheckpointPlanner.boundaries(
            promptIDs: promptIDs,
            messages: messages,
            imStartID: imStartID,
            gridTokens: Self.checkpointGridTokens,
            includeMessageBoundaries: !Self.checkpointGridOnly)
        guard !boundaries.isEmpty else { return .none }
        let kinds = Dictionary(
            uniqueKeysWithValues: boundaries.map { ($0.position, $0.kind) })
        let captureRunner = runner
        let captureDomain = promptCacheDomain
        let capturePromptIDs = promptIDs
        let conversation = PromptCheckpointPlanner.conversationKey(
            domain: captureDomain, messages: messages)
        let onCheckpoint: @Sendable (Int) -> Void = { position in
            do {
                let snapshot = try captureRunner.captureInferenceState(
                    maximumBytes: PromptCheckpointStore.maximumCaptureBytes)
                guard snapshot.descriptor.position == position else {
                    ServerLog.diagnostic(
                        "TinyTitan prompt_checkpoint capture_mismatch "
                            + "expected=\(position) "
                            + "actual=\(snapshot.descriptor.position)")
                    return
                }
                let auxiliary = try captureRunner.captureAuxiliaryState()
                let prefix = Array(capturePromptIDs.prefix(position))
                let key = PromptCheckpointKey.make(
                    domain: captureDomain, prefixTokenIDs: prefix)
                let metadata = PromptCheckpointMetadata(
                    key: key,
                    domain: captureDomain,
                    kind: kinds[position] ?? .lastTurnStart,
                    conversationKey: conversation,
                    prefixTokenIDs: prefix,
                    descriptor: snapshot.descriptor)
                Task.detached(priority: .utility) {
                    guard !(await checkpointStore.contains(key)) else {
                        return
                    }
                    do {
                        try await checkpointStore.save(
                            metadata, snapshot: snapshot, auxiliary: auxiliary)
                        ServerLog.diagnostic(
                            "TinyTitan prompt_checkpoint stored "
                                + "kind=\(metadata.kind.rawValue) "
                                + "tokens=\(position) "
                                + "state_bytes=\(snapshot.payload.count) "
                                + "aux_bytes=\(auxiliary?.payload.count ?? 0)")
                    } catch {
                        ServerLog.diagnostic(
                            "TinyTitan prompt_checkpoint save_failed "
                                + "tokens=\(position) error=\(error)")
                    }
                }
            } catch {
                // A capture failure must not fail the request: the prefill
                // already succeeded and a full prefill next time is correct,
                // just slower.
                ServerLog.diagnostic(
                    "TinyTitan prompt_checkpoint capture_failed "
                        + "tokens=\(position) error=\(error)")
            }
        }
        return InteriorCheckpointCapture(
            positions: boundaries.map(\.position),
            onCheckpoint: onCheckpoint)
    }

    /// The optional token-grid interval (`TINYTITAN_CHECKPOINT_GRID`).
    ///
    /// Nil in the shipped configuration. T24 uses it to price a 16K/32K grid
    /// against the message-boundary points; T20's design argues a grid should
    /// not be the default, and the benchmark decides whether it earns a switch
    /// rather than an assumption.
    static var checkpointGridOnly: Bool {
        ProcessInfo.processInfo.environment[
            "TINYTITAN_CHECKPOINT_GRID_ONLY"] == "1"
    }

    static var checkpointGridTokens: Int? {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "TINYTITAN_CHECKPOINT_GRID"],
            let value = Int(raw), value >= 1_024
        else { return nil }
        return value
    }
}
