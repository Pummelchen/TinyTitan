import Foundation

/// The load-time `ram profile` and `ram target` lines.
///
/// They are the operator's only record of what the engine allocated, so every
/// number describes the slot count actually in effect. The derivation is named
/// as `derived_slots=` only when an override replaced it: a line that printed
/// the derived rung while a different one was allocated read as evidence of a
/// configuration that never ran.
package enum ExpertCacheDiagnostics {
    private static func gib(_ bytes: Double) -> String {
        String(format: "%.2fG", bytes / 1_073_741_824)
    }

    private static func slotClause(appliedSlots: Int, derivedSlots: Int) -> String {
        appliedSlots == derivedSlots
            ? "slots=\(appliedSlots)"
            : "slots=\(appliedSlots) derived_slots=\(derivedSlots)"
    }

    /// The no-flag case: the budget came from the model's profile, so the line
    /// says it is a cache budget rather than a ceiling on the whole process.
    package static func profileLine(
        appliedSlots: Int,
        derivedSlots: Int,
        cacheBytesPerSlot: Double,
        residentFloorBytes: Int
    ) -> String {
        let cache = Double(appliedSlots) * cacheBytesPerSlot
        let floor = Double(residentFloorBytes)
        return """
            TinyTitan ram profile cache=\(gib(cache)) \
            \(slotClause(appliedSlots: appliedSlots, derivedSlots: derivedSlots)) \
            resident_floor=\(gib(floor)) estimate=\(gib(floor + cache)) \
            (cache budget, not a process target)
            """
    }

    /// The `--ram` case: the number names the whole process, so a footprint
    /// above it gets a second line saying the cache is already at its smallest.
    package static func targetLines(
        targetBytes: Int,
        appliedSlots: Int,
        derivedSlots: Int,
        cacheBytesPerSlot: Double,
        residentFloorBytes: Int
    ) -> [String] {
        let cache = Double(appliedSlots) * cacheBytesPerSlot
        let floor = Double(residentFloorBytes)
        let estimate = floor + cache
        let line = """
            TinyTitan ram target=\(gib(Double(targetBytes))) cache=\(gib(cache)) \
            \(slotClause(appliedSlots: appliedSlots, derivedSlots: derivedSlots)) \
            resident_floor=\(gib(floor)) estimate=\(gib(estimate))
            """
        guard Double(targetBytes) < estimate else { return [line] }
        return [
            line,
            """
            TinyTitan ram warning: \(gib(Double(targetBytes))) is below this install's \
            \(gib(estimate)) floor (\(gib(floor)) resident + the \(appliedSlots)-slot \
            minimum cache); the cache is already at its smallest.
            """,
        ]
    }
}
