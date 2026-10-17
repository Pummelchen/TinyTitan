import Darwin
import Foundation

/// Opt-in experiment: page-cache advice for the resident weight mapping.
///
/// `TINYTITAN_RESIDENT_ADVICE` selects the hint:
/// - `random` (default) — the historical `POSIX_MADV_RANDOM`, which disables
///   readahead and may be the wrong hint for a mapping that is read mostly
///   sequentially;
/// - `willneed` — `MADV_WILLNEED`, prefault the range;
/// - `sequential` — `POSIX_MADV_SEQUENTIAL`;
/// - `none` — leave the mapping alone.
///
/// Unknown values fall back to `random`, so a typo cannot change behaviour
/// silently in a way the operator cannot see from the env var they set.
enum ResidentAdvice: String, CaseIterable {
    case random
    case willneed
    case sequential
    case none
}

enum MemoryAdvice {
    static func residentAdvice(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ResidentAdvice {
        guard
            let raw = environment["TINYTITAN_RESIDENT_ADVICE"]?.lowercased(),
            let advice = ResidentAdvice(rawValue: raw)
        else {
            return .random
        }
        return advice
    }

    static func apply(
        _ advice: ResidentAdvice,
        base: UnsafeMutableRawPointer,
        length: Int
    ) {
        guard length > 0 else { return }
        switch advice {
        case .random:
            _ = posix_madvise(base, length, POSIX_MADV_RANDOM)
        case .willneed:
            _ = madvise(base, length, MADV_WILLNEED)
        case .sequential:
            _ = posix_madvise(base, length, POSIX_MADV_SEQUENTIAL)
        case .none:
            break
        }
    }
}
