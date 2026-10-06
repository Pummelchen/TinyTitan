/// How the CLI decides which group key to present, and what it must say about it.
///
/// Named for what it is because `FleetKey` is already taken: that enum is a key on
/// the keyboard, in `FleetDashboard.swift`.
///
/// The order is unchanged from the flag's first version: `--key`, then
/// `DSH_LAN_KEY`, then `DSH_LAN_TOKEN`, then the plugin's shipped default
/// (`plugins/dsh-lan-manager/src/config.js:51` owns that string, and the two
/// literals here have to move with it). Only the spelling is a security choice:
/// `--key` puts the secret in this process's argv, and macOS shows every
/// account's argv to every other account through `ps`, while a process's
/// environment belongs to its own user. The shipped default is public by design
/// -- it groups a fleet rather than protects it, which is what `router.js` says
/// to a peer that presents it -- so the warning exists for the case where the
/// operator replaced it precisely to keep other people out.
public enum FleetGroupKey {
    /// The key this fleet falls back to when nobody chose one.
    public static let shippedDefault = "tinytitan-lan"

    public struct Resolution: Sendable, Equatable {
        public let key: String

        /// What to put on stderr, or `nil` when the spelling needs no advice.
        public let warning: String?

        public init(key: String, warning: String?) {
            self.key = key
            self.warning = warning
        }
    }

    public static func resolve(
        option: String?,
        environment: [String: String]
    ) -> Resolution {
        if let option {
            return Resolution(
                key: option,
                warning: """
                    --key puts the group key in this process's argv, which every other \
                    local account can read with ps; set DSH_LAN_KEY instead.
                    """
            )
        }
        if let fromEnvironment = environment["DSH_LAN_KEY"] ?? environment["DSH_LAN_TOKEN"] {
            return Resolution(key: fromEnvironment, warning: nil)
        }
        return Resolution(key: shippedDefault, warning: nil)
    }
}
