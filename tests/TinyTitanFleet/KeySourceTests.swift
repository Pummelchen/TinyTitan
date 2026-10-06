import Testing
@testable import TinyTitanFleetCore

// AUD-170: the shipped default groups a fleet rather than protects it, so the
// only spelling that needs care is the one an operator chooses *because* they
// want protection. `--key` is that spelling, and it is also the only one that
// puts the value in argv, where every other local account reads it with `ps`.
//
// The order is pinned alongside the warning because the two are one contract:
// a future edit that reordered resolution would change which key a machine
// presents without touching any of the assertions that matter here.
@Test func theCommandLineKeyResolvesFirstAndSaysWhereItIsVisible() {
    let resolved = FleetGroupKey.resolve(
        option: "chosen-by-me",
        environment: ["DSH_LAN_KEY": "from-environment", "DSH_LAN_TOKEN": "from-token"])
    #expect(resolved.key == "chosen-by-me", "--key still wins: the order is unchanged")
    let warning = resolved.warning ?? ""
    #expect(warning.contains("argv") && warning.contains("ps"), "and it names the leak")
    #expect(warning.contains("DSH_LAN_KEY"), "and points at the spelling that does not leak")
}

@Test func theEnvironmentFormsResolveWithoutAdvice() {
    #expect(
        FleetGroupKey.resolve(option: nil, environment: ["DSH_LAN_KEY": "from-environment"])
            == FleetGroupKey.Resolution(key: "from-environment", warning: nil))
    #expect(
        FleetGroupKey.resolve(option: nil, environment: ["DSH_LAN_TOKEN": "from-token"]).key
            == "from-token", "DSH_LAN_TOKEN is the second name for the same thing")
    #expect(
        FleetGroupKey.resolve(option: nil, environment: [:]).key == FleetGroupKey.shippedDefault,
        "and nothing set at all is the public default, which needs no warning")
}
