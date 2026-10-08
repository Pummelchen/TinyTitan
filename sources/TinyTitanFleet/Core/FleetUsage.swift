/// The command line's own contract, kept where the tests can read it.
///
/// The executable target cannot be imported, so a help text that lives in it
/// cannot be pinned: nothing can assert that it names the options the parser
/// actually takes. It lives here for the same reason `FleetBrand` does.
public enum FleetUsage {
    /// What `--help` prints, and what a test asserts covers every option.
    public static let text = """
        \(FleetBrand.name) (\(FleetBrand.command)) — a control plane for a group of DSH hosts.

        usage: \(FleetBrand.command) [--peer HOST[:PORT]] [--key KEY] [--json] [--timeout SECONDS] <command>

          top                                              live dashboard: every member, what it holds, act on it
          list                                             every Mac in the group, with its workspaces and sessions
          prompt --session ID --text TEXT                  prompt one session, on the Mac that owns it
          prompt-all --text TEXT [--limit N] [--concurrency N]
                                                           prompt every active session in the group
          workspace create --on NAME --path DIR [--title TITLE]
                                                           register a folder as a workspace on one Mac
          session archive --session ID                     hide a session (reversible; history kept)
          workspace delete --workspace ID [--keep-sessions]
                                                           remove a workspace from the registry

        `top` draws the group live; a scanner on its own task polls the fleet every 30 s
        by default — half the plugin's discovery period, so its polling adds at most half
        a cycle of latency — while the window keeps drawing. It resizes with the window
        and never needs more than 44x6. `--once` prints a single frame instead
        (useful in a pipe, and with --width/--height for a fixed size). `--from FILE`
        renders an inventory JSON taken earlier — or from stdin with `-` — with no fleet
        running.

        --peer is the member the group is *read* from (default 127.0.0.1:3080); --on is
        the member an action is sent to. Every action then goes directly to the Mac that
        owns it — nothing is relayed through another instance. --json prints the raw
        answer. An action exits 0 only when the Mac that owns it confirms it: a refusal,
        or an answer that does not confirm, exits 1. --version prints the name and
        --help (or -h) prints this. --base-path is the route prefix a member serves
        the plugin under (default /dsh-lan); --interval SECONDS sets how often `top`
        rescans the fleet (default 30 s, floored at 2 s). --timeout, --limit,
        --concurrency, --width and --height are a duration, a count or a size, so
        they take a positive number: a cap of zero would mean every session, a frame
        of zero columns prints nothing, and a timeout of zero is no timeout at all,
        so such a value is refused rather than obeyed differently.
        Keys resolve in this order:
        --key, DSH_LAN_KEY, DSH_LAN_TOKEN, the plugin's shipped default. Prefer the
        environment forms: --key puts the key in argv, where every other local
        account can read it with ps.
        """
}
