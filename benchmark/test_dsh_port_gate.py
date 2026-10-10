"""The port bar `tools/dsh_local.sh` claims and does not keep.

`port` answers what the browser UI would bind, and `web` binds it. The number
comes from `TINYTITAN_DSH_PORT` or `--dsh-port`, and neither path checks the
shape it was given: `resolve_port` puts the value straight into a `for (( ))`
arithmetic loop, so the string is evaluated as bash arithmetic rather than read
as a port. Measured on the current tree, model-free, against the real script:

    TINYTITAN_DSH_PORT=-1   ->  exit 0, prints -1        (accepted silently)
    TINYTITAN_DSH_PORT=0    ->  exit 0, prints 1         (invented a reserved port)
    TINYTITAN_DSH_PORT=" "  ->  exit 0, prints 1         (the same, from whitespace)
    TINYTITAN_DSH_PORT=abc  ->  exit 1, "abc: unbound variable"
    TINYTITAN_DSH_PORT=7788a->  exit 1, "value too great for base"
    TINYTITAN_DSH_PORT=99999->  exit 1, "ports 99999-100099 are all in use"

The last one is the loudest lie: the loop's guard is `candidate <= 65535`, so an
out-of-range base never enters the loop and never asks a single port, yet the
message reports one hundred and one ports occupied. The three silent ones are
the expensive kind: port 1 and port -1 are not bindable by an unprivileged
process, so the harness this script launches dies on its own argument while the
script has already printed a port as if it had chosen one.

`tools/server_launcher.sh` refuses all of these with `unknown port: <value> (a
number 1-65535)` before it uses the value. This suite pins the same bar here.

    cd benchmark && python3 -m unittest test_dsh_port_gate -v
"""

from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DSH_LOCAL = ROOT / "tools/dsh_local.sh"


def run_port(env_port: str | None, flag_port: str | None = None) -> subprocess.CompletedProcess:
    """Ask the real script what port it would use, with nothing started.

    `port` only reads the port table (`lsof`, `ps`) and prints a number, so no
    harness, node process or network service is involved.
    """
    env = dict(os.environ)
    env.pop("TINYTITAN_DSH_PORT", None)
    if env_port is not None:
        env["TINYTITAN_DSH_PORT"] = env_port
    argv = ["bash", str(DSH_LOCAL), "port"]
    if flag_port is not None:
        argv += ["--dsh-port", flag_port]
    return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120, check=False)


class TheVariablePath(unittest.TestCase):
    """A port named in the environment is checked before it is used."""

    def test_a_non_numeric_port_is_refused_and_the_value_is_named(self) -> None:
        result = run_port("7788a")
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertEqual(result.stdout.strip(), "", f"a bad port was printed as a choice: {output}")
        self.assertIn("7788a", output)
        self.assertIn("1-65535", output)

    def test_a_word_is_refused_rather_than_reaching_bash_arithmetic(self) -> None:
        result = run_port("abc")
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("abc", output)
        # The script runs under `set -u`, so an unset name is what a word
        # evaluates to: a user reading "unbound variable" has nothing to act on.
        self.assertNotIn("unbound variable", output)

    def test_zero_is_not_resolved_into_a_reserved_port(self) -> None:
        result = run_port("0")
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertNotEqual(result.stdout.strip(), "1", f"0 became port 1: {output}")
        self.assertIn("0", output)

    def test_a_negative_port_is_not_accepted_silently(self) -> None:
        result = run_port("-1")
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertNotEqual(result.stdout.strip(), "-1", f"-1 came back as a choice: {output}")
        self.assertIn("1-65535", output)

    def test_whitespace_is_not_a_port(self) -> None:
        result = run_port(" ")
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertNotEqual(result.stdout.strip(), "1", f"whitespace became port 1: {output}")

    def test_a_number_too_large_for_the_machine_does_not_wrap_into_the_space(self) -> None:
        # Bash arithmetic is 64-bit and wraps silently, so 2**64+1 evaluates to
        # 1. A digit-count bound is what keeps such a value out of the range
        # comparison rather than a range comparison that happens to catch it.
        result = run_port("18446744073709551617")
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertEqual(result.stdout.strip(), "", f"a wrapped port was printed: {output}")
        self.assertIn("1-65535", output)

    def test_a_port_above_the_space_is_not_reported_as_a_range_in_use(self) -> None:
        # Nothing is scanned: the loop's own guard rejects the base on entry.
        result = run_port("99999")
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertNotIn("all in use", output)
        self.assertIn("99999", output)
        self.assertIn("1-65535", output)


class TheFlagPath(unittest.TestCase):
    """`--dsh-port` promises a number and has to mean it."""

    def test_the_flag_is_held_to_the_same_bar_as_the_variable(self) -> None:
        result = run_port(None, "7788a")
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertEqual(result.stdout.strip(), "", f"a bad port was printed: {output}")
        self.assertIn("7788a", output)
        self.assertIn("1-65535", output)

    def test_the_flag_does_not_report_an_unscanned_range_as_in_use(self) -> None:
        result = run_port(None, "70000")
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertNotIn("all in use", output)
        self.assertIn("70000", output)


class TheWorkingPathStaysWorking(unittest.TestCase):
    """The guard rejects shapes, not ports."""

    def test_the_default_is_still_a_port_in_the_space(self) -> None:
        result = run_port(None)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertNotIn("unknown port", output)
        chosen = result.stdout.strip()
        self.assertTrue(chosen.isdigit(), chosen)
        self.assertTrue(1 <= int(chosen) <= 65535, chosen)

    def test_a_valid_port_is_never_rejected_as_a_shape(self) -> None:
        # 65535 is the last legal port; whether it is free is the machine's
        # business, but a legal number must not read as an illegal shape.
        result = run_port("65535")
        output = result.stdout + result.stderr
        self.assertNotIn("unknown port", output)
        if result.returncode != 0:
            self.assertIn("all in use", output, output)

    def test_a_taken_port_still_walks_up_to_the_first_free_one(self) -> None:
        held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        try:
            result = run_port(str(port))
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            chosen = result.stdout.strip()
            self.assertTrue(chosen.isdigit(), chosen)
            self.assertNotEqual(int(chosen), port, f"the walk skipped a listener: {output}")
            self.assertIn("is taken; using", output)
        finally:
            held.close()


if __name__ == "__main__":
    unittest.main()
