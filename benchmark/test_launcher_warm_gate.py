"""The launcher's two warm-up numbers are sizes, and both are read after the load.

`tools/server_launcher.sh` reads `TINYTITAN_WARM_TOKENS` at :1826 and puts it
through an arithmetic expansion at :1837 (`warm_repeat=$(( warm_tokens / 11 ))`),
and it reads `TINYTITAN_WARM_TIMEOUT` at :1847 and hands it to
`curl --max-time`. The `:NNNN` references are the file as it stood before this fix
landed, since that is what the table was measured on: the block, taken verbatim from
that file and run under `/bin/bash` 3.2 with a `curl` that records its arguments:

    TINYTITAN_WARM_TOKENS=abc      -> exit 1, `abc: unbound variable`, no request
    TINYTITAN_WARM_TOKENS=08000    -> exit 1, `value too great for base`, no request
    TINYTITAN_WARM_TOKENS=4000.5   -> exit 1, syntax error, no request
    TINYTITAN_WARM_TOKENS=-5 / 0   -> exit 0, one request, "~ -5 tokens" printed
    TINYTITAN_WARM_TIMEOUT=abc     -> handed to curl as written
    TINYTITAN_WARM_TIMEOUT=0       -> handed to curl as written (curl's no-timeout)

Three things are wrong in that table. The first is the class: a word is read as
an arithmetic expression, so it dies with a bash message that names neither the
variable nor the fix, and it dies *after* the model has loaded and *before* the
client opens — the launcher's own `--port` check at :1213-1218 refuses exactly
this shape, and refuses it before anything expensive happens. The second is the
cause: `curl` documents `--max-time 0` as "continue forever", so the one value
that removes the bound is accepted, and any curl failure at :1847 is reported by
:1852 as `The warm-up did not finish`, which is true of a word the same way it is
true of a slow model. The third is the message: `0` and `-5` print "~ 0 tokens"
and then send the clamped one-repeat prompt, so the size the operator asked for
is not the size the line reports.

The repair puts both checks where the port's already is: no word, leading zero or
zero reaches the arithmetic or curl. These tests run the port block and the warm
block as one script, so a guard that is present but wrongly scoped — dropped from
the web-only run, or left on a run that switched the warm-up off — shows up as a
refusal that fired when it should not have, or one that never fired. The mutation
sweep over the guard killed nine of eleven shapes. The two survivors are the warm
block and the curl call re-reading the environment instead of using the checked
value, and the sweep is what proved those equivalent: this guard refuses rather than
rewrites, so every value it accepts is the value a second read would return.

    cd benchmark && python3 -m unittest test_launcher_warm_gate -v
"""

from __future__ import annotations

import pathlib
import socket
import subprocess
import tempfile
import threading
import time
import unittest

import launcher_fixture
from test_launcher_port import run_launcher

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/server_launcher.sh"
BASH = "/bin/bash"
CURL = "/usr/bin/curl"

PORT_START = 'if [[ -n "$PORT_ARG" ]]; then'
PORT_END = "# The context, KV and YaRN flags reach the GPU runtime only."
WARM_START = 'if [[ "${TINYTITAN_WARM:-1}" != "0" ]]; then'
WARM_END = '  echo "Opening DeepSeek Harness in your default browser..."'

PRELUDE = [
    "set -euo pipefail",
    'PORT_ARG="8080"',
    "WEB=1",
    "INTERACTIVE=0",
    'TINYTITAN_DEFAULT_PORT="8080"',
    'MODEL="qwen36-4bit"',
    "warn_red() { printf 'WARN: %s\\n' \"$1\" >&2; }",
    'curl() { for a in "$@"; do printf \'%s\\n\' "$a" >> "$TT_ARGV"; done; '
    'return "${CURL_RC:-0}"; }',
    'TT_ARGV="${TT_ARGV:?}"',
]


def warm_script() -> str:
    """The launcher's port block and warm block, verbatim, joined as they run."""
    text = LAUNCHER.read_text(encoding="utf-8")
    port_start = text.index(PORT_START)
    port_end = text.index(PORT_END, port_start)
    # Searched from after the preflight so the guard's own copy of the skip
    # condition cannot be mistaken for the warm-up block itself.
    warm_start = text.index(WARM_START, port_end)
    warm_end = text.index(WARM_END, warm_start)
    return "\n".join(PRELUDE) + "\n" + text[port_start:port_end] + text[warm_start:warm_end]


def run_warm(env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run the extracted script with the operator's environment, and record curl's argv."""
    with tempfile.TemporaryDirectory() as work:
        argv_path = pathlib.Path(work) / "argv.txt"
        argv_path.write_text("", encoding="utf-8")
        script_path = pathlib.Path(work) / "warm.sh"
        script_path.write_text(warm_script(), encoding="utf-8")
        full_env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(pathlib.Path.home()),
            "LANG": "C",
            "TT_ARGV": str(argv_path),
        }
        for name in ("TINYTITAN_WARM", "TINYTITAN_WARM_TOKENS", "TINYTITAN_WARM_TIMEOUT"):
            if env and name in env:
                full_env[name] = env[name]
        result = subprocess.run(
            [BASH, str(script_path)],
            capture_output=True,
            text=True,
            env=full_env,
            timeout=60,
            check=False,
        )
        result.argv = [line for line in argv_path.read_text(encoding="utf-8").split("\n") if line]
    return result


def max_time_arg(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--max-time":
            return argv[index + 1] if index + 1 < len(argv) else None
    return None


def body_arg(argv: list[str]) -> str:
    for index, token in enumerate(argv):
        if token == "-d":
            return argv[index + 1] if index + 1 < len(argv) else ""
    return ""


def warm_sentence() -> str:
    """The sentence the launcher repeats, read from the launcher rather than restated."""
    text = LAUNCHER.read_text(encoding="utf-8")
    marker = 'warm_sentence="'
    start = text.index(marker) + len(marker)
    return text[start : text.index('"', start)]


class TheWarmNumbersAreCheckedBeforeTheLoad(unittest.TestCase):
    """A shape that is not a size is refused at the read, in front of the port's style."""

    def test_a_word_for_the_size_is_refused_before_any_request(self) -> None:
        result = run_warm({"TINYTITAN_WARM_TOKENS": "abc"})
        self.assertEqual(result.returncode, 2, f"stdout: {result.stdout}\nstderr: {result.stderr}")
        self.assertEqual(result.argv, [], f"a request was made: {result.argv}")
        self.assertIn("TINYTITAN_WARM_TOKENS", result.stderr)
        self.assertIn("abc", result.stderr)

    def test_a_leading_zero_size_is_refused_not_read_as_octal(self) -> None:
        result = run_warm({"TINYTITAN_WARM_TOKENS": "08000"})
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("08000", result.stderr)
        self.assertEqual(result.argv, [])

    def test_a_fractional_size_is_refused_at_the_read(self) -> None:
        result = run_warm({"TINYTITAN_WARM_TOKENS": "4000.5"})
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.argv, [])

    def test_a_negative_size_is_refused_at_the_read(self) -> None:
        result = run_warm({"TINYTITAN_WARM_TOKENS": "-5"})
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.argv, [])

    def test_a_zero_size_is_refused_rather_than_clamped(self) -> None:
        # 0 reaches the clamp at :1838 and becomes one repeat, after the line has
        # printed "~ 0 tokens". A prompt of no tokens is not a thing the operator
        # can ask for, so the read says so instead of the print lying.
        result = run_warm({"TINYTITAN_WARM_TOKENS": "0"})
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.argv, [])
        self.assertIn("warms nothing", result.stderr)
        # Without the message, a 0 refused by the leading-zero arm would read as
        # "08000: no leading zero" — a refusal that names the wrong problem.
        self.assertNotIn("leading zero", result.stderr)

    def test_the_launcher_does_not_die_with_a_bash_internal_message(self) -> None:
        result = run_warm({"TINYTITAN_WARM_TOKENS": "abc"})
        self.assertNotIn("unbound variable", result.stdout + result.stderr)
        self.assertNotIn("value too great for base", result.stdout + result.stderr)

    def test_a_zero_timeout_is_refused_because_curl_reads_it_as_no_limit(self) -> None:
        result = run_warm({"TINYTITAN_WARM_TIMEOUT": "0"})
        self.assertEqual(result.returncode, 2, f"stdout: {result.stdout}\n{result.stderr}")
        self.assertEqual(result.argv, [], f"an unbounded request was made: {result.argv}")
        self.assertIn("TINYTITAN_WARM_TIMEOUT", result.stderr)
        # The shape deserves its own reason: a general "not a number" message would
        # let a repair through that refused 0 for the wrong cause, and 0 is the one
        # value that is a valid number and still removes the bound.
        self.assertIn("forever", result.stderr)

    def test_a_word_for_the_timeout_is_not_reported_as_a_slow_warmup(self) -> None:
        result = run_warm({"TINYTITAN_WARM_TIMEOUT": "abc"})
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn("did not finish", result.stdout + result.stderr)
        self.assertEqual(result.argv, [])

    def test_a_negative_timeout_is_refused_at_the_read(self) -> None:
        result = run_warm({"TINYTITAN_WARM_TIMEOUT": "-1"})
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.argv, [])


class TheDocumentedDefaultsAndTheSkipStay(unittest.TestCase):
    """The guard must not move the warm-up, change its size, or make it unskippable."""

    def test_the_defaults_still_warm_at_the_documented_sizes(self) -> None:
        result = run_warm()
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        self.assertEqual(result.argv.count("-s"), 1, f"expected one request: {result.argv}")
        self.assertEqual(max_time_arg(result.argv), "900")
        self.assertIn("~4000 tokens", result.stdout)

    def test_a_blank_assignment_means_the_default_not_an_empty_number(self) -> None:
        result = run_warm({"TINYTITAN_WARM_TOKENS": "", "TINYTITAN_WARM_TIMEOUT": ""})
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        self.assertEqual(max_time_arg(result.argv), "900")
        self.assertIn("~4000 tokens", result.stdout)

    def test_the_checked_value_is_the_one_the_warmup_sends(self) -> None:
        # A guard that validates and then lets the block re-read the environment is
        # a guard that can be bypassed, so the raised size has to show up in the
        # request body rather than only in the printed line.
        small = run_warm({"TINYTITAN_WARM_TOKENS": "330"})
        large = run_warm({"TINYTITAN_WARM_TOKENS": "1320"})
        self.assertEqual(small.returncode, 0, small.stderr)
        self.assertEqual(large.returncode, 0, large.stderr)
        self.assertIn("~330 tokens", small.stdout)
        self.assertIn("~1320 tokens", large.stdout)
        self.assertEqual(body_arg(small.argv).count(warm_sentence()), 30)
        self.assertEqual(body_arg(large.argv).count(warm_sentence()), 120)

    def test_the_timeout_reaches_the_request_as_written(self) -> None:
        result = run_warm({"TINYTITAN_WARM_TIMEOUT": "45"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(max_time_arg(result.argv), "45")

    def test_the_documented_skip_still_sends_nothing(self) -> None:
        result = run_warm({"TINYTITAN_WARM": "0"})
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        self.assertEqual(result.argv, [])


class CurlsOwnSemanticsAreWhyTheReadIsGuarded(unittest.TestCase):
    """Real curl, no stub: the two values the guard refuses are not bounds."""

    def test_zero_is_a_switch_for_curl_not_a_timeout(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            host, port = listener.getsockname()
            stalled = threading.Thread(target=self._hold, args=(listener,), daemon=True)
            stalled.start()
            process = subprocess.Popen(
                [CURL, "-s", "--max-time", "0", f"http://{host}:{port}/v1/models"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                code = process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return
            self.fail(f"curl returned {code} with --max-time 0 against a stalled listener")

    @staticmethod
    def _hold(listener: socket.socket) -> None:
        """Accept the request and answer nothing, so only a timeout can end it."""
        try:
            connection, _ = listener.accept()
        except OSError:
            return
        time.sleep(8)
        connection.close()

    def test_a_word_is_curls_error_not_the_servers(self) -> None:
        result = subprocess.run(
            [CURL, "-s", "--max-time", "abc", "http://127.0.0.1:1/v1/models"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("curl", (result.stdout + result.stderr).lower())


class TheGuardIsOnTheLiveLaunchPath(unittest.TestCase):
    """The real launcher, in a dry run against the fixture installs, refuses and scopes."""

    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def setUp(self) -> None:
        self.model = self.installs.first_gpu()
        self.web_base = ("--client", "server", "--web", "--model", self.model)

    def test_a_bad_size_is_refused_before_the_run_plans_anything(self) -> None:
        run = run_launcher(self.installs, *self.web_base, env={"TINYTITAN_WARM_TOKENS": "abc"})
        self.assertEqual(run.returncode, 2, f"{run.stdout}\n{run.stderr}")
        self.assertIn("TINYTITAN_WARM_TOKENS", run.stderr)
        self.assertNotIn("Port:", run.stdout)

    def test_a_run_that_does_not_warm_keeps_a_number_it_never_reads(self) -> None:
        # The warm-up lives in the --web hand-over, so a server-only run must not be
        # refused for the shape of a variable it never reads.
        run = run_launcher(
            self.installs,
            "--client",
            "server",
            "--model",
            self.model,
            env={"TINYTITAN_WARM_TOKENS": "abc"},
        )
        self.assertEqual(run.returncode, 0, f"{run.stdout}\n{run.stderr}")

    def test_the_documented_switch_off_still_skips_the_check(self) -> None:
        run = run_launcher(
            self.installs,
            *self.web_base,
            env={"TINYTITAN_WARM": "0", "TINYTITAN_WARM_TOKENS": "abc"},
        )
        self.assertEqual(run.returncode, 0, f"{run.stdout}\n{run.stderr}")

    def test_a_web_run_with_valid_numbers_still_plans_the_launch(self) -> None:
        run = run_launcher(
            self.installs,
            *self.web_base,
            env={"TINYTITAN_WARM_TOKENS": "6000", "TINYTITAN_WARM_TIMEOUT": "120"},
        )
        self.assertEqual(run.returncode, 0, f"{run.stdout}\n{run.stderr}")
        self.assertIn("Port:", run.stdout)


if __name__ == "__main__":
    unittest.main()
