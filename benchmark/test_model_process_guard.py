"""AUD-268: the model-process guard must refuse a pgrep that did not answer.

AGENTS.md makes "no process from `pgrep -fl 'TinyTitanServer|TinyTitanCLI|...'`"
a precondition of every model run, and this repository enforces it in eleven call
sites across nine files. Each of them can only act on the answer it is looking
for, so every other status falls through as "clear".

`pgrep` answers with three statuses -- 0 matched, 1 matched nothing, 2 an error --
and a missing `pgrep` adds a fourth (the shell's 127). Measured on this host
(`/usr/bin/pgrep -fl '('` -> status 2 with
`pgrep: Cannot compile regular expression `(' (parentheses not balanced)`, and
`PATH=/nonexistent-dir pgrep` -> 127), the three shell shapes then all proceed:

    # tools/golden-baseline.sh:130 -- the gate that runs before a capture
    if busy=$(pgrep -fl 'TinyTitanServer|...' 2>/dev/null); then

    # tools/verify_cpu_models.sh:33 -- and it asks a second time to print them
    if pgrep -fl "$GUARD" >/dev/null 2>&1; then

    # tools/install_tinytitan.sh:184 -- counts lines, so an error is "0 lines"
    if [[ "$(pgrep -fl '...' 2>/dev/null | wc -l | tr -d ' ')" != "0" ]]; then

Run against those three shapes with a stub pgrep answering 2
(/tmp/aud268b/shapes.sh): `GOLDEN: reached the run (busy=[])`, `INSTALLER:
proceeded without warning`, `CPU: reached the build`, exit 0. The consequence is
the one the guard exists to prevent -- a second model process beside a live one,
which AGENTS.md forbids and which makes every number the run produces noise.

The six python drivers discard the status in different shapes: four read only
stdout (`benchmark/tinytitan_knob_sweep.py:278`, `ane_steady_state_decode.py:208`,
`expert_cache_slots.py:162`, `tinytitan_gate0_profile.py:194`),
`benchmark/coder_cli_benchmark.py:250` does look at `returncode` but only to
confirm a match, so 2 still reads as clear, and
`benchmark/tinytitan_vs_competitors.py:553` is the same busy guard while its
`:257`/`:317` take an errored `pgrep -n` as "no pid" and report a competitor's RSS
as absent while it is running, which mislabels the comparison rather than racing
anything.

The fix is one owner per language, the way AUD-266 left the build-log gate:
`tools/model-guard.sh` answers the tri-state (0 found with the lines on stdout,
1 nothing found, 2 cannot ask with pgrep's status and message on stderr), and
`tinytitan_profile.pgrep_answer` gives python the same three verdicts. A caller
that did not get an answer refuses; "clear" is only what pgrep said.

`tools/install_tinytitan.sh` is run through `bash -c "$(curl ...)"`, where no
`tools/` directory exists beside it, so it carries the guard function as an
inline copy. `test_the_installers_copy_is_the_guards_function` pins that the two
copies are the same text; a drift between them is the bug this row is about,
caught in the other direction.

The shell scripts are driven with a stub `pgrep` on `PATH` and a stub `swift`, so
nothing here starts a model, runs a build, or touches a real process table.

    cd benchmark && python3 -m unittest test_model_process_guard -v
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
import unittest.mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
GUARD = ROOT / "tools" / "model-guard.sh"
INSTALLER = ROOT / "tools" / "install_tinytitan.sh"
PROFILE = ROOT / "benchmark" / "tinytitan_profile.py"
PATTERN = "TinyTitanServer|TinyTitanCLI"

# Both strings are what this host actually produced (module docstring).
BUSY_LINES = (
    "17514 /Users/example/TinyTitan/.build/release/TinyTitanServer"
    " --model models/qwen3.8-flash-next_4-Bit --port 8080\n"
)
PGREP_ERROR = "pgrep: Cannot compile regular expression `(' (parentheses not balanced)"

PGREP_STUB = """#!/bin/sh
if [ -n "${STUB_PGREP_ERR-}" ]; then printf '%s\\n' "$STUB_PGREP_ERR" >&2; fi
if [ -n "${STUB_PGREP_OUT-}" ]; then printf '%s' "$STUB_PGREP_OUT"; fi
exit "${STUB_PGREP_RC:-1}"
"""

SWIFT_STUB = """#!/bin/sh
echo "SWIFT-STUB-WAS-CALLED $*" >&2
exit 1
"""

SHELL_CALLERS = [
    "tools/golden-baseline.sh",
    "tools/verify_cpu_models.sh",
]
PYTHON_CALLERS = [
    "benchmark/tinytitan_knob_sweep.py",
    "benchmark/ane_steady_state_decode.py",
    "benchmark/coder_cli_benchmark.py",
    "benchmark/expert_cache_slots.py",
    "benchmark/tinytitan_gate0_profile.py",
    "benchmark/tinytitan_vs_competitors.py",
]


@contextlib.contextmanager
def stub_bin(*, pgrep: bool = True, swift: bool = True):
    """A PATH directory whose pgrep says exactly what the test asks."""
    with tempfile.TemporaryDirectory(prefix="aud268-") as work:
        bin_dir = pathlib.Path(work)
        if pgrep:
            tool = bin_dir / "pgrep"
            tool.write_text(PGREP_STUB, encoding="utf-8")
            tool.chmod(0o755)
        if swift:
            tool = bin_dir / "swift"
            tool.write_text(SWIFT_STUB, encoding="utf-8")
            tool.chmod(0o755)
        yield bin_dir


def run_bash(script: str, env: dict[str, str], bin_dir: pathlib.Path, *, path: str | None = None):
    """Run `script` with the stub directory first on PATH.

    `path` replaces PATH entirely, which is how a test asks for a machine with no
    `pgrep` on it at all: leaving /usr/bin in place would find the real one.
    """
    full = dict(os.environ)
    for key in ("STUB_PGREP_RC", "STUB_PGREP_OUT", "STUB_PGREP_ERR"):
        full.pop(key, None)
    full["PATH"] = path if path is not None else f"{bin_dir}:/usr/bin:/bin"
    full.update(env)
    return subprocess.run(
        ["/bin/bash", "-c", script], capture_output=True, text=True, check=False, env=full
    )


def code_lines(text: str) -> str:
    """The file without its comment lines, so a pin cannot be satisfied by prose."""
    return "\n".join(
        line for line in text.splitlines() if not line.strip().lstrip().startswith("#")
    )


def guard_function(text: str) -> str:
    """The `model_guard_matches` function as one file holds it, braces included."""
    start = text.index("model_guard_matches() {")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


class ModelGuardTriStateTests(unittest.TestCase):
    """`tools/model-guard.sh`'s own answer, for each status pgrep can give."""

    def _call(self, env: dict[str, str], *, pgrep: bool = True) -> subprocess.CompletedProcess:
        with stub_bin(pgrep=pgrep, swift=False) as bin_dir:
            script = f'. \'{GUARD}\'\nmodel_guard_matches "{PATTERN}"\nprintf "status=%s\\n" "$?"\n'
            # With no pgrep at all the PATH holds only the empty stub directory:
            # /usr/bin has the real one, and an answer from it is not the case.
            path = str(bin_dir) if not pgrep else None
            return run_bash(script, env, bin_dir, path=path)

    def test_the_guard_file_exists_and_is_sourced_not_executed(self) -> None:
        self.assertTrue(GUARD.is_file(), "no tools/model-guard.sh")
        text = GUARD.read_text(encoding="utf-8")
        self.assertIn("model_guard_matches() {", text)

    def test_a_match_is_reported_as_busy_with_the_lines(self) -> None:
        proc = self._call({"STUB_PGREP_RC": "0", "STUB_PGREP_OUT": BUSY_LINES})
        self.assertIn("status=0", proc.stdout, proc.stdout)
        self.assertIn("TinyTitanServer", proc.stdout, "the matched lines must reach the caller")

    def test_no_match_is_reported_as_clear(self) -> None:
        proc = self._call({"STUB_PGREP_RC": "1", "STUB_PGREP_OUT": ""})
        self.assertIn("status=1", proc.stdout, proc.stdout)
        self.assertEqual(proc.stdout, "status=1\n", "a clear answer prints nothing else")

    def test_an_erroring_pgrep_is_not_reported_as_clear(self) -> None:
        """Status 2 is the whole finding: the guard could not ask."""
        proc = self._call(
            {"STUB_PGREP_RC": "2", "STUB_PGREP_OUT": "", "STUB_PGREP_ERR": PGREP_ERROR}
        )
        self.assertIn("status=2", proc.stdout, f"an error must not read as status 1: {proc.stdout}")
        self.assertIn("2", proc.stderr, f"should name pgrep's status: {proc.stderr}")
        self.assertIn(
            "parentheses not balanced", proc.stderr, f"should carry pgrep's reason: {proc.stderr}"
        )

    def test_a_missing_pgrep_is_reported_as_cannot_ask(self) -> None:
        """`command not found` must not fall through as the empty answer it prints."""
        proc = self._call({}, pgrep=False)
        self.assertIn("status=2", proc.stdout, f"a missing pgrep read as an answer: {proc.stdout}")
        self.assertIn("pgrep", proc.stderr, proc.stderr)

    def test_the_guard_names_the_pattern_it_searched(self) -> None:
        """A refusal that does not say what it looked for sends the operator hunting."""
        proc = self._call(
            {"STUB_PGREP_RC": "2", "STUB_PGREP_OUT": "", "STUB_PGREP_ERR": PGREP_ERROR}
        )
        self.assertIn(PATTERN, proc.stderr, proc.stderr)

    def test_the_refusal_carries_pgreps_reason_in_its_own_message(self) -> None:
        """The reason rides inside the guard's line, not merely somewhere on stderr.

        A stub's stderr reaches the terminal whether or not the guard captured it,
        so `assertIn(reason, proc.stderr)` cannot tell a quoting guard from one that
        dropped `2>&1` and prints an empty explanation. This is that test.
        """
        proc = self._call(
            {"STUB_PGREP_RC": "2", "STUB_PGREP_OUT": "", "STUB_PGREP_ERR": PGREP_ERROR}
        )
        lines = [line for line in proc.stderr.splitlines() if line.startswith("model-guard:")]
        self.assertEqual(len(lines), 1, proc.stderr)
        self.assertIn("pgrep exited 2", lines[0], lines[0])
        self.assertIn("parentheses not balanced", lines[0], f"refusal lost the reason: {lines[0]}")


class GoldenBaselineGuardTests(unittest.TestCase):
    """End to end through the script that gates a golden capture."""

    def setUp(self) -> None:
        sys.path.insert(0, str(ROOT / "benchmark"))
        import test_golden_baseline_capture as fixture

        self.fixture = fixture

    def _run(self, rc: str, out: str, err: str = "") -> subprocess.CompletedProcess:
        # The fixture's own stub pgrep answers, now that its status is choosable.
        with self.fixture.tree() as root:
            env = dict(os.environ)
            env["PATH"] = f"{root / 'stubs'}{os.pathsep}{env['PATH']}"
            env["GOLDEN_FIXTURE_OUTPUT"] = self.fixture.SAMPLE
            env["STUB_PGREP_RC"] = rc
            env["STUB_PGREP_OUT"] = out
            env["STUB_PGREP_ERR"] = err
            return subprocess.run(
                ["/bin/bash", str(root / "tools" / "golden-baseline.sh"), "ornith-8"],
                capture_output=True,
                text=True,
                check=False,
                env=env,
                cwd=str(root),
            )

    def test_a_live_model_process_stops_the_capture(self) -> None:
        """The busy branch had no test anywhere before this finding."""
        proc = self._run("0", BUSY_LINES)
        self.assertEqual(
            proc.returncode, 3, f"a running server did not stop the capture: {proc.stderr}"
        )
        self.assertIn("refusing to start", proc.stderr, proc.stderr)
        self.assertIn("TinyTitanServer", proc.stderr, proc.stderr)

    def test_an_erroring_pgrep_stops_the_capture(self) -> None:
        proc = self._run("2", "", PGREP_ERROR)
        self.assertEqual(
            proc.returncode,
            3,
            f"the capture started while the guard could not answer: {proc.stderr}",
        )
        self.assertIn("refusing to start", proc.stderr, f"should refuse, not warn: {proc.stderr}")
        self.assertIn("pgrep", proc.stderr.lower(), f"should name the command: {proc.stderr}")
        self.assertIn("parentheses not balanced", proc.stderr, proc.stderr)

    def test_no_process_still_runs_the_capture(self) -> None:
        """The guard must not become a refusal that blocks every capture: a
        proven-clear answer still reaches the run."""
        proc = self._run("1", "")
        self.assertNotIn("refusing to start", proc.stderr, proc.stderr)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("captured", proc.stdout, proc.stdout)


class VerifyCpuModelsGuardTests(unittest.TestCase):
    """End to end through the CPU-model verifier, which must not reach its build."""

    def _run(self, rc: str, out: str, err: str = "") -> subprocess.CompletedProcess:
        with stub_bin() as bin_dir:
            env = {
                "STUB_PGREP_RC": rc,
                "STUB_PGREP_OUT": out,
                "STUB_PGREP_ERR": err,
            }
            proc = run_bash(f"/bin/bash '{ROOT / 'tools' / 'verify_cpu_models.sh'}'", env, bin_dir)
            self.assertNotIn(
                "SWIFT-STUB",
                proc.stdout + proc.stderr,
                "the verifier reached its build past the guard",
            )
            return proc

    def test_a_live_model_process_stops_the_verifier(self) -> None:
        proc = self._run("0", BUSY_LINES)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("already running", (proc.stdout + proc.stderr).lower(), proc.stdout)

    def test_an_erroring_pgrep_stops_the_verifier(self) -> None:
        proc = self._run("2", "", PGREP_ERROR)
        self.assertEqual(
            proc.returncode, 2, f"an unanswered guard read as clear: {proc.stdout}{proc.stderr}"
        )
        self.assertIn(
            "parentheses not balanced",
            proc.stdout + proc.stderr,
            proc.stdout + proc.stderr,
        )

    def test_the_verifier_prints_the_lines_it_was_given(self) -> None:
        """The old script asked pgrep a second time to print the matches; the
        guard hands back the lines it already collected, once."""
        proc = self._run("0", BUSY_LINES)
        self.assertIn("17514", proc.stdout + proc.stderr, proc.stdout + proc.stderr)


class InstallerGuardBlockTests(unittest.TestCase):
    """The installer's own guard block, extracted and driven with stub warn/die."""

    def _block(self) -> str:
        """The installer's guard site: the blank-line-delimited statement block
        that calls the guard, comments dropped, run under stub `warn`/`die`."""
        text = INSTALLER.read_text(encoding="utf-8")
        at = text.index(f"model_guard_matches '{PATTERN}'")
        start = text.rindex("\n\n", 0, at) + 2
        end = text.index("\nfi\n", at) + 4
        lines = [line for line in text[start:end].splitlines() if not line.strip().startswith("#")]
        return "\n".join(lines) + "\n"

    def _run(self, rc: str, out: str, err: str = "") -> subprocess.CompletedProcess:
        with stub_bin(pgrep=True, swift=False) as bin_dir:
            script = textwrap.dedent(
                f"""
                set -uo pipefail
                warn() {{ echo "WARN: $*"; }}
                die() {{ echo "DIE: $*"; exit 1; }}
                {guard_function(INSTALLER.read_text(encoding="utf-8"))}
                {self._block()}
                echo "REACHED_THE_REST"
                """
            )
            return run_bash(
                script,
                {"STUB_PGREP_RC": rc, "STUB_PGREP_OUT": out, "STUB_PGREP_ERR": err},
                bin_dir,
            )

    def test_a_live_model_process_is_still_only_a_warning(self) -> None:
        """The site is warn-only by design: the installer starts no model itself."""
        proc = self._run("0", BUSY_LINES)
        self.assertIn("WARN:", proc.stdout, proc.stdout)
        self.assertIn("REACHED_THE_REST", proc.stdout, proc.stdout)
        self.assertIn(
            "TinyTitanServer",
            proc.stdout,
            "the guard handed back the lines it matched; the warning should name them",
        )

    def test_an_erroring_pgrep_warns_that_it_could_not_check(self) -> None:
        proc = self._run("2", "", PGREP_ERROR)
        self.assertIn(
            "WARN:",
            proc.stdout,
            f"an unanswered guard printed nothing at all: {proc.stdout}",
        )
        self.assertIn("could not", proc.stdout.lower(), proc.stdout)

    def test_no_process_still_warns_nothing(self) -> None:
        proc = self._run("1", "")
        self.assertNotIn("WARN:", proc.stdout, proc.stdout)
        self.assertIn("REACHED_THE_REST", proc.stdout, proc.stdout)

    def test_the_installers_copy_is_the_guards_function(self) -> None:
        """The installer cannot source tools/model-guard.sh (it arrives over a
        pipe with no repository beside it), so it carries a copy -- and the copy
        has to be the same function, or the two guards answer differently."""
        self.assertEqual(
            guard_function(INSTALLER.read_text(encoding="utf-8")),
            guard_function(GUARD.read_text(encoding="utf-8")),
        )


class CallerWiringTests(unittest.TestCase):
    """Every guard in the tree goes through one owner, in each language."""

    def test_no_shell_script_asks_pgrep_outside_the_guard(self) -> None:
        """One owner per language: a script that answers pgrep itself can only
        ever act on the status it is looking for, which is the defect."""
        offenders = []
        for directory in ("tools", "benchmark"):
            for script in sorted((ROOT / directory).rglob("*.sh")):
                if script.name in {"model-guard.sh", "install_tinytitan.sh"}:
                    continue  # the owner, and the one copy a pipe cannot source
                for line in code_lines(script.read_text(encoding="utf-8")).splitlines():
                    if re.search(r"\bpgrep\b", line):
                        offenders.append(f"{script.relative_to(ROOT)}: {line.strip()}")
        self.assertEqual(offenders, [], f"guards that answer pgrep themselves: {offenders}")

    def test_the_repository_callers_source_the_guard(self) -> None:
        for rel in SHELL_CALLERS:
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIn("model-guard.sh", text, f"{rel} does not load the guard")
            self.assertIn("model_guard_matches", text, f"{rel} does not call the guard")
            self.assertNotIn(
                "pgrep",
                code_lines(text),
                f"{rel} asks pgrep beside the guard",
            )

    def test_the_installer_carries_the_guard_inline(self) -> None:
        text = INSTALLER.read_text(encoding="utf-8")
        self.assertIn("model_guard_matches() {", text, "no inline guard")
        self.assertIn("model_guard_matches 'TinyTitanServer|TinyTitanCLI'", text)

    def test_the_python_drivers_ask_through_the_shared_answer(self) -> None:
        for rel in PYTHON_CALLERS:
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIn("pgrep_answer", text, f"{rel} still reads pgrep's stdout as its answer")
            self.assertIn(
                '!= "clear"',
                text,
                f"{rel} treats anything the guard did not answer as clear",
            )
            self.assertNotRegex(
                text, r'subprocess\.run\(\s*\[\s*"pgrep"', f"{rel} runs pgrep beside the helper"
            )

    def test_no_python_reads_pgrep_outside_the_helper(self) -> None:
        offenders = []
        for module in sorted((ROOT / "benchmark").glob("*.py")):
            if module.name in {"tinytitan_profile.py"} or module.name.startswith("test_"):
                continue  # the owner, and fixtures that write a stub named pgrep
            for number, line in enumerate(module.read_text(encoding="utf-8").splitlines(), 1):
                if '"pgrep"' in line and not line.strip().startswith("#"):
                    offenders.append(f"{module.name}:{number}: {line.strip()}")
        self.assertEqual(offenders, [], f"python guards that discard their own status: {offenders}")

    def test_the_helper_is_the_only_python_owner(self) -> None:
        text = PROFILE.read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"def pgrep_answer", text)), 1)
        # One `subprocess` call names pgrep, and it is the helper's.
        self.assertEqual(text.count('"pgrep"'), 1, text[:200])


class PgrepAnswerTests(unittest.TestCase):
    """`tinytitan_profile.pgrep_answer` maps pgrep's statuses to three verdicts."""

    def setUp(self) -> None:
        sys.path.insert(0, str(ROOT / "benchmark"))
        import tinytitan_profile

        self.profile = tinytitan_profile
        self._stack = contextlib.ExitStack()
        self.bin = self._stack.enter_context(stub_bin())
        self._stack.enter_context(
            unittest.mock.patch.dict(os.environ, {"PATH": f"{self.bin}:/usr/bin:/bin"})
        )
        self.addCleanup(self._stack.close)

    def _answer(self, env: dict[str, str], argv: list[str] | None = None):
        os.environ.update(env)
        return self.profile.pgrep_answer(argv or ["-fl", PATTERN])

    def test_a_match_is_busy_and_carries_the_lines(self) -> None:
        verdict, lines = self._answer({"STUB_PGREP_RC": "0", "STUB_PGREP_OUT": BUSY_LINES})
        self.assertEqual(verdict, "busy", (verdict, lines))
        self.assertIn("TinyTitanServer", "\n".join(lines))

    def test_no_match_is_clear(self) -> None:
        verdict, lines = self._answer({"STUB_PGREP_RC": "1", "STUB_PGREP_OUT": ""})
        self.assertEqual(verdict, "clear", (verdict, lines))
        self.assertEqual(lines, [])

    def test_an_erroring_pgrep_is_neither_busy_nor_clear(self) -> None:
        """The status the drivers threw away has to reach them."""
        verdict, lines = self._answer(
            {"STUB_PGREP_RC": "2", "STUB_PGREP_OUT": "", "STUB_PGREP_ERR": PGREP_ERROR}
        )
        self.assertEqual(verdict, "unknown", (verdict, lines))
        joined = "\n".join(lines)
        self.assertIn("2", joined, "the verdict should name pgrep's status")
        self.assertIn("parentheses not balanced", joined)

    def test_a_missing_pgrep_is_unknown(self) -> None:
        (self.bin / "pgrep").unlink()
        # A PATH with nothing else on it: /usr/bin holds the real pgrep, and an
        # answer from it is not the machine this case is about.
        with unittest.mock.patch.dict(os.environ, {"PATH": str(self.bin)}):
            verdict, lines = self._answer({})
        self.assertEqual(verdict, "unknown", (verdict, lines))
        self.assertIn("pgrep", "\n".join(lines))

    def test_a_pid_answer_is_handed_back_unchanged(self) -> None:
        """vs_competitors reads a pid out of `pgrep -n`, so the lines are the
        payload, not just a yes/no."""
        verdict, lines = self._answer(
            {"STUB_PGREP_RC": "0", "STUB_PGREP_OUT": "4321\n"}, ["-n", "ollama"]
        )
        self.assertEqual(verdict, "busy", (verdict, lines))
        self.assertEqual(lines, ["4321"])

    def test_no_match_on_a_pid_is_clear(self) -> None:
        verdict, lines = self._answer(
            {"STUB_PGREP_RC": "1", "STUB_PGREP_OUT": ""}, ["-n", "ollama"]
        )
        self.assertEqual((verdict, lines), ("clear", []))


class CompetitorPidSiteTests(unittest.TestCase):
    """`benchmark/tinytitan_vs_competitors.py` samples a pid out of `pgrep -n`.

    Its three answers have to reach the table as three different results: a pid
    means a memory number, "nothing matches" means the competitor is honestly
    absent, and an unanswered probe means *not known* — not the zero RSS row that
    made a running competitor look like it used no memory (AUD-268). The
    competitor's own HTTP reply is faked, so nothing here dials a live engine.
    """

    OLLAMA_BODY = json.dumps({"eval_count": 100, "eval_duration": 10_000_000_000}).encode()
    LMSTUDIO_BODY = json.dumps({"stats": {"tokens_per_second": 12.5}}).encode()
    UNKNOWN = ["pgrep exited 2 and did not answer -n ollama: " + PGREP_ERROR]

    def setUp(self) -> None:
        sys.path.insert(0, str(ROOT / "benchmark"))
        import tinytitan_vs_competitors as rivals

        self.rivals = rivals

    def _run(
        self, runner: str, body: bytes, verdict: str, lines: list[str]
    ) -> tuple[tuple, list[int]]:
        pids: list[int] = []

        class FakeResponse:
            def read(self) -> bytes:
                return body

        class FakeConnection:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def request(self, *args, **kwargs) -> None:
                pass

            def getresponse(self) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        def fake_rss(pid: int) -> float:
            pids.append(pid)
            return 1024.0

        with (
            unittest.mock.patch.object(self.rivals, "pgrep_answer", return_value=(verdict, lines)),
            unittest.mock.patch("http.client.HTTPConnection", FakeConnection),
            unittest.mock.patch.object(self.rivals, "sample_rss", fake_rss),
        ):
            return getattr(self.rivals, runner)("a prompt"), pids

    def test_an_unanswered_pid_probe_is_not_reported_as_no_memory(self) -> None:
        for runner, body, name in (
            ("run_ollama", self.OLLAMA_BODY, "ollama"),
            ("run_lmstudio", self.LMSTUDIO_BODY, "LM Studio"),
        ):
            (rate, rss, note), pids = self._run(runner, body, "unknown", self.UNKNOWN)
            self.assertIsNone(rate, f"{runner} published a rate it should have refused")
            self.assertIsNone(rss, f"{runner} published an RSS it never measured")
            self.assertEqual(pids, [], f"{runner} sampled a pid that was never answered")
            self.assertIn("cannot ask", note, f"{runner} hid the reason: {note!r}")
            self.assertIn(name, note, f"{runner} did not say which engine: {note!r}")

    def test_an_answered_pid_is_sampled(self) -> None:
        (rate, rss, note), pids = self._run("run_ollama", self.OLLAMA_BODY, "busy", ["4321"])
        self.assertEqual((rate, rss, note), (10.0, 1024.0, ""))
        self.assertEqual(pids, [4321], "the pid is the whole point of `pgrep -n`")

    def test_a_competitor_that_is_actually_absent_reports_no_rss(self) -> None:
        (rate, rss, note), pids = self._run("run_ollama", self.OLLAMA_BODY, "clear", [])
        self.assertEqual((rate, rss, note), (10.0, None, ""))
        self.assertEqual(pids, [])


if __name__ == "__main__":
    unittest.main()
