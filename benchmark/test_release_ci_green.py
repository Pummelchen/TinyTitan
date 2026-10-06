"""`tools/ci-green.sh` decides whether a commit may be tagged and published.

The row this pins (AUD-105) is a real one: v5.18 was published an hour after its
own tag commit came back `CI | completed | failure`. Nothing asked CI before
publishing, and `docs/release-process.md` said "a green main" in prose while no
check read it. So the arithmetic here is the whole defence, and every branch of
it is tested against a stub `gh` -- green, red, still running, and the two shapes
that would let a vacuous pass through: a commit CI never saw, and a short sha the
API matches against nothing.

Run from `benchmark/`:

    python3 -m unittest test_release_ci_green
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "ci-green.sh"
REPO = "Pummelchen/TinyTitan"
FULL_SHA = "0123456789abcdef0123456789abcdef01234567"

GH_STUB = """#!/bin/sh
if [ "${CI_GREEN_GH_FAIL:-0}" = "1" ]; then
    echo "gh: HTTP 502" >&2
    exit 1
fi
if [ -n "${CI_GREEN_FIXTURE:-}" ]; then
    cat "$CI_GREEN_FIXTURE"
fi
exit 0
"""


def run(
    *args: str,
    fixture: str | None = None,
    gh_fails: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run the helper with `gh` replaced by the stub and a fixture as its answer."""
    with tempfile.TemporaryDirectory() as work:
        stub = pathlib.Path(work) / "gh"
        stub.write_text(GH_STUB, encoding="utf-8")
        stub.chmod(0o755)
        env = {
            "PATH": f"{work}{os.pathsep}/bin:/usr/bin",
            "CI_GREEN_FIXTURE": str(pathlib.Path(work) / "fixture.tsv"),
        }
        fixture_path = pathlib.Path(work) / "fixture.tsv"
        fixture_path.write_text(fixture or "", encoding="utf-8")
        if gh_fails:
            env["CI_GREEN_GH_FAIL"] = "1"
        return subprocess.run(
            ["/bin/bash", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )


class CiGreenTests(unittest.TestCase):
    def test_all_success_is_green(self) -> None:
        result = run(
            REPO,
            FULL_SHA,
            fixture="CI\tcompleted\tsuccess\thttps://x/1\nCodeQL\tcompleted\tsuccess\thttps://x/2\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CI green", result.stdout)
        self.assertEqual(result.stdout.splitlines()[-1], "result\tgreen\t-")

    def test_a_skipped_workflow_is_not_a_failure(self) -> None:
        # CodeQL has its own trigger conditions; a run GitHub skipped says nothing
        # about this commit either way, and refusing it would block releases on a
        # workflow that was never meant to run.
        result = run(
            REPO,
            FULL_SHA,
            fixture="CI\tcompleted\tsuccess\thttps://x/1\nDocs\tcompleted\tskipped\thttps://x/2\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_a_failed_run_is_refused_and_names_the_run(self) -> None:
        result = run(
            REPO,
            FULL_SHA,
            fixture="CI\tcompleted\tfailure\thttps://x/9\nCodeQL\tcompleted\tsuccess\thttps://x/2\n",
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("CI is red", result.stderr)
        self.assertIn("CI (failure)", result.stderr)
        self.assertIn("https://x/9", result.stderr)

    def test_a_cancelled_or_timed_out_run_is_refused(self) -> None:
        # Not `failure`, and not a pass: the workflow did not complete its checks.
        for conclusion in ("cancelled", "timed_out", "action_required", "stale"):
            with self.subTest(conclusion=conclusion):
                result = run(
                    REPO,
                    FULL_SHA,
                    fixture=f"CI\tcompleted\t{conclusion}\thttps://x/9\n",
                )
                self.assertEqual(result.returncode, 1, conclusion)

    def test_a_run_still_going_is_refused(self) -> None:
        # Releasing over an in-flight run is guessing at an answer that is about to
        # arrive, and the override must not be the way out of waiting.
        result = run(REPO, FULL_SHA, fixture="CI\tin_progress\t-\thttps://x/9\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("has not finished", result.stderr)
        self.assertIn("--allow-red", result.stderr)

    def test_no_run_at_all_is_refused(self) -> None:
        # The vacuous pass this row is about: a query that matches nothing must not
        # read as "nothing failed, so go".
        result = run(REPO, FULL_SHA, fixture="")
        self.assertEqual(result.returncode, 1)
        self.assertIn("never saw", result.stderr)

    def test_a_short_sha_is_refused_before_the_query(self) -> None:
        # `head_sha` with a short sha returns no runs, so the answer to a short sha
        # would always be "no run at all" -- an error message that reads like a
        # finding about CI rather than about the argument. Say which it is.
        result = run(REPO, "ea5de8c", fixture="CI\tcompleted\tsuccess\thttps://x/1\n")
        self.assertEqual(result.returncode, 2)
        self.assertIn("full commit sha", result.stderr)

    def test_a_failed_query_is_not_a_pass(self) -> None:
        # No token, no network, a renamed repo: whatever it is, the helper cannot
        # claim CI passed on an answer it did not get.
        result = run(
            REPO,
            FULL_SHA,
            fixture="CI\tcompleted\tsuccess\thttps://x/1\n",
            gh_fails=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot read", result.stderr)

    def test_missing_arguments_are_a_usage_error(self) -> None:
        result = run(REPO)
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)

    def test_red_ci_passes_only_with_a_reason(self) -> None:
        fixture = "CI\tcompleted\tfailure\thttps://x/9\n"
        refused = run(REPO, FULL_SHA, fixture=fixture)
        self.assertEqual(refused.returncode, 1)

        allowed = run(REPO, FULL_SHA, "--allow-red", "flake in the runner image", fixture=fixture)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        self.assertIn("releasing over it: flake in the runner image", allowed.stdout)
        # The caller needs the run URL out of the machine-readable line: it is what
        # the release notes then have to quote.
        self.assertEqual(allowed.stdout.splitlines()[-1], "result\toverridden\thttps://x/9")

    def test_an_empty_reason_is_refused(self) -> None:
        # `--allow-red ""` would make the override as anonymous as having no check.
        result = run(
            REPO, FULL_SHA, "--allow-red", "", fixture="CI\tcompleted\tfailure\thttps://x/9\n"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("non-empty reason", result.stderr)

    def test_an_unrecognised_option_is_refused(self) -> None:
        result = run(REPO, FULL_SHA, "--force", fixture="")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be --allow-red", result.stderr)


class ReleaseScriptTests(unittest.TestCase):
    """`tools/release.sh` must ask the question and honour both answers."""

    def script(self) -> str:
        return (ROOT / "tools" / "release.sh").read_text(encoding="utf-8")

    def test_the_release_script_calls_the_helper(self) -> None:
        self.assertIn("ci-green.sh", self.script())

    def test_the_override_needs_a_reason_in_the_script_too(self) -> None:
        # The helper enforces a reason for its own argument; the script's env-var
        # path is a second door to the same room and needs the same lock.
        self.assertIn("TINYTITAN_RELEASE_ALLOW_RED_CI_REASON", self.script())

    def test_an_overridden_release_requires_the_run_url_in_the_notes(self) -> None:
        text = self.script()
        self.assertIn('CI_NOTES_REQUIRE="$CI_RED_URL"', text)
        start = text.index("CI_NOTES_REQUIRE=")
        end = text.index('REQUIRE_ARGS+=(--require "$CI_NOTES_REQUIRE")')
        self.assertLess(start, end, "the notes guard must come after the precondition")


if __name__ == "__main__":
    unittest.main()
