"""Gates `benchmark/tinytitan_determinism_ab.py`'s verdict.

The driver starts the server twice and compares the streamed content deltas, to
answer one question: is greedy decoding reproducible across fresh processes. Until
AUD-224 the answer was printed unconditionally —

    print("None for both => 512-token greedy is deterministic across fresh processes")

— after the per-prompt rows, whatever those rows measured. And the comparison itself
cannot tell an empty stream from a reproducible one: `extract_deltas` skips every
line that is not `data: ` and every chunk it cannot parse, so a server that answered
with a JSON error instead of an SSE stream returns `[]` with no exception raised. Two
empty streams give `first_diff_index == None` and `sha256("") == sha256("")`, so
`equal=True` on every row and the page below it announces determinism. That is the
shape AUD-218 and AUD-212 each just fixed elsewhere: a verdict a run that measured
nothing passes.

The distinction is the one AUD-213 drew in `memval_master.sh`: a difference *is* a
result and exits 0, while a pair of streams that carried no content is not a result
at all. These tests call the pure `verdict()` over synthesized delta lists — the
servers themselves need a model, which this repository's verification rules reserve
for installs already present, and no determinism claim is being made here.

    cd benchmark && python3 -m unittest test_tinytitan_determinism_ab -v
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "tinytitan_determinism_ab", ROOT / "benchmark" / "tinytitan_determinism_ab.py"
)
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)

TOKENS = ["The", " history", " of"]
OTHER = ["The", " future", " of"]


class VerdictTests(unittest.TestCase):
    def test_an_empty_stream_on_both_sides_is_not_determinism(self):
        """The dead-run case: both servers answered something that carries no
        content delta, so nothing was compared and nothing was proved."""
        lines, status = ab.verdict([("essay", [], [])], max_tokens=512)
        page = "\n".join(lines)
        self.assertNotEqual(status, 0, "a run that streamed nothing must not exit 0")
        self.assertNotIn("deterministic across fresh processes", page)
        self.assertIn("NOT MEASURED", page)

    def test_an_empty_stream_on_one_side_names_the_prompt_and_the_side(self):
        """Which prompt, and which server. `tokens A=0 B=17` is readable but the old
        conclusion line still printed under it, so the row and the verdict disagreed.
        """
        lines, status = ab.verdict([("digits", [], TOKENS)], max_tokens=512)
        self.assertNotEqual(status, 0)
        self.assertIn("digits", "\n".join(lines))
        self.assertIn("A", "\n".join(lines))

    def test_a_real_difference_is_a_result_and_exits_zero(self):
        """The guard the fix must not swallow: two streams that disagree is the
        finding this driver exists to produce, and it is not an error status."""
        lines, status = ab.verdict([("essay", TOKENS, OTHER)], max_tokens=512)
        self.assertEqual(status, 0, "\n".join(lines))
        self.assertIn("first_diff_index=1", "\n".join(lines))
        self.assertNotIn("NOT MEASURED", "\n".join(lines))

    def test_two_matching_streams_still_conclude(self):
        """The ceiling case: content on both sides, identical, is exactly what the
        conclusion was written for, and it survives the guard."""
        lines, status = ab.verdict([("essay", TOKENS, list(TOKENS))], max_tokens=512)
        self.assertEqual(status, 0, "\n".join(lines))
        self.assertIn("deterministic across fresh processes", "\n".join(lines))

    def test_the_conclusion_names_the_length_that_was_run(self):
        """`MAX_TOKENS` is the driver's one argument, and the sentence hardcoded
        `512-token`, so `... 1024` printed a conclusion about a run it did not do."""
        lines, status = ab.verdict([("essay", TOKENS, list(TOKENS))], max_tokens=1024)
        self.assertEqual(status, 0)
        page = "\n".join(lines)
        self.assertIn("1024-token", page)
        self.assertNotIn("512-token", page)

    def test_a_json_error_body_streams_no_deltas(self):
        """The mechanism, pinned so the guard above is known to be reachable rather
        than theoretical: a non-SSE error response parses to zero deltas and raises
        nothing."""
        body = [b'{"error": {"message": "model directory mismatch"}}\n', b"data: [DONE]\n"]
        self.assertEqual(ab.extract_deltas(body), [])


class ArgumentTests(unittest.TestCase):
    def test_importing_the_module_does_not_read_the_importers_argv(self):
        """`MAX_TOKENS = int(sys.argv[1])` at module level meant any program that
        imported this file ran the cast against its own argv: measured as a
        `ValueError` naming the importer's argument at import time. The test passes
        a non-integer as the `-c` program's own `argv[1]`, which is what the old
        code died on.
        """
        result = subprocess.run(
            [sys.executable, "-c", "import tinytitan_determinism_ab", "not-a-number"],
            cwd=str(ROOT / "benchmark"),
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT / "benchmark")},
        )
        self.assertEqual(result.returncode, 0, result.stderr[:400])
        self.assertNotIn("server A", result.stdout)

    def test_a_non_numeric_length_is_refused_with_the_value_that_failed(self):
        for argv in (["prog", "report"], ["prog", "0"], ["prog", "-5"]):
            value, error = ab.parse_max_tokens(argv)
            self.assertIsNone(value, argv)
            self.assertIn(argv[1], error)

    def test_no_argument_keeps_the_documented_default(self):
        self.assertEqual(ab.parse_max_tokens(["prog"]), (512, None))

    def test_an_explicit_length_is_used(self):
        self.assertEqual(ab.parse_max_tokens(["prog", "1024"]), (1024, None))


if __name__ == "__main__":
    unittest.main()
