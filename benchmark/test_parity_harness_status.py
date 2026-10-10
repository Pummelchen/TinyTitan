"""AUD-269: the parity harness prints its verdicts and exits 0 whatever they say.

`tools/qwen38_parity.py`, `tools/qwen38_full_forward.py` and
`tools/qwen38_sequence_parity.py` are the only instruments that check the runtime's
forward pass against an independent numpy reference -- `docs/qwen38-flash-next-port.md`
calls them the parity harness, and `docs/deepseek-v41-flash-port.md` makes them Gate 3
for porting a new model. They compute a verdict for every stage and then discard it:

    def report(label, mine, reference) -> bool:   # tools/qwen38_parity.py:44
    report("embed (wide)", embed, np.tile(row, HC))   # :71 -- the bool is dropped

Nine such calls in `main()`, no `sys.exit` in the file, and `main()` returns `None`.
`qwen38_full_forward.py` is the same shape with its verdict made explicit -- it sets
`first_bad`, prints "first divergence at layer 3", and `return`s at line 87 without a
status. Measured on this host (the probe this suite carries): a synthetic weight table
and dumps that disagree with it print

    embed (wide)  ok   max_rel=0.0003 cos=1.00000
    L0 attn read gate  FAIL ...   (eight more of these)

and the process exits **0**, so `python3 tools/qwen38_parity.py ... && echo "parity
ok"` prints "parity ok" over a run where eight of nine stages disagreed. The one
consequence that matters is the one the audit keeps finding: the status is the only
thing that composes, so a harness that cannot say "no" cannot be wired into a gate, a
release step or CI -- and an operator who wires it in anyway is told the check passed.
`tools/qwen35_reference.py:411` already returns `2` on its failures, so the convention
exists in this tree; only the qwen38 trio drops it.

The harness has a second half of the same defect: it cannot tell "these two
implementations disagree" from "I had nothing to compare". A dump directory holding no
`posN`/`L*_entry.f16` files makes `qwen38_sequence_parity.py` glob an empty list, print
nothing and exit 0, and makes the other two die on `np.fromfile` with a traceback whose
status is 1 -- the same status a genuine disagreement now has to carry. So the three
statuses below are the whole point of the fix, and the tests pin all three:

    0  compared, and every stage agreed
    1  compared, and at least one stage disagreed
    2  could not compare, and why is on stdout

Nothing here loads a model, opens a socket or needs an install: the weights are a
synthetic table with the shapes the real stages demand, the dumps are real `.f16` files
in a scratch directory, and the scripts run through their own `__main__` so the exit
status is the one a shell would read. If numpy is missing the suite fails rather than
skipping -- a harness proven on nothing is the defect this file is about.

It is the slowest suite in its gate group (about three minutes), because the sequence
cases run the whole 48-layer reference in numpy before they reach the guard they are
testing, and the script's order is what the guard is.

Run from `benchmark/`:

    python3 -m unittest test_parity_harness_status
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PARITY = ROOT / "tools" / "qwen38_parity.py"
FULL_FORWARD = ROOT / "tools" / "qwen38_full_forward.py"
SEQUENCE = ROOT / "tools" / "qwen38_sequence_parity.py"

# One driver, three scripts. It installs a synthetic weight reader under the name the
# scripts import, writes the dumps the script asked for, then runs the real file as
# __main__ so the process status is the real thing.
PROBE = textwrap.dedent(
    """
    import importlib, json, runpy, sys, types
    from pathlib import Path

    import numpy as np

    script, dump, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    sys.path.insert(0, str(script.parent))
    import qwen38_parity as m
    import qwen38_reference as r

    RNG = np.random.default_rng(7)
    D, HCD = m.D, m.HC * m.D
    HID = 8

    # In the agreeing modes every block's output projection is zero, so each layer
    # leaves the wide residual exactly as it found it and every entry dump can hold
    # the one vector the run started from. That is how a clean pass is reachable
    # here without a checkpoint: the reference and the dumps are the same numbers.
    NULL_OUTPUT = (
        "linear_attn.out_proj.weight",
        "self_attn.o_proj.weight",
        "shared_expert.down_proj.weight",
        "ple.value_proj.weight",
        "ple.conv1d",
    )

    AGREE_MODES = ("agree", "stackout", "agree_no_token")

    def is_null(name):
        return mode in AGREE_MODES and name.endswith(NULL_OUTPUT)

    def shape_for(name):
        if name.endswith(("hc_norm", "norm_key", "norm_query", "norm_conv")):
            return (HCD,)
        if name.endswith("input_mix_weight_down.weight"):
            return (HID, HCD)
        if name.endswith("input_mix_weight_up.weight"):
            return (HCD, HID)
        if name.endswith("block_inject_weight.weight"):
            return (m.HC, HCD)
        if name.endswith("ple.key_proj.weight"):
            return (HCD, HID)
        if name.endswith("ple.value_proj.weight"):
            return (D, HID)
        if name.endswith("ple.conv1d"):
            return (HCD, m.PLE_K)
        if name.endswith("linear_attn.in_proj_qkv.weight"):
            return (m.HK * m.DK * 2 + m.HV * m.DV, D)
        if name.endswith("linear_attn.in_proj_z.weight"):
            return (m.HV * m.DV, D)
        if name.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")):
            return (m.HV, D)
        if name.endswith("linear_attn.conv1d.weight"):
            return (m.HK * m.DK * 2 + m.HV * m.DV, m.CONV_K)
        if name.endswith(("linear_attn.A_log", "linear_attn.dt_bias")):
            return (m.HV,)
        if name.endswith("linear_attn.norm.weight"):
            return (m.DV,)
        if name.endswith("linear_attn.out_proj.weight"):
            return (D, m.HV * m.DV)
        if name.endswith("mlp.gate.weight"):
            return (m.NUM_EXPERTS, D)
        if name.endswith("mlp.shared_expert_gate.weight"):
            return (1, D)
        if name.endswith(("shared_expert.gate_proj.weight", "shared_expert.up_proj.weight")):
            return (HID, D)
        if name.endswith("shared_expert.down_proj.weight"):
            return (D, HID)
        if name.endswith("self_attn.q_proj.weight"):
            return (m.N_HEADS * 2 * m.HEAD_DIM, D)
        if name.endswith("self_attn.k_proj.weight"):
            return (m.N_KV_HEADS * m.HEAD_DIM, D)
        if name.endswith(("self_attn.q_norm", "self_attn.k_norm")):
            return (m.HEAD_DIM,)
        if name.endswith("indexer.index_k_proj.weight"):
            return (r.INDEXER_DIM, D)
        if name.endswith("self_attn.v_proj.weight"):
            return (m.N_KV_HEADS * m.HEAD_DIM, D)
        if name.endswith("self_attn.o_proj.weight"):
            return (D, m.N_HEADS * m.HEAD_DIM)
        if name.endswith("embed_tokens.weight"):
            return (1, D)
        if name == "lm_head.weight":
            return (16, D)
        raise AssertionError("unwired weight: " + name)

    class FakeWeights:
        cache = {}

        def __init__(self, model_dir):
            pass

        def get(self, name):
            # Cached by name, so a dump can be written to equal the reference the
            # script will compute from the same draw.
            if name not in self.cache:
                shape = shape_for(name)
                draw = np.zeros(shape) if is_null(name) else RNG.standard_normal(shape)
                self.cache[name] = draw.astype(np.float32)
            return self.cache[name]

    class FakeExperts:
        cache = {}

        def __init__(self, model_dir):
            pass

        def tensor(self, layer, expert, which):
            key = (layer, expert, which)
            if key not in self.cache:
                shape = (D, HID) if which == "down" else (HID, D)
                null = mode in AGREE_MODES and which == "down"
                draw = np.zeros(shape) if null else RNG.standard_normal(shape)
                self.cache[key] = draw.astype(np.float32)
            return self.cache[key]

    fake = types.ModuleType("ssdai_reader")
    fake.SSDAIWeights = FakeWeights
    fake.PackedExperts = FakeExperts
    sys.modules["ssdai_reader"] = fake
    importlib.reload(m)
    # Reference holds the weight classes in its own namespace, so it needs the same reload.
    importlib.reload(r)

    dump.mkdir(parents=True, exist_ok=True)
    # The stateful reference reads two things straight out of the model directory
    # instead of through the weight reader: its PLE constants and its n-gram table.
    # One row of a table whose heads are all modulo 1 puts every lookup at row 0,
    # which is the smallest table the block will accept.
    model = dump.parent / "synthetic-model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "ple_constants.json").write_text(
        json.dumps(
            {
                "ngram_size": 2,
                "heads_per_ngram": 4,
                "ple_head_dim": 2,
                "layer_multipliers": [1, 1],
                "ngram_heads_offsets": [0, 0, 0, 0],
                "ngram_heads_vocab_sizes": [1, 1, 1, 1],
                "eos_token_id": 0,
            }
        )
    )
    np.zeros((1, 2), dtype=np.float16).tofile(model / "ngram_table.bin")
    PARITY_DUMPS = {
        "embed": HCD,
        "L0_attn_in": D,
        "L0_attn_out": D,
        "L0_hidden_post_attn": HCD,
        "L0_mlp_in": D,
        "L0_mlp_out": D,
        "L1_entry": HCD,
        "L1_attn_in": D,
        "L3_attn_in": D,
        "L3_attn_out": D,
        "ple_embedding": HID,
    }
    FORWARD_DUMPS = {f"L{layer}_entry": HCD for layer in range(48)}
    FORWARD_DUMPS["stack_out"] = HCD
    FORWARD_DUMPS["ple_embedding"] = HID

    names = dict(PARITY_DUMPS)
    names.update(FORWARD_DUMPS)
    if mode == "disagree":
        for name, length in names.items():
            np.zeros(length, dtype=np.float16).tofile(dump / f"{name}.f16")
        # The embedding stage is the one whose reference comes from the weights alone,
        # so it can be made to agree. A fixture in which every comparison is impossible
        # would prove nothing about the verdicts.
        weights = FakeWeights("synthetic")
        table = weights.get("model.language_model.embed_tokens.weight")
        np.tile(table[0], m.HC).astype(np.float16).tofile(dump / "embed.f16")
        (dump / "token.txt").write_text("0\\n")

    if mode in AGREE_MODES:
        # Every block leaves the residual alone, so every entry the run compares on
        # is the embedding row it started with -- and the check can be made to agree.
        row = np.tile(FakeWeights("synthetic").get("model.language_model.embed_tokens.weight")[0],
                      m.HC)
        for name, length in names.items():
            value = row if length == HCD and name != "ple_embedding" else np.zeros(length)
            np.asarray(value, dtype=np.float16).tofile(dump / f"{name}.f16")
        if mode == "stackout":
            # The one disagreement this fixture can place after the last layer: the
            # stack output is compared only once the whole stack agreed.
            np.zeros(HCD, dtype=np.float16).tofile(dump / "stack_out.f16")
        if mode != "agree_no_token":
            (dump / "token.txt").write_text("0\\n")
    if mode.startswith("sequence"):
        # Sequence parity reads one subdirectory per position, so its fixture roots at
        # pos0 and holds back exactly the file the guard is about.
        pos = dump / "pos0"
        pos.mkdir(parents=True, exist_ok=True)
        (pos / "token.txt").write_text("0\\n")
        if mode == "sequence_no_entries":
            pass
        elif mode == "sequence_no_stack_out":
            np.zeros(HCD, dtype=np.float16).tofile(pos / "L0_entry.f16")
        else:
            # The dumps are written from the reference the script will recompute in the
            # same process, from the same cached draw, so they agree exactly: this is the
            # run that must exit 0, which is what tells the two refusals apart from a
            # guard that refuses whatever it is handed.
            from qwen38_reference import Reference

            entries, stack_out, _ = Reference(str(model)).step(0)
            for layer, vector in enumerate(entries):
                np.asarray(vector, dtype=np.float16).tofile(pos / f"L{layer}_entry.f16")
            np.asarray(stack_out, dtype=np.float16).tofile(pos / "stack_out.f16")
    if script.name == "qwen38_sequence_parity.py":
        sys.argv = [str(script), str(model), str(dump)]
    elif script.name == "qwen38_parity.py":
        sys.argv = [str(script), str(model), str(dump), "0"]
    else:
        sys.argv = [str(script), str(model), str(dump)]
    runpy.run_path(str(script), run_name="__main__")
    """
)


class HarnessRun(unittest.TestCase):
    """Each case runs the real script as its own process and reads its status."""

    maxDiff = None

    def setUp(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError as missing:  # pragma: no cover - the gate needs it anyway
            self.fail(f"the parity harness needs numpy and it is not importable: {missing}")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.probe = pathlib.Path(self.tmp.name) / "parity_probe.py"
        self.probe.write_text(PROBE, encoding="utf-8")

    def run_harness(self, script: pathlib.Path, mode: str) -> subprocess.CompletedProcess:
        dump = pathlib.Path(self.tmp.name) / f"dump-{script.stem}-{mode}"
        proc = subprocess.run(
            [sys.executable, str(self.probe), str(script), str(dump), mode],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,  # the status is what each case asserts on
        )
        return proc


class ParityStatusTests(HarnessRun):
    def test_a_disagreeing_stage_makes_the_parity_script_refuse(self) -> None:
        proc = self.run_harness(PARITY, "disagree")
        self.assertNotIn("Traceback", proc.stderr, proc.stderr)
        self.assertIn("FAIL", proc.stdout, proc.stdout)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)

    def test_the_parity_run_reports_how_many_stages_agreed(self) -> None:
        """An operator reading the transcript must not have to count FAIL lines, and
        a count is what makes the status checkable."""
        proc = self.run_harness(PARITY, "disagree")
        self.assertIn("stage(s) agree", proc.stdout, proc.stdout)
        self.assertIn("1 of 9", proc.stdout, proc.stdout)

    def test_a_missing_dump_refuses_with_a_different_status(self) -> None:
        """'They disagree' and 'I had nothing to compare' cannot share a status, or
        the fix only moves the conflation."""
        proc = self.run_harness(PARITY, "missing")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertNotIn("Traceback", proc.stderr, proc.stderr)
        self.assertIn("embed", proc.stdout, proc.stdout)


class FullForwardStatusTests(HarnessRun):
    def test_a_divergence_makes_the_full_forward_check_refuse(self) -> None:
        proc = self.run_harness(FULL_FORWARD, "disagree")
        self.assertIn("first divergence at layer", proc.stdout, proc.stdout)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)

    def test_a_missing_dump_refuses_with_a_different_status(self) -> None:
        proc = self.run_harness(FULL_FORWARD, "missing")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertNotIn("Traceback", proc.stderr, proc.stderr)

    def test_a_run_missing_only_its_token_file_refuses(self) -> None:
        """Every dump was there and the token was not: the same unreadable run by a
        different door, and an unguarded read_text made it a traceback whose status
        aliased a disagreement."""
        proc = self.run_harness(FULL_FORWARD, "agree_no_token")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertNotIn("Traceback", proc.stderr, proc.stderr)
        self.assertIn("token.txt", proc.stdout, proc.stdout)

    def test_an_agreeing_stack_exits_0(self) -> None:
        """The status has to mean something, so it has to be 0 on a run that agrees
        and not merely non-1: a fix that always refused would pass every other case
        in this file."""
        proc = self.run_harness(FULL_FORWARD, "agree")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertNotIn("FAIL", proc.stdout, proc.stdout)
        self.assertIn("L47 entry", proc.stdout, proc.stdout)

    def test_the_stack_output_verdict_reaches_the_status(self) -> None:
        """The last comparison in the file was printed and dropped the same way the
        others were, one step past the loop that this defect is about."""
        proc = self.run_harness(FULL_FORWARD, "stackout")
        self.assertIn("stack out   FAIL", proc.stdout, proc.stdout)
        self.assertNotIn("first divergence", proc.stdout, proc.stdout)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)


class SequenceParityStatusTests(HarnessRun):
    def test_a_dump_root_holding_no_positions_is_not_a_pass(self) -> None:
        """The profile script claims no verdict, so its only failure mode is a scan
        that read nothing: an empty glob printed nothing and exited 0."""
        proc = self.run_harness(SEQUENCE, "missing")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertNotIn("Traceback", proc.stderr, proc.stderr)
        self.assertIn("pos", proc.stdout, proc.stdout)

    def test_a_position_holding_no_layer_dumps_refuses(self) -> None:
        """A position directory that exists but carries no `L<N>_entry.f16` printed a
        heading and then fell through to the next position, so a half-written dump
        tree profiled as a run with nothing wrong in it."""
        proc = self.run_harness(SEQUENCE, "sequence_no_entries")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertNotIn("Traceback", proc.stderr, proc.stderr)
        self.assertIn("no L<N>_entry.f16", proc.stdout, proc.stdout)

    def test_a_position_holding_no_stack_output_refuses(self) -> None:
        """The layer profile printed, the stack output did not exist, and the run
        that compared one of its two answers reported success."""
        proc = self.run_harness(SEQUENCE, "sequence_no_stack_out")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertNotIn("Traceback", proc.stderr, proc.stderr)
        self.assertIn("no stack_out.f16", proc.stdout, proc.stdout)

    def test_a_complete_position_exits_0(self) -> None:
        """The refusals above have to come from their own missing file, not from a
        guard that refuses whatever it is handed."""
        proc = self.run_harness(SEQUENCE, "sequence_agrees")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertNotIn("Traceback", proc.stderr, proc.stderr)
        self.assertIn("stack out cos=", proc.stdout, proc.stdout)


class VerdictTests(unittest.TestCase):
    """`report` is the judge the statuses rest on, pinned on its own."""

    def setUp(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        import qwen38_parity

        self._report = qwen38_parity.report

    def report(self, label, mine, reference) -> bool:
        # report() prints the line an operator reads; the transcript here is the
        # suite's, so the verdict is returned to the assertion, not echoed.
        with contextlib.redirect_stdout(io.StringIO()):
            return self._report(label, mine, reference)

    def test_matching_vectors_agree(self) -> None:
        import numpy as np

        vector = np.arange(64, dtype=np.float32) + 1.0
        self.assertTrue(self.report("pin", vector, vector))

    def test_differing_vectors_do_not(self) -> None:
        import numpy as np

        self.assertFalse(self.report("pin", np.ones(64, np.float32), np.arange(64) + 1.0))

    def test_a_vector_of_the_wrong_shape_does_not(self) -> None:
        import numpy as np

        self.assertFalse(self.report("pin", np.ones(64, np.float32), np.ones(32, np.float32)))


class HarnessWiringTests(unittest.TestCase):
    """Recurrence guard: a harness that prints a verdict has to carry one."""

    HARNESSES = (PARITY, FULL_FORWARD)

    def test_the_parity_scripts_exit_on_their_verdict(self) -> None:
        for script in self.HARNESSES:
            text = script.read_text(encoding="utf-8")
            self.assertIn("sys.exit(main())", text, f"{script.name} ends on a bare main()")

    def test_no_tools_script_prints_a_verdict_it_cannot_carry(self) -> None:
        offenders = []
        for module in sorted((ROOT / "tools").glob("*.py")):
            text = module.read_text(encoding="utf-8")
            prints_a_verdict = "'FAIL'" in text or '"FAIL"' in text
            carries_a_status = "sys.exit" in text or "raise SystemExit" in text
            if prints_a_verdict and not carries_a_status:
                offenders.append(module.name)
        self.assertEqual(offenders, [], f"scripts that print FAIL and always exit 0: {offenders}")


if __name__ == "__main__":
    unittest.main()
