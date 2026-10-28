#!/usr/bin/env python3
"""Tests for the Qwen3.5-MoE converter, and the per-expert fusion at two widths.

`FusedExperts` stacks a per-expert checkpoint (KAT-Coder-V2.5-Dev: 30,720
`mlp.experts.<E>.{gate,up,down}_proj` tensors) into the fused `switch_mlp`
spelling the repacker and runtime read. Two things about it failed in the
field, and both are pinned here:

- **The duplicate guard was blind to the width.** `convert_shard` adds one
  source tensor once per requested width, and `tools/install_models.sh` always
  converts this family with `--bits 4 8`, so the guard's name-only key made
  the second width look like a repeated tensor. Every katcoder install died on
  the first routed expert of the first shard that carried one, with the
  message in issue #19. The regression test walks the real call site
  (`convert_shard` over a synthetic safetensors shard), not just `add`.
- **Expert placement is by index, never by arrival.** A per-expert checkpoint
  does not promise that expert k arrives before expert j, and appending would
  put one expert's bytes in another's slot: every shape and every byte check
  would still pass, and the model would route to one expert while reading
  another's weights. The ordering is asserted, so it is tested.

The guard still has to fire on a genuine repeat (the same tensor twice at the
same width), which is the reason it exists; that is tested too.

    cd benchmark && python3.13 -m unittest test_prepare_agentworld -v

It imports the converter, which imports numpy, ml_dtypes and safetensors, so it
skips where those are absent.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

try:
    import prepare_agentworld as prepare

    IMPORT_ERROR = ""
except SystemExit as exc:  # the module exits when a dependency is missing
    prepare = None
    IMPORT_ERROR = str(exc)


class Recorder:
    """The writer surface `FusedExperts.release` uses, without any IO."""

    def __init__(self) -> None:
        self.added: list[tuple[str, tuple]] = []
        self.flushes = 0

    def add(self, name: str, value) -> None:
        self.added.append((name, tuple(value.shape)))

    def flush(self) -> None:
        self.flushes += 1


EXPERT = "model.language_model.layers.6.mlp.experts.{}.down_proj.weight"
FUSED = "language_model.model.layers.6.mlp.switch_mlp.down_proj.weight"


@unittest.skipIf(prepare is None, f"prepare_agentworld unavailable: {IMPORT_ERROR}")
class PerExpertNamingTests(unittest.TestCase):
    def test_kats_failing_name_is_recognised_as_a_per_expert_routed_weight(self):
        target, expert, role = prepare.per_expert_routed(prepare.rename(EXPERT.format(0)))
        self.assertEqual(target, FUSED)
        self.assertEqual(expert, 0)
        self.assertEqual(role, "down_proj")

    def test_a_fused_name_is_not_taken_for_a_per_expert_one(self):
        self.assertIsNone(
            prepare.per_expert_routed(
                prepare.rename("model.language_model.layers.6.mlp.experts.down_proj.weight")
            )
        )


@unittest.skipIf(prepare is None, f"prepare_agentworld unavailable: {IMPORT_ERROR}")
class FusedExpertsBothWidthsTests(unittest.TestCase):
    def test_one_tensor_adds_at_every_width(self):
        """Issue #19: the second width is not a duplicate."""
        import numpy as np

        fused = prepare.FusedExperts()
        for width in (4, 8):
            fused.add(EXPERT.format(0), np.full((64, 64), 7, dtype=np.float32), width, None)
        self.assertEqual({width for width, _ in fused._layers}, {4, 8})

    def test_a_repeat_at_one_width_is_still_a_duplicate(self):
        import numpy as np

        fused = prepare.FusedExperts()
        value = np.full((64, 64), 7, dtype=np.float32)
        fused.add(EXPERT.format(0), value, 4, None)
        with self.assertRaisesRegex(ValueError, "duplicate source tensor"):
            fused.add(EXPERT.format(0), value, 4, None)

    def test_experts_land_at_their_own_index_not_arrival_order(self):
        import numpy as np

        fused = prepare.FusedExperts()
        for expert in (2, 0, 1):  # arrival order is not expert order
            fused.add(EXPERT.format(expert), np.full((64, 64), expert, dtype=np.float32), 4, None)
        stack = fused._layers[(4, FUSED)]["stack"]
        self.assertEqual(stack.shape[0], 3)
        for expert in range(3):
            self.assertEqual(float(stack[expert][0, 0]), float(expert))

    def test_release_emits_a_fused_tensor_per_width(self):
        import numpy as np

        fused = prepare.FusedExperts()
        for expert in (0, 1):
            for width in (4, 8):
                fused.add(
                    EXPERT.format(expert), np.full((64, 64), expert, dtype=np.float32), width, None
                )
        writers = {4: Recorder(), 8: Recorder()}
        fused.release(experts_per_layer=2, writers=writers)
        for writer in writers.values():
            names = [name for name, _ in writer.added]
            self.assertEqual(
                names,
                [FUSED, FUSED.replace(".weight", ".scales"), FUSED.replace(".weight", ".biases")],
            )
            self.assertEqual(writer.added[0][1][0], 2)  # the expert axis
        four, eight = (dict(writers[w].added) for w in (4, 8))
        self.assertNotEqual(four[FUSED], eight[FUSED])  # 4-bit packs narrower than 8-bit
        self.assertEqual(fused.pending(), [])

    def test_an_incomplete_layer_is_reported_rather_than_released(self):
        import numpy as np

        fused = prepare.FusedExperts()
        fused.add(EXPERT.format(0), np.full((64, 64), 0, dtype=np.float32), 4, None)
        writers = {4: Recorder()}
        fused.release(experts_per_layer=2, writers=writers)
        self.assertEqual(writers[4].added, [])
        self.assertEqual(len(fused.pending()), 1)


@unittest.skipIf(prepare is None, f"prepare_agentworld unavailable: {IMPORT_ERROR}")
class ConvertShardBothWidthsTests(unittest.TestCase):
    """The regression, through the caller that failed: `convert_shard`."""

    def test_a_per_expert_shard_converts_at_both_widths(self):
        import numpy as np
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmp:
            shard = pathlib.Path(tmp) / "model-00001-of-00013.safetensors"
            save_file({EXPERT.format(0): np.full((64, 64), 3, dtype=np.float32)}, str(shard))
            writers = {4: Recorder(), 8: Recorder()}
            prepare.convert_shard(shard, writers, prepare.FusedExperts(), experts_per_layer=1)
        for width, writer in writers.items():
            self.assertEqual(
                [name for name, _ in writer.added],
                [FUSED, FUSED.replace(".weight", ".scales"), FUSED.replace(".weight", ".biases")],
                f"{width}-bit",
            )


@unittest.skipIf(prepare is None, f"prepare_agentworld unavailable: {IMPORT_ERROR}")
class PrefetchShutdownTests(unittest.TestCase):
    """A SIGTERM during the download must not leave a daemon fetcher alive.

    Observed in the field as a macOS crash report: the fetchers were daemon
    threads left blocked on the ready queue (nobody drains it once the caller
    raises), and CPython 3.14 turns a daemon thread writing stdout at
    interpreter finalization into a fatal error that calls `abort()`.
    """

    def test_closing_the_generator_leaves_no_fetcher_thread_running(self):
        import threading
        import time

        # The instrument is the set of threads alive before the pool starts,
        # not a process-global count. A count fails whenever a foreign thread
        # dies inside the window -- on CI the two suites that run before this
        # one in the same process leave one behind (run 38130598551 failed
        # `8 != 9`, actual BELOW baseline, which a leaked fetcher cannot
        # produce). Only this generator creates threads in the window, so a
        # thread outside `before` after close() is a fetcher that survived it.
        before = set(threading.enumerate())

        def fetch(shard):
            time.sleep(0.05)
            return shard

        generator = prepare.prefetch_shards(
            [f"shard-{n}" for n in range(6)], fetch, fetchers=3, depth=1
        )
        self.assertIsInstance(next(generator), str)  # one consumed, the rest queue up
        time.sleep(0.4)  # let the other fetchers fill the one-slot queue and wait
        generator.close()
        deadline = time.time() + 3
        while set(threading.enumerate()) - before and time.time() < deadline:
            time.sleep(0.05)
        leaked = sorted(t.name for t in set(threading.enumerate()) - before)
        self.assertEqual([], leaked)

    def test_the_pool_still_fetches_every_shard_in_the_normal_case(self):
        def fetch(shard):
            return shard

        self.assertEqual(
            sorted(prepare.prefetch_shards(["a", "b", "c", "d"], fetch, fetchers=2, depth=2)),
            ["a", "b", "c", "d"],
        )


@unittest.skipIf(prepare is None, f"prepare_agentworld unavailable: {IMPORT_ERROR}")
class DownloadStopTests(unittest.TestCase):
    """The download itself has to notice the stop, not just the queue.

    Joining the fetchers is worthless while one of them sits in a chunk retry
    that can wait two minutes: the join times out, the interpreter finalizes
    with the thread alive, and the abort comes back.
    """

    def setUp(self):
        prepare._stopping.clear()

    def tearDown(self):
        prepare._stopping.clear()

    def test_a_shard_is_not_started_when_the_run_is_stopping(self):
        prepare._stopping.set()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(InterruptedError):
                prepare.download("model-00000-of-00013.safetensors", pathlib.Path(tmp))

    def test_a_stop_during_a_retry_wait_ends_the_download(self):
        from unittest import mock

        def fake_popen(*_args, **_kwargs):
            proc = mock.MagicMock()
            proc.wait.return_value = 1  # curl failed
            return proc

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(prepare, "expected_size", return_value=1024),
                mock.patch.object(prepare, "resolve_url", return_value="http://example.invalid/x"),
                mock.patch.object(prepare.subprocess, "Popen", side_effect=fake_popen),
                mock.patch.object(prepare, "_sleep_unless_stopped", return_value=True),
            ):
                with self.assertRaises(InterruptedError):
                    prepare.download("model-00000-of-00013.safetensors", pathlib.Path(tmp))


@unittest.skipIf(prepare is None, f"prepare_agentworld unavailable: {IMPORT_ERROR}")
class DenseBitsTests(unittest.TestCase):
    """`--dense-bits` splits the always-on blocks from the routed experts.

    The mixed build is what the 35B catalogue row ships: attention, GDN and the
    shared expert at the checkpoint's own bf16, routed experts at the build
    width. The split has to be exactly the dense roles -- a routed expert
    promoted to bf16 would triple the install, and a dense projection left at
    4-bit would make the row a lie about what it runs.
    """

    ATTN = "language_model.model.layers.6.self_attn.q_proj.weight"
    GDN = "language_model.model.layers.6.linear_attn.in_proj_qkv.weight"
    GDN_OUT = "language_model.model.layers.6.linear_attn.out_proj.weight"
    SHARED = "language_model.model.layers.6.mlp.shared_expert.down_proj.weight"
    ROUTED = "language_model.model.layers.6.mlp.switch_mlp.gate_proj.weight"
    ROUTER = "language_model.model.layers.6.mlp.gate.weight"
    HEAD = "language_model.lm_head.weight"
    NORM = "language_model.model.layers.6.input_layernorm.weight"

    def setUp(self):
        self.saved = prepare.DENSE_BITS

    def tearDown(self):
        prepare.DENSE_BITS = self.saved

    def test_16_copies_the_dense_blocks_through_at_bf16(self):
        prepare.DENSE_BITS = 16
        for name in (self.ATTN, self.GDN, self.GDN_OUT, self.SHARED):
            self.assertIsNone(prepare.quant_bits(name, 4), name)

    def test_16_leaves_the_experts_and_the_head_at_their_own_widths(self):
        prepare.DENSE_BITS = 16
        self.assertEqual(prepare.quant_bits(self.ROUTED, 4), 4)
        self.assertEqual(prepare.quant_bits(self.HEAD, 4), prepare.HEAD_BITS)
        # The router and every norm were already bf16, which is why `None` here
        # is the same answer as before rather than a new promotion.
        self.assertIsNone(prepare.quant_bits(self.ROUTER, 4))
        self.assertIsNone(prepare.quant_bits(self.NORM, 4))

    def test_8_quantizes_the_dense_blocks_but_not_the_experts(self):
        prepare.DENSE_BITS = 8
        for name in (self.ATTN, self.GDN, self.GDN_OUT, self.SHARED):
            self.assertEqual(prepare.quant_bits(name, 4), 8, name)
        self.assertEqual(prepare.quant_bits(self.ROUTED, 4), 4)

    def test_the_config_names_the_head_but_no_width_for_a_bf16_tensor(self):
        # The repacker classifies each tensor by its own dtype; the config's
        # `bits` is the slot width and every override is a tensor that really
        # is packed at that width. Writing an override for a bf16 tensor would
        # claim a packing that is not on disk, so 16 is expressed by absence.
        prepare.DENSE_BITS = 16
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            prepare.write_config(
                {"model_type": "qwen3_5_moe", "architectures": ["Qwen3_5MoeForCausalLM"]},
                out,
                [self.ATTN, self.SHARED, self.HEAD],
                4,
            )
            written = json.loads((out / "config.json").read_text())
        quant = written["quantization"]
        self.assertEqual(quant["bits"], 4)
        self.assertEqual(quant["language_model.lm_head"]["bits"], prepare.HEAD_BITS)
        self.assertNotIn(self.ATTN[: -len(".weight")], quant)
        self.assertNotIn(self.SHARED[: -len(".weight")], quant)

    def test_unset_keeps_the_single_width_rule(self):
        prepare.DENSE_BITS = None
        for name in (self.ATTN, self.GDN, self.SHARED, self.ROUTED):
            self.assertEqual(prepare.quant_bits(name, 4), 4, name)

    def test_two_widths_are_refused_before_anything_is_fetched(self):
        # One dense policy cannot describe two snapshots, and refusing after a
        # 70 GB fetch would waste the run. The parse is where it has to fail.
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/prepare_agentworld.py"),
                    "--model",
                    "ornith15",
                    "--bits",
                    "4",
                    "8",
                    "--dense-bits",
                    "16",
                    "--output",
                    str(out),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("--dense-bits names one configuration", result.stderr)
            self.assertFalse(out.exists(), "the refused run wrote an output directory")


if __name__ == "__main__":
    unittest.main()
