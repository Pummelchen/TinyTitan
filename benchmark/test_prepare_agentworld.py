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

import pathlib
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

        baseline = threading.active_count()

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
        while threading.active_count() > baseline and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(threading.active_count(), baseline)

    def test_the_pool_still_fetches_every_shard_in_the_normal_case(self):
        def fetch(shard):
            return shard

        self.assertEqual(
            sorted(prepare.prefetch_shards(["a", "b", "c", "d"], fetch, fetchers=2, depth=2)),
            ["a", "b", "c", "d"],
        )


if __name__ == "__main__":
    unittest.main()
