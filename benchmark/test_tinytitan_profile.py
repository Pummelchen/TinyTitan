import os
import pathlib
import sys
import unittest
from unittest.mock import patch

import coder_cli_benchmark
import launcher_fixture
import tinytitan_benchmark
from tinytitan_profile import (
    DEFAULT_API_MODEL,
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_CONCISE,
    DEFAULT_EXPERT_CACHE_BUDGET,
    DEFAULT_KV_BITS,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPT_CACHE_MEMORY_MIB,
    DEFAULT_PROMPT_CACHE_MODE,
    catalog_id_for,
    request_model,
    server_command,
    server_environment,
    configured_thinking_mode,
)


class BenchmarkProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        launcher_fixture.install_fixture(cls)

    def installed_model(self) -> pathlib.Path:
        """An install the harness can name, from the fixture rather than `models/`.

        A command names an install by the catalog id read from that install's
        manifest, so the shape test needs one installed. A checkout is not
        required to hold any particular one -- the operator prunes `models/`
        deliberately, and a clean clone holds none -- so the synthetic tree stands
        in for it. The protocol's own defaults are asserted as constants instead
        of by reading them.
        """
        return pathlib.Path(self.installs.rows[0]["path"])

    def test_server_command_goes_through_the_launcher(self) -> None:
        model = self.installed_model()
        command = server_command("TinyTitanServer", 8081, model=model)
        # The harness never builds an TinyTitanServer command line itself: it asks
        # the user-facing launcher, so a benchmarked server is configured the
        # way a user's server is -- catalog resolution plus the model's own
        # measured profile -- and only ever from an install already on disk.
        self.assertEqual(command[0], "bash")
        self.assertTrue(command[1].endswith("tools/server_launcher.sh"))
        self.assertEqual(command[command.index("--client") + 1], "server")
        self.assertEqual(command[command.index("--model") + 1], catalog_id_for(model))
        # Not an identity: the manifest under the synthetic install reads back as
        # a catalog id that is one of the launcher's rows, so the harness names an
        # install the way the catalog does rather than echoing its own argument.
        self.assertIn(catalog_id_for(model), self.installs.ids)
        self.assertEqual(command[command.index("--port") + 1], "8081")
        self.assertEqual(command[command.index("--thinking") + 1], "off")
        self.assertEqual(command[command.index("--engine") + 1], "gpu")
        self.assertEqual(command[command.index("--kv") + 1], str(DEFAULT_KV_BITS))
        # Production profile: the launcher's own multi-prefix cache, no draft
        # head, and the model's measured expert-cache budget rather than a pin.
        self.assertNotIn("--prompt-cache", command)
        self.assertNotIn("--mtp-model", command)
        self.assertEqual("--ram" in command, DEFAULT_EXPERT_CACHE_BUDGET is not None)
        # The protocol defaults are constants, and Ornith 1.5 8-bit may
        # legitimately be absent from models/.
        self.assertEqual(DEFAULT_MODEL_PATH.name, "ornith-1.5_35B_A3B_8Bit")
        self.assertEqual(DEFAULT_CONTEXT_TOKENS, 262_144)
        self.assertEqual(DEFAULT_PROMPT_CACHE_MODE, "multi-prefix")
        self.assertEqual(DEFAULT_PROMPT_CACHE_MEMORY_MIB, 256)

    def test_explicit_cache_off_is_not_a_default(self) -> None:
        command = server_command(
            "TinyTitanServer", 8081, model=self.installed_model(), cache_mode="off"
        )
        self.assertEqual(command[command.index("--prompt-cache") + 1], "off")

    def test_shell_launchers_explicitly_enable_both_caches(self) -> None:
        launcher = (DEFAULT_MODEL_PATH.parents[1] / "tools/server_launcher.sh").read_text()
        # The launcher owns the prompt cache, and it can turn it off for the
        # cache A/B harnesses: 256 MiB multi-prefix by default, 0 when off.
        self.assertIn('prompt_cache_mode="multi-prefix"', launcher)
        self.assertIn(f"prompt_cache_mib={DEFAULT_PROMPT_CACHE_MEMORY_MIB}", launcher)
        self.assertIn("--prompt-cache) CACHE_ARG=", launcher)
        self.assertIn(
            'gpu_runtime=(--prompt-cache-mode "$prompt_cache_mode" '
            '--prompt-cache-memory-mib "$prompt_cache_mib"',
            launcher,
        )
        # MTP is a launcher flag too, so the speculative harnesses go through
        # the same script rather than building their own command line.
        self.assertIn("--mtp-model) MTP_MODEL_ARG=", launcher)
        self.assertIn(
            'gpu_runtime+=(--mtp-model "$MTP_MODEL_ARG" --mtp-memory-mib "${MTP_MEMORY_ARG:-384}")',
            launcher,
        )
        # The expert-cache budget is per-family (decodeTuning), so neither the
        # benchmark protocol nor a plain launcher run pins one size: the
        # runtime picks the model's own measured row. The launcher offers an
        # explicit `--ram` override, which must stay opt-in so the default path
        # keeps the measured optimum.
        self.assertIn('gpu_runtime+=(--ram-budget "${ram_gb}G")', launcher)
        self.assertIn('if [[ -n "$ram_gb" && "$MODEL_BACKEND" != "cpu" ]]', launcher)
        self.assertIn("${TINYTITAN_THINKING_MODE:-off}", launcher)
        # The dynamic path takes the reasoning level the catalog says the
        # model supports (`--reasoning`); the single-model fallback for a
        # binary-thinking build keeps the old `--thinking` flag.
        self.assertIn('--reasoning "$thinking_level"', launcher)
        self.assertIn(
            '--thinking "$( [[ "$thinking_level" == off ]] && echo off || echo on )"', launcher
        )
        # Model and quantization are one list now, and the server starts
        # in dynamic mode over the whole models directory; the first
        # question is what to launch -- the server alone, or the server plus
        # one coding client -- rather than a coding CLI. Pin both so a revert
        # to the old flow shows here.
        self.assertIn('--models-dir "$MODELS_DIR"', launcher)
        self.assertIn("What do you want to launch?", launcher)
        self.assertIn("Which answer style?", launcher)
        self.assertIn('case "${answers_choice:-1}"', launcher)

    def test_environment_and_model_select_standard_base_alias(self) -> None:
        environment = server_environment({"PATH": os.environ.get("PATH", "")})
        self.assertFalse(DEFAULT_CONCISE)
        self.assertNotIn("TINYTITAN_CONCISE_MODE", environment)
        self.assertEqual(environment["TINYTITAN_THINKING_MODE"], "off")
        self.assertEqual(request_model(), DEFAULT_API_MODEL)

    def test_thinking_mode_is_binary_and_configurable(self) -> None:
        self.assertEqual(configured_thinking_mode({}), "off")
        self.assertEqual(configured_thinking_mode({"TINYTITAN_THINKING_MODE": "yes"}), "on")
        with self.assertRaises(ValueError):
            configured_thinking_mode({"TINYTITAN_THINKING_MODE": "medium"})
        command = server_command("server", 8080, model=self.installed_model(), thinking_mode="on")
        self.assertEqual(command[command.index("--thinking") + 1], "on")
        self.assertFalse(request_model().endswith("-fast"))

    def test_coder_harness_plain_invocation_uses_ornith_eight_bit(self) -> None:
        with patch.object(sys, "argv", ["coder_cli_benchmark.py"]):
            arguments = coder_cli_benchmark.parse_args()
        self.assertEqual(arguments.round, "coder")
        self.assertEqual(arguments.quantizations, [8])

    def test_precise_benchmark_plain_invocation_excludes_mtp_matrix(self) -> None:
        configurations = tinytitan_benchmark.selected_configs("8bit")
        self.assertEqual(
            configurations,
            [("multi-prefix", "off", "cache_on_mtp_off_8bit", 8081, 0.6)],
        )


if __name__ == "__main__":
    unittest.main()
