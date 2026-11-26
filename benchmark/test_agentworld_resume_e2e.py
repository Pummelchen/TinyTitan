#!/usr/bin/env python3
"""End-to-end tests for a killed Qwen3.5-MoE conversion and its resume.

`test_prepare_agentworld.py` pins the pieces. This file runs the whole
converter (`main()`) against a synthetic but structurally real checkpoint served
over a real HTTP server on 127.0.0.1, over real `curl` downloads into a test
folder, and kills it with a real `SIGKILL` at the one moment that matters:

    the output block has not flushed yet.

`convert_shard` returns after `OutputWriter.add`, and `add` only writes a shard
at `OUTPUT_SHARD_BYTES` (4 GiB); a per-expert checkpoint's experts also sit in
`FusedExperts` until their whole layer has arrived. The converter used to
`unlink` each source shard as `convert_shard` returned, so a kill before that
flush lost every converted byte while the source was already gone: the resume
re-downloaded every shard since the last flush, and on a 250 GB checkpoint that
is most of a fetch.

The kill is deterministic and sleep-free: the child process stops *itself* with
`SIGSTOP` at the conversion boundary (after main's loop has had its chance to
unlink the previous source, before the next conversion), the parent blocks in
`os.waitpid(..., WUNTRACED)` until it has actually stopped, inspects the
directory, then sends `SIGKILL`. One fetcher (`TINYTITAN_FETCHERS=1`) keeps the
conversion order equal to the shard order so the boundary is predictable.

Cases:

- the paired `--bits 4 8` path (both writers must have flushed, not just one),
- the per-expert checkpoint path, where a source must stay on disk even after
  the writers flushed, because its experts are still buffered in `FusedExperts`,
- the solo `--bits 4` path the 122B install uses (one writer; the same
  ownership rule -- there is no separate rule to document).

Every case ends on the snapshot a clean run produces, so the recovery may do
work again but never change the bytes.

    cd benchmark && python3.13 -m unittest test_agentworld_resume_e2e -v

It needs `numpy`, `ml_dtypes`, `safetensors` and the system `curl`, and it
never contacts the network: every URL it fetches points at 127.0.0.1.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import io
import json
import os
import pathlib
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.parse
import zlib
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

try:
    import ml_dtypes
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file
except ImportError as exc:  # pragma: no cover - environment, not logic
    ml_dtypes = np = safe_open = save_file = None
    DEPS_ERROR = str(exc)
else:
    DEPS_ERROR = ""

try:
    import prepare_agentworld as prepare

    PREPARE_ERROR = ""
except SystemExit as exc:  # the module exits when a dependency is missing
    prepare = None
    PREPARE_ERROR = str(exc)


def requires_deps(obj):
    reason = DEPS_ERROR or PREPARE_ERROR
    return unittest.skipIf(reason, f"converter or its dependencies unavailable: {reason}")(obj)


# --- the synthetic checkpoints ---------------------------------------------


def _float(tag: str, shape) -> "np.ndarray":
    rng = np.random.default_rng(zlib.crc32(tag.encode()) or 1)
    return (rng.random(shape, dtype=np.float32) * 2.0 - 1.0).astype(np.float32)


def expert(layer: int, index: int, role: str, tag: str) -> tuple[str, "np.ndarray"]:
    name = f"model.language_model.layers.{layer}.mlp.experts.{index}.{role}.weight"
    return name, _float(tag, (64, 64))


def fused_plan() -> dict[str, dict[str, "np.ndarray"]]:
    """AgentWorld/Ornith's shape: routed experts fused per layer.

    Four shards, one tensor group each, so the converted bytes per shard are
    small enough to keep in memory and the output flush threshold can be set
    past all of them -- which is exactly the kill-before-the-first-flush state.
    """
    gate_up = _float("gateup", (4, 128, 64))
    return {
        "model-00001-of-00004.safetensors": {
            "model.language_model.layers.0.input_layernorm.weight": _float("inln0", (64,)),
            "model.language_model.layers.0.mlp.gate.weight": _float("gate0", (8, 64)),
        },
        "model-00002-of-00004.safetensors": {
            "model.language_model.layers.1.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.1.mlp.experts.down_proj": _float("down1", (4, 64, 64)),
        },
        "model-00003-of-00004.safetensors": {
            "model.language_model.embed_tokens.weight": _float("embed", (128, 64)),
            "model.language_model.layers.1.self_attn.q_proj.weight": _float("q1", (64, 64)),
        },
        "model-00004-of-00004.safetensors": {
            "model.language_model.layers.1.mlp.shared_expert.down_proj.weight": _float(
                "shared1", (64, 64)
            ),
            "lm_head.weight": _float("head", (128, 64)),
        },
    }


def per_expert_plan() -> dict[str, dict[str, "np.ndarray"]]:
    """KAT-Coder's shape, arranged around the layer boundary.

    Shard 1 completes layer 0 (both experts, all three roles) *and* supplies
    layer 1's first expert. So after shard 1 `FusedExperts.release` emits layer
    0 and flushes both writers -- `writer.bytes == 0`, the durable point -- while
    layer 1's experts are still in memory. The shard therefore may not be
    deleted even though the writers flushed: that is the fused half of the rule,
    and a writer-only fix loses it.
    """
    return {
        "model-00001-of-00004.safetensors": dict(
            [
                expert(0, 0, "gate_proj", "g00"),
                expert(0, 0, "up_proj", "u00"),
                expert(0, 0, "down_proj", "d00"),
                expert(0, 1, "gate_proj", "g01"),
                expert(0, 1, "up_proj", "u01"),
                expert(0, 1, "down_proj", "d01"),
                expert(1, 0, "gate_proj", "g10"),
                expert(1, 0, "up_proj", "u10"),
                expert(1, 0, "down_proj", "d10"),
            ]
        ),
        "model-00002-of-00004.safetensors": dict(
            [
                expert(1, 1, "gate_proj", "g11"),
                expert(1, 1, "up_proj", "u11"),
                expert(1, 1, "down_proj", "d11"),
                ("model.language_model.layers.1.input_layernorm.weight", _float("inln1", (64,))),
            ]
        ),
        "model-00003-of-00004.safetensors": dict(
            [
                expert(2, 0, "gate_proj", "g20"),
                expert(2, 0, "up_proj", "u20"),
                expert(2, 0, "down_proj", "d20"),
                expert(2, 1, "gate_proj", "g21"),
                expert(2, 1, "up_proj", "u21"),
                expert(2, 1, "down_proj", "d21"),
                ("model.language_model.embed_tokens.weight", _float("embed", (128, 64))),
            ]
        ),
        "model-00004-of-00004.safetensors": {
            "lm_head.weight": _float("head", (128, 64)),
        },
    }


TOKENIZER_BODY = b"{}  // served by the test mirror\n"


def checkpoint_files(shards: dict[str, dict[str, "np.ndarray"]], num_experts: int):
    """Every file the mirror serves, plus the plan behind them."""
    stage = pathlib.Path(tempfile.mkdtemp(prefix="tt-agentworld-checkpoint-"))
    try:
        files: dict[str, bytes] = {}
        weight_map: dict[str, str] = {}
        total = 0
        for shard, tensors in shards.items():
            path = stage / shard
            save_file(tensors, str(path))
            files[shard] = path.read_bytes()
            for name, value in tensors.items():
                weight_map[name] = shard
                total += value.nbytes
        files["model.safetensors.index.json"] = json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1
        ).encode()
        files["config.json"] = json.dumps(
            {
                "architectures": ["Qwen3_5MoeForCausalLM"],
                "model_type": "qwen3_5_moe",
                "text_config": {"num_experts": num_experts, "hidden_size": 64},
            },
            indent=1,
        ).encode()
        for name, _required in prepare.TOKENIZER_FILES:
            files[name] = TOKENIZER_BODY
        return files, shards
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def header_bytes(shard_bytes: bytes) -> int:
    """The safetensors header length, which bounds the size-probe range."""
    return struct.unpack("<Q", shard_bytes[:8])[0]


# --- the mirror -------------------------------------------------------------


class Mirror(http.server.ThreadingHTTPServer):
    """A Hub-shaped mirror: ranged GETs, HEAD, and what it was asked for."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, files: dict[str, bytes]):
        super().__init__(("127.0.0.1", 0), _MirrorHandler)
        self.files = files
        self.requests: list[dict] = []
        self.lock = threading.Lock()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/agentworld"

    def record(self, entry: dict) -> None:
        with self.lock:
            self.requests.append(entry)

    def requests_for(self, name: str) -> list[dict]:
        with self.lock:
            return [r for r in self.requests if r["file"] == name]

    def handle_error(self, *_args) -> None:
        # A transfer the killed converter left behind resets its connection;
        # that is the state under test, not a traceback.
        pass


class _MirrorHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AgentWorldMirror/1"

    def log_message(self, *_args) -> None:  # keep the test output readable
        pass

    def _name(self) -> str | None:
        path = urllib.parse.urlparse(self.path).path
        return path.rsplit("/", 1)[-1] or None

    def do_HEAD(self) -> None:  # noqa: N802 - http.server's spelling
        # A size probe, not a fetch: only GETs are recorded, so a complete
        # source can be proved to have moved no body bytes.
        data = self.server.files.get(self._name() or "")
        if data is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
        name = self._name() or ""
        data = self.server.files.get(name)
        if data is None:
            self.send_error(404)
            return
        header = self.headers.get("Range")
        span = None
        if header and header.startswith("bytes="):
            first, _, last = header[len("bytes=") :].split(",")[0].partition("-")
            start = int(first) if first else 0
            end = int(last) if last else len(data) - 1
            span = (start, min(end, len(data) - 1))
        self.server.record({"file": name, "range": span})
        body = data[span[0] : span[1] + 1] if span else data
        self.send_response(206 if span else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if span:
            self.send_header("Content-Range", f"bytes {span[0]}-{span[1]}/{len(data)}")
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def mirror(files: dict[str, bytes]):
    server = Mirror(files)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def body_requests(server: Mirror, name: str, files: dict[str, bytes], since: int = 0) -> list[dict]:
    """The requests that moved payload bytes for `name`, after position `since`.

    The size probe reads the 8-byte length and then the header, so a ranged
    request no longer than the header region is not a body fetch. Anything
    larger -- or un-ranged -- is. `since` is an index into the *mirror's* whole
    request log, taken before the run under test: filtering the per-file list at
    a global index, as this first did, returns an empty slice and makes every
    "not fetched again" assertion pass while asserting nothing.
    """
    limit = 8 + header_bytes(files[name])
    with server.lock:
        window = list(server.requests[since:])
    return [
        request
        for request in window
        if request["file"] == name
        and (request["range"] is None or request["range"][1] - request["range"][0] + 1 > limit)
    ]


# --- running the converter --------------------------------------------------


def run_in_process(
    out: pathlib.Path,
    work: pathlib.Path,
    base: str,
    extra: tuple = (),
    patches: dict | None = None,
) -> tuple[int, str]:
    """`main()` in this process, against the mirror, with output captured."""
    argv = [
        "prepare_agentworld.py",
        "--output",
        str(out),
        "--work",
        str(work),
        *[str(item) for item in extra],
    ]
    saved_base = prepare.BASE
    saved_select = prepare.select_model
    stopping_was_set = prepare._stopping.is_set()
    prepare.expected_size.cache_clear()  # the two fixtures share shard names
    prepare.select_model = lambda _name: None  # keep the mirror's BASE
    prepare.BASE = base
    buffer = io.StringIO()
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.dict(os.environ, {"TINYTITAN_NO_PROGRESS": "1"}),
        redirect_stdout(buffer),
        redirect_stderr(buffer),
    ):
        stack = contextlib.ExitStack()
        try:
            for name, value in (patches or {}).items():
                stack.enter_context(mock.patch.object(prepare, name, value))
            try:
                code = prepare.main()
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                print(f"SystemExit: {exc}", file=buffer)
        except BaseException:  # noqa: BLE001 - reported through the log
            import traceback

            code = 1
            traceback.print_exc(file=buffer)
        finally:
            stack.close()
            prepare.BASE = saved_base
            prepare.select_model = saved_select
            # `prefetch_shards` sets this in its `finally` so a shutdown stops
            # the fetchers, and `download` refuses to start a shard while it is
            # set. One conversion per process is the product's model, so the
            # flag never matters after `main` returns there -- but this process
            # runs more suites, and inheriting it made
            # `test_prepare_agentworld`'s direct `download` tests fail when it
            # ran second. Restore what the caller had.
            if not stopping_was_set:
                prepare._stopping.clear()
    return code, buffer.getvalue()


def driver_script(
    directory: pathlib.Path,
    base: str,
    *,
    marker: pathlib.Path | None = None,
    stop_after: int | None = None,
    output_shard_bytes: int | None = None,
) -> pathlib.Path:
    """A converter process that stops itself before conversion `stop_after+1`.

    `SIGSTOP` is the synchronization: a stopped process is not a guess about
    timing, and the parent's `waitpid(WUNTRACED)` returns exactly when the
    boundary is reached. The stop is taken *before* the next `convert_shard`,
    which is after the loop body of the previous one -- the moment the old code
    had already unlinked that source.
    """
    path = directory / "driver.py"
    path.write_text(
        "import os, signal, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT / 'tools')!r})\n"
        "import prepare_agentworld as prepare\n"
        f"BASE = {base!r}\n"
        f"MARKER = Path({str(marker)!r}) if {marker is not None!r} else None\n"
        f"prepare.OUTPUT_SHARD_BYTES = {output_shard_bytes!r} or prepare.OUTPUT_SHARD_BYTES\n"
        "\n"
        "def local_select(name):\n"
        "    prepare.BASE = BASE\n"
        "\n"
        "prepare.select_model = local_select\n"
        f"STOP_AFTER = {stop_after!r}\n"
        "converted = []\n"
        "if STOP_AFTER is not None:\n"
        "    real_convert = prepare.convert_shard\n"
        "\n"
        "    def stopping_convert(path, writers, fused, experts):\n"
        "        if len(converted) == STOP_AFTER:\n"
        "            MARKER.write_text('\\n'.join(str(p) for p in converted))\n"
        "            os.kill(os.getpid(), signal.SIGSTOP)\n"
        "        real_convert(path, writers, fused, experts)\n"
        "        converted.append(path)\n"
        "\n"
        "    prepare.convert_shard = stopping_convert\n"
        "\n"
        "raise SystemExit(prepare.main())\n"
    )
    return path


def converter_env() -> dict[str, str]:
    """One fetcher, so conversion order equals shard order; quiet output."""
    env = dict(os.environ)
    env.update(
        {
            "TINYTITAN_NO_PROGRESS": "1",
            "TINYTITAN_FETCHERS": "1",
            "TINYTITAN_PREFETCH_DEPTH": "2",
        }
    )
    return env


class KilledRun:
    """The state a `SIGKILL` at the conversion boundary leaves behind."""

    def __init__(self, root: pathlib.Path, driver: pathlib.Path, argv, log_path: pathlib.Path):
        self.root = root
        self.driver = driver
        self.argv = argv
        self.log_path = log_path
        self.converted: list[pathlib.Path] = []
        self.converted_at_stop: list[str] = []
        self.returncode: int | None = None


def kill_at_conversion_boundary(
    root: pathlib.Path,
    base: str,
    argv: list[str],
    *,
    stop_after: int,
    output_shard_bytes: int | None = None,
    timeout: float = 120.0,
) -> KilledRun:
    """Run the converter and `SIGKILL` it once `stop_after` shards converted.

    The child stops itself at the boundary; the parent waits for the stop with
    `waitpid(WUNTRACED)` (no sleep, no polling), records what the directory
    holds, and only then kills it -- a kill, not a stop, because the field
    failure is `SIGKILL` and a stopped process could still be resumed.
    """
    marker = root / "converted.txt"
    log_path = root / "killed.log"
    driver = driver_script(
        root, base, marker=marker, stop_after=stop_after, output_shard_bytes=output_shard_bytes
    )
    run = KilledRun(root, driver, argv, log_path)
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            [sys.executable, str(driver), *argv],
            env=converter_env(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        stopped: dict = {}

        def reap_stop() -> None:
            stopped["status"] = os.waitpid(process.pid, os.WUNTRACED)

        watcher = threading.Thread(target=reap_stop, daemon=True)
        watcher.start()
        watcher.join(timeout)
        if watcher.is_alive():
            process.kill()
            os.waitpid(process.pid, 0)
            raise AssertionError(
                "the converter never reached the stop point:\n" + log_path.read_text()
            )
        _pid, status = stopped["status"]
        if os.WIFSTOPPED(status):
            run.converted_at_stop = marker.read_text().splitlines() if marker.exists() else []
            os.kill(process.pid, signal.SIGKILL)
            _pid, status = os.waitpid(process.pid, 0)
            process.returncode = -signal.SIGKILL if os.WIFSIGNALED(status) else -1
            run.returncode = process.returncode
        else:
            # It finished on its own: that is not this case, and the caller's
            # assertions about a kill would be meaningless.
            process.returncode = os.waitstatus_to_exitcode(status)
            run.returncode = process.returncode
            raise AssertionError(
                "the conversion finished before it could be killed "
                f"(exit {run.returncode}):\n" + log_path.read_text()
            )
    run.converted = [pathlib.Path(item) for item in run.converted_at_stop]
    return run


def run_resume(
    root: pathlib.Path, base: str, argv: list[str], timeout: int = 300
) -> tuple[int, str]:
    """A fresh process for the resume, so no size cache hides a re-fetch."""
    driver = driver_script(root, base)
    result = subprocess.run(
        [sys.executable, str(driver), *argv],
        env=converter_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


def fingerprint(out: pathlib.Path) -> dict:
    """Everything a loader reads about a snapshot, content-only."""
    index = json.loads((out / "model.safetensors.index.json").read_text())
    tensors = {}
    for name, shard in index["weight_map"].items():
        with safe_open(out / shard, framework="np") as handle:
            value = handle.get_tensor(name)
        tensors[name] = (
            str(value.dtype),
            tuple(value.shape),
            hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        )
    return {"tensors": tensors, "total_size": index["metadata"]["total_size"]}


_BASELINES: dict[tuple[str, tuple], dict] = {}


def baseline(fixture: str, bits: tuple) -> dict:
    """The snapshot a clean run of the same checkpoint produces, cached."""
    key = (fixture, bits)
    if key not in _BASELINES:
        files, _plan = (
            checkpoint_files(fused_plan(), 8)
            if fixture == "fused"
            else checkpoint_files(per_expert_plan(), 2)
        )
        root = pathlib.Path(tempfile.mkdtemp(prefix="tt-agentworld-baseline-"))
        try:
            with mirror(files) as server:
                code, log = run_in_process(
                    root / "out", root / "work", server.base, extra=("--bits", *bits)
                )
            if code != 0:
                raise AssertionError(f"the clean baseline run failed:\n{log}")
            _BASELINES[key] = {
                width: fingerprint(
                    root / "out" if len(bits) == 1 else pathlib.Path(f"{root / 'out'}-{width}bit")
                )
                for width in bits
            }
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return _BASELINES[key]


def dequantise(packed, scales, biases, bits: int, group: int = 64):
    """Unpack an affine tensor back to float, independently of the converter."""
    lanes = 32 // bits
    mask = np.uint32((1 << bits) - 1)
    shifts = np.arange(lanes, dtype=np.uint32) * np.uint32(bits)
    values = (packed.astype(np.uint32)[..., None] >> shifts) & mask
    values = values.reshape(*packed.shape[:-1], packed.shape[-1] * lanes)
    flat = values.astype(np.float32).reshape(*values.shape[:-1], values.shape[-1] // group, group)
    scaled = flat * scales.astype(np.float32)[..., None] + biases.astype(np.float32)[..., None]
    return scaled.reshape(values.shape)


def read_tensor(out: pathlib.Path, name: str):
    index = json.loads((out / "model.safetensors.index.json").read_text())
    with safe_open(out / index["weight_map"][name], framework="np") as handle:
        return handle.get_tensor(name)


# --- cases ------------------------------------------------------------------


class RestoresModuleState(unittest.TestCase):
    """Leave the converter's stop flag clear for whatever runs next.

    `prefetch_shards` sets `_stopping` in its `finally` -- that is how a
    shutdown stops the fetchers -- and `download` refuses to start a shard while
    it is set. The product runs one conversion per process, so the flag never
    matters after `main` returns; this process runs several suites, and one that
    inherits the flag makes an unrelated `download` unit test fail. Every case
    therefore starts and ends with it clear, and a case that leaves it set fails
    here rather than in the next suite.
    """

    def setUp(self) -> None:
        prepare._stopping.clear()

    def tearDown(self) -> None:
        self.assertFalse(
            prepare._stopping.is_set(),
            "the end-to-end harness left prepare._stopping set for the next suite",
        )
        prepare._stopping.clear()


@requires_deps
class SnapshotContractTests(RestoresModuleState):
    """One clean run: a valid snapshot, and the round trip still holds."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.files, cls.plan = checkpoint_files(fused_plan(), 8)
        cls.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-agentworld-contract-"))
        cls.server_ctx = mirror(cls.files)
        cls.server = cls.server_ctx.__enter__()
        cls.code, cls.log = run_in_process(
            cls.root / "out", cls.root / "work", cls.server.base, extra=("--bits", "4", "8")
        )
        cls.out = {width: pathlib.Path(f"{cls.root / 'out'}-{width}bit") for width in (4, 8)}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server_ctx.__exit__(None, None, None)
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_the_run_succeeds(self) -> None:
        self.assertEqual(self.code, 0, self.log)

    def test_each_width_writes_the_clean_snapshot(self) -> None:
        for width in (4, 8):
            with self.subTest(width=width):
                self.assertEqual(fingerprint(self.out[width]), baseline("fused", (4, 8))[width])

    def test_the_routed_experts_round_trip_experts_first(self) -> None:
        fused = self.plan["model-00002-of-00004.safetensors"][
            "model.language_model.layers.1.mlp.experts.gate_up_proj"
        ]
        half = fused.shape[1] // 2
        out = self.out[4]
        for suffix, piece in (("gate_proj", fused[:, :half, :]), ("up_proj", fused[:, half:, :])):
            stem = f"language_model.model.layers.1.mlp.switch_mlp.{suffix}"
            scales = read_tensor(out, stem + ".scales")
            rebuilt = dequantise(
                read_tensor(out, stem + ".weight"),
                scales,
                read_tensor(out, stem + ".biases"),
                4,
            )
            self.assertEqual(rebuilt.shape, piece.shape, stem)
            limit = 2 * float(scales.astype(np.float32).max())
            self.assertLessEqual(float(np.abs(rebuilt - piece).max()), limit + 1e-6, stem)
        stem = "language_model.model.layers.1.mlp.switch_mlp.down_proj"
        piece = self.plan["model-00002-of-00004.safetensors"][
            "model.language_model.layers.1.mlp.experts.down_proj"
        ]
        rebuilt = dequantise(
            read_tensor(out, stem + ".weight"),
            read_tensor(out, stem + ".scales"),
            read_tensor(out, stem + ".biases"),
            4,
        )
        self.assertEqual(rebuilt.shape, piece.shape)

    def test_a_clean_run_leaves_no_source_behind(self) -> None:
        # The completion point releases every source: keeping one until its
        # output is durable must not mean keeping it forever.
        leftovers = sorted(p.name for p in (self.root / "work").glob("*.safetensors"))
        self.assertEqual(leftovers, [])


@requires_deps
class KilledBeforeFlushTests(RestoresModuleState):
    """The kill the resume exists for: no output shard yet, sources on disk."""

    def setUp(self) -> None:
        super().setUp()
        self.files, _plan = checkpoint_files(fused_plan(), 8)
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-agentworld-kill-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.mirror_ctx = mirror(self.files)
        self.server = self.mirror_ctx.__enter__()
        self.addCleanup(self.mirror_ctx.__exit__, None, None, None)
        self.work = self.root / "work"

    def out_paths(self, prefix: pathlib.Path, bits: tuple) -> dict[int, pathlib.Path]:
        if len(bits) == 1:
            return {bits[0]: prefix}
        return {width: pathlib.Path(f"{prefix}-{width}bit") for width in bits}

    def test_a_kill_before_the_first_flush_keeps_its_sources(self) -> None:
        # `OUTPUT_SHARD_BYTES` past the whole fixture: no flush can make a
        # source droppable, so every converted source must still be in `--work`
        # at the kill. The old code unlinked each one as `convert_shard`
        # returned, and this is the one case a 250 GB fetch must not lose.
        prefix = self.root / "out"
        run = kill_at_conversion_boundary(
            self.root,
            self.server.base,
            ["--bits", "4", "8", "--output", str(prefix), "--work", str(self.work)],
            stop_after=2,
            output_shard_bytes=10 << 30,
        )
        self.assertEqual(
            run.returncode, -signal.SIGKILL, self.root.joinpath("killed.log").read_text()
        )
        self.assertEqual(len(run.converted), 2, run.converted_at_stop)
        for out in self.out_paths(prefix, (4, 8)).values():
            self.assertEqual(
                sorted(p.name for p in out.glob("*.safetensors")) if out.exists() else [],
                [],
                "a flush happened before the kill, so this is not the case under test",
            )
        for source in run.converted:
            self.assertTrue(
                source.is_file(),
                f"{source.name} was dropped before its output reached disk",
            )
            self.assertEqual(source.stat().st_size, len(self.files[source.name]))

    def test_the_resume_reuses_those_sources_and_writes_the_clean_snapshot(self) -> None:
        prefix = self.root / "out"
        argv = ["--bits", "4", "8", "--output", str(prefix), "--work", str(self.work)]
        run = kill_at_conversion_boundary(
            self.root,
            self.server.base,
            argv,
            stop_after=2,
            output_shard_bytes=10 << 30,
        )
        self.assertEqual(len(run.converted), 2, run.converted_at_stop)
        retained = [source.name for source in run.converted]
        present = sorted(p.name for p in self.work.glob("*.safetensors"))
        for name in retained:
            self.assertIn(name, present, "a source vanished between the kill and the resume")
        mark = len(self.server.requests)
        code, log = run_resume(self.root, self.server.base, argv)
        self.assertEqual(code, 0, log)
        self.assertIn(f"resuming: {len(present)} checkpoint shard(s) already in", log)
        # A complete source on disk is not fetched again: its size probe is a
        # header range, and no request moves its payload.
        for name in retained:
            self.assertEqual(
                body_requests(self.server, name, self.files, mark),
                [],
                f"the resume fetched {name} again:\n{log}",
            )
        self.assertGreater(len(self.server.requests), mark)
        for width, out in self.out_paths(prefix, (4, 8)).items():
            with self.subTest(width=width):
                self.assertEqual(fingerprint(out), baseline("fused", (4, 8))[width], log)

    def test_a_solo_width_kill_keeps_its_sources_and_resumes(self) -> None:
        # The 122B install runs one width per pass (`convert_qwen35moe_solo`).
        # The ownership rule is the same with one writer -- every writer that
        # exists must have flushed -- but it is worth pinning on the path that
        # actually takes it.
        prefix = self.root / "out"
        argv = ["--bits", "4", "--output", str(prefix), "--work", str(self.work)]
        run = kill_at_conversion_boundary(
            self.root,
            self.server.base,
            argv,
            stop_after=2,
            output_shard_bytes=10 << 30,
        )
        self.assertEqual(len(run.converted), 2, run.converted_at_stop)
        for source in run.converted:
            self.assertTrue(source.is_file(), f"{source.name} was dropped before its flush")
        present = sorted(p.name for p in self.work.glob("*.safetensors"))
        code, log = run_resume(self.root, self.server.base, argv)
        self.assertEqual(code, 0, log)
        self.assertIn(f"resuming: {len(present)} checkpoint shard(s) already in", log)
        self.assertEqual(fingerprint(prefix), baseline("fused", (4,))[4], log)


@requires_deps
class DurableReleaseTests(RestoresModuleState):
    """A mid-run flush releases the sources it wrote, and only those.

    The fixture's per-shard output is uneven (the expert shard is much larger),
    so a threshold placed just past the second shard's cumulative output makes
    the third conversion's flush land inside that shard. Everything converted
    before reachable output is then durable and may go; the shard the flush
    stopped in may not. The threshold is derived from the fixture, not guessed.
    """

    def setUp(self) -> None:
        super().setUp()
        self.files, _plan = checkpoint_files(fused_plan(), 8)
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-agentworld-durable-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.mirror_ctx = mirror(self.files)
        self.server = self.mirror_ctx.__enter__()
        self.addCleanup(self.mirror_ctx.__exit__, None, None, None)
        self.work = self.root / "work"
        self.prefix = self.root / "out"
        self.argv = ["--bits", "4", "8", "--output", str(self.prefix), "--work", str(self.work)]
        self.shards = sorted(name for name in self.files if name.endswith(".safetensors"))
        self.threshold = self.offsets()[4][1] + 1

    def out_paths(self, prefix: pathlib.Path, bits: tuple) -> dict[int, pathlib.Path]:
        if len(bits) == 1:
            return {bits[0]: prefix}
        return {width: pathlib.Path(f"{prefix}-{width}bit") for width in bits}

    def offsets(self) -> dict[int, list[int]]:
        """Cumulative output bytes after each source, per width, in shard order.

        Converted through the real `convert_shard` into real `OutputWriter`s with
        no flush, so these are the numbers the converter's own threshold sees.
        """
        source = self.root / "measure-source"
        source.mkdir()
        for name in self.shards:
            (source / name).write_bytes(self.files[name])
        saved = prepare.OUTPUT_SHARD_BYTES
        prepare.OUTPUT_SHARD_BYTES = 10 << 30
        try:
            offsets = {}
            for width in (4, 8):
                writer = prepare.OutputWriter(self.root / f"measure-{width}bit")
                fused = prepare.FusedExperts()
                cumulative = []
                for name in self.shards:
                    prepare.convert_shard(source / name, {width: writer}, fused, 8)
                    cumulative.append(writer.total)
                offsets[width] = cumulative
            return offsets
        finally:
            prepare.OUTPUT_SHARD_BYTES = saved

    def run_with_one_fetcher(self, patches: dict, finish=None) -> tuple[int, str]:
        """`main()` in this process with one fetcher, so the conversion order is
        the shard order the offsets were measured in."""
        real_prefetch = prepare.prefetch_shards

        def one_fetcher(shards, fetch):
            return real_prefetch(shards, fetch, fetchers=1, depth=2)

        stack = contextlib.ExitStack()
        with stack:
            if finish is not None:
                stack.enter_context(mock.patch.object(prepare.OutputWriter, "finish", finish))
            return run_in_process(
                self.prefix,
                self.work,
                self.server.base,
                extra=("--bits", "4", "8"),
                patches={"prefetch_shards": one_fetcher, **patches},
            )

    def test_a_mid_run_flush_releases_the_earlier_sources(self) -> None:
        at_first_finish: dict[str, list[str]] = {}
        real_finish = prepare.OutputWriter.finish

        def recording_finish(writer):
            at_first_finish.setdefault(
                "present", sorted(path.name for path in self.work.glob("*.safetensors"))
            )
            return real_finish(writer)

        code, log = self.run_with_one_fetcher(
            {"OUTPUT_SHARD_BYTES": self.threshold}, finish=recording_finish
        )
        self.assertEqual(code, 0, log)
        retained = at_first_finish["present"]
        self.assertNotIn(
            self.shards[0],
            retained,
            f"the first source's output was flushed but it was kept until finish:\n{log}",
        )
        self.assertTrue(
            retained,
            f"every source was released before finish, so none came from the flush interval:\n{log}",
        )
        self.assertLess(
            len(retained),
            len(self.shards),
            f"nothing was released mid-run, so this is the retain-everything case:\n{log}",
        )

    def test_a_kill_after_a_flush_keeps_only_the_unflushed_sources(self) -> None:
        run = kill_at_conversion_boundary(
            self.root,
            self.server.base,
            self.argv,
            stop_after=3,
            output_shard_bytes=self.threshold,
        )
        converted = [source.name for source in run.converted]
        self.assertEqual(len(converted), 3, converted)
        self.assertFalse(
            (self.work / converted[0]).exists(),
            "the first source's output was flushed but the source was kept",
        )
        self.assertTrue(
            (self.work / converted[-1]).is_file(),
            "the source the last flush stopped in was dropped before it was durable",
        )
        retained = [name for name in converted if (self.work / name).is_file()]
        released = [name for name in converted if name not in retained]
        self.assertTrue(retained and released, (released, retained))

        mark = len(self.server.requests)
        code, log = run_resume(self.root, self.server.base, self.argv)
        self.assertEqual(code, 0, log)
        self.assertIn("resuming:", log)
        # This converter adopts no output, so the sources released at the flush
        # are fetched again and the unflushed ones are converted from disk.
        for name in retained:
            self.assertEqual(
                body_requests(self.server, name, self.files, mark),
                [],
                f"the resume fetched the retained source {name} again:\n{log}",
            )
        for name in released:
            self.assertTrue(
                body_requests(self.server, name, self.files, mark),
                f"{name}'s output was released, so the resume must fetch it again:\n{log}",
            )
        for width, out in self.out_paths(self.prefix, (4, 8)).items():
            with self.subTest(width=width):
                self.assertEqual(fingerprint(out), baseline("fused", (4, 8))[width], log)


@requires_deps
class FusedExpertRetentionTests(RestoresModuleState):
    """A source whose experts are still buffered may not be dropped."""

    def setUp(self) -> None:
        super().setUp()
        self.files, _plan = checkpoint_files(per_expert_plan(), 2)
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-agentworld-fused-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.mirror_ctx = mirror(self.files)
        self.server = self.mirror_ctx.__enter__()
        self.addCleanup(self.mirror_ctx.__exit__, None, None, None)
        self.work = self.root / "work"

    def test_a_kill_keeps_a_source_whose_experts_are_still_buffered(self) -> None:
        # Shard 1 completes layer 0 and supplies layer 1's first expert, so the
        # writers are durable (`release` flushed them) while the fused buffer is
        # not. A writer-only rule drops this source and loses layer 1's expert.
        prefix = self.root / "out"
        argv = ["--bits", "4", "8", "--output", str(prefix), "--work", str(self.work)]
        run = kill_at_conversion_boundary(
            self.root, self.server.base, argv, stop_after=1, output_shard_bytes=10 << 30
        )
        self.assertEqual(len(run.converted), 1, run.converted_at_stop)
        source = run.converted[0]
        # The writer flush really happened: layer 0's output shard is on disk.
        self.assertIn("wrote model-00001.safetensors", self.root.joinpath("killed.log").read_text())
        for width in (4, 8):
            self.assertTrue(
                (pathlib.Path(f"{prefix}-{width}bit") / "model-00001.safetensors").is_file()
            )
        self.assertTrue(
            source.is_file(),
            f"{source.name} was dropped while layer 1's experts were still in memory",
        )

        present = sorted(p.name for p in self.work.glob("*.safetensors"))
        mark = len(self.server.requests)
        code, log = run_resume(self.root, self.server.base, argv)
        self.assertEqual(code, 0, log)
        self.assertIn(f"resuming: {len(present)} checkpoint shard(s) already in", log)
        for name in (source.name,):
            self.assertEqual(body_requests(self.server, name, self.files, mark), [], log)
        for width in (4, 8):
            with self.subTest(width=width):
                self.assertEqual(
                    fingerprint(pathlib.Path(f"{prefix}-{width}bit")),
                    baseline("per-expert", (4, 8))[width],
                    log,
                )


if __name__ == "__main__":
    unittest.main()
