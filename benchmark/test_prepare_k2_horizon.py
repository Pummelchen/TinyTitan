#!/usr/bin/env python3
"""Tests for the K2-Horizon streaming conversion, and its resume.

`tools/prepare_k2_horizon.py --from-hf` downloads one checkpoint shard at a
time, quantizes it into a `ShardWriter` per width, and writes the affine
snapshots. `ShardWriter` buffers up to `SHARD_LIMIT_BYTES` (1 GB) of converted
tensors and only `flush` writes a shard, so a source shard is *not* durable when
its tensors have been converted. The loop used to unlink it right there: a kill
before the next flush lost the converted bytes with the source already gone, and
the converter keeps no resume state, so the next run began from zero.

What is pinned here:

- a source leaves `--work` only once every writer has written past the output
  offset it ended at -- not when an open block happens to be empty, which is the
  test that looks right and never fires (the writer flushes in the middle of a
  source, so a block is almost never empty at a source boundary);
- a source already complete on disk is converted from disk with no request at
  all, a partial one continues from its last byte, and an oversized one is
  discarded rather than resumed onto;
- a real `SIGKILL` while a run is fetching the next shard leaves the sources it
  had converted behind, and the resume produces the snapshot a clean run
  produces without fetching them again; and
- the streamed output is byte-identical to the single-file converter's.

The checkpoint is synthetic (`prepare_k2_horizon.build_fixture`, split into real
safetensors shards) and served from 127.0.0.1, so nothing is fetched and no
install exists. The kill is deterministic and sleep-free: the mirror parks the
payload response for one shard and signals an event, the parent inspects the
directory while the child is blocked inside `curl` for that shard, then sends
`SIGKILL`. The park point is "after the previous shard converted, before the
next one is read", which is where the old code had already unlinked its source.

    cd benchmark && python3.13 -m unittest test_prepare_k2_horizon -v
"""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import io
import json
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
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

try:
    import ml_dtypes  # noqa: F401 - imported for its side effect of existing
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file
except ImportError as exc:  # pragma: no cover - environment, not logic
    ml_dtypes = np = safe_open = load_file = save_file = None
    DEPS_ERROR = str(exc)
else:
    DEPS_ERROR = ""

try:
    import prepare_k2_horizon as prepare

    PREPARE_ERROR = ""
except SystemExit as exc:  # the module exits when a dependency is missing
    prepare = None
    PREPARE_ERROR = str(exc)


def requires_deps(obj):
    reason = DEPS_ERROR or PREPARE_ERROR
    return unittest.skipIf(reason, f"converter or its dependencies unavailable: {reason}")(obj)


SHARDS = 4


# --- the synthetic checkpoint ----------------------------------------------


def checkpoint_files(root: pathlib.Path) -> dict[str, bytes]:
    """`build_fixture`'s checkpoint split into real shards, plus every served file.

    The split is round-robin over the sorted tensor names so each shard carries
    a mix of kinds: the conversion sees the same tensors in a per-shard order a
    real checkpoint does not promise anyway.
    """
    source = root / "checkpoint"
    prepare.build_fixture(source)
    tensors = load_file(str(source / "model.safetensors"))
    weight_map: dict[str, str] = {}
    files: dict[str, bytes] = {}
    names = sorted(tensors)
    for index in range(SHARDS):
        shard = f"model-{index + 1:05d}-of-{SHARDS:05d}.safetensors"
        members = names[index::SHARDS]
        save_file({name: tensors[name] for name in members}, str(source / shard))
        files[shard] = (source / shard).read_bytes()
        for name in members:
            weight_map[name] = shard
    (source / "model.safetensors").unlink()
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    files["config.json"] = (source / "config.json").read_bytes()
    files["model.safetensors.index.json"] = (source / "model.safetensors.index.json").read_bytes()
    for name in prepare.TOKENIZER_FILES:
        files[name] = f"{{}}  // {name} from the test mirror\n".encode()
    return files


def shard_names(files: dict[str, bytes]) -> list[str]:
    return sorted(name for name in files if name.endswith(".safetensors"))


# --- the mirror -------------------------------------------------------------


class Mirror(http.server.ThreadingHTTPServer):
    """A Hub-shaped mirror: ranged GETs, and one payload it can park.

    `hold_body_of(name)` parks the *payload* response for that file (a size
    probe is too small to count) and sets `parked`; nothing moves until
    `release` is set. That is how the kill test observes the converter between
    two shards without a seam inside it.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, files: dict[str, bytes]):
        super().__init__(("127.0.0.1", 0), _MirrorHandler)
        self.files = files
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        self.held_file: str | None = None
        self.parked = threading.Event()
        self.release = threading.Event()
        self.unwound = threading.Event()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/k2"

    def hold_body_of(self, name: str) -> None:
        self.held_file = name
        self.parked.clear()
        self.release.clear()
        self.unwound.clear()

    def record(self, entry: dict) -> None:
        with self.lock:
            self.requests.append(entry)

    def requests_for(self, name: str) -> list[dict]:
        with self.lock:
            return [request for request in self.requests if request["file"] == name]

    def handle_error(self, *_args) -> None:
        # The parked client is killed mid-response and the connection resets;
        # that is the state under test, not a traceback.
        pass


class _MirrorHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "K2Mirror/1"

    def log_message(self, *_args) -> None:  # keep the test output readable
        pass

    def _park(self, name: str, data: bytes, span: tuple[int, int] | None) -> bool:
        """Block a held payload response; True means "served nothing"."""
        if name != self.server.held_file or self.server.release.is_set():
            return False
        header_len = struct.unpack("<Q", data[:8])[0] if len(data) >= 8 else 0
        if span is not None and span[1] - span[0] + 1 <= 8 + header_len:
            return False  # a size probe, not the payload
        self.server.parked.set()
        self.server.release.wait(timeout=120)
        # No response at all: the client is killed while parked, and a partial
        # body would race the resume that follows this test.
        self.server.unwound.set()
        self.close_connection = True
        return True

    def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
        path = urllib.parse.urlparse(self.path).path
        name = path.rsplit("/", 1)[-1]
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
        if self._park(name, data, span):
            return
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
        server.release.set()
        server.shutdown()
        server.server_close()


# --- running the converter --------------------------------------------------


def output_paths(prefix: pathlib.Path, bits: tuple) -> dict[int, pathlib.Path]:
    if len(bits) == 1:
        return {bits[0]: prefix}
    return {width: pathlib.Path(f"{prefix}-{width}bit") for width in bits}


def clear_size_cache() -> None:
    """Start each run with no size cached, as a fresh process would.

    Guarded because the release-rule tests are also run against a converter
    that predates the cache, where there is nothing to clear and the absence of
    the helpers is the point.
    """
    for name in ("header", "expected_size"):
        cached = getattr(prepare, name, None)
        if cached is not None:
            cached.cache_clear()


def run_in_process(
    out: pathlib.Path,
    work: pathlib.Path,
    base: str,
    bits: tuple = (4, 8),
    patches: dict | None = None,
) -> tuple[int, str]:
    """`main()` in this process, against the mirror, with output captured."""
    argv = [
        "prepare_k2_horizon.py",
        "--from-hf",
        "--output",
        str(out),
        "--work",
        str(work),
        "--bits",
        *[str(width) for width in bits],
    ]
    saved_base = prepare.BASE
    prepare.BASE = base
    clear_size_cache()
    buffer = io.StringIO()
    with (
        mock.patch.object(sys, "argv", argv),
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
    return code, buffer.getvalue()


def driver_script(directory: pathlib.Path, base: str, *, shard_limit: int | None = None):
    """A converter process against the test mirror, with an optional flush size."""
    path = directory / "driver.py"
    path.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'tools')!r})\n"
        "import prepare_k2_horizon as prepare\n"
        f"prepare.BASE = {base!r}\n"
        f"LIMIT = {shard_limit!r}\n"
        "if LIMIT is not None:\n"
        "    prepare.SHARD_LIMIT_BYTES = LIMIT\n"
        "raise SystemExit(prepare.main())\n"
    )
    return path


def converter_argv(prefix: pathlib.Path, work: pathlib.Path, bits: tuple = (4, 8)) -> list[str]:
    return [
        "--from-hf",
        "--output",
        str(prefix),
        "--work",
        str(work),
        "--bits",
        *[str(width) for width in bits],
    ]


# --- reading the output -----------------------------------------------------


def tensor_map(out: pathlib.Path) -> dict[str, tuple]:
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
    return tensors


def fingerprint(out: pathlib.Path) -> dict:
    index = json.loads((out / "model.safetensors.index.json").read_text())
    return {"tensors": tensor_map(out), "total_size": index["metadata"]["total_size"]}


_BASELINES: dict[tuple, dict] = {}


def baseline(bits: tuple = (4, 8)) -> dict[int, dict]:
    """The snapshot a clean run of the same checkpoint produces, cached."""
    if bits not in _BASELINES:
        root = pathlib.Path(tempfile.mkdtemp(prefix="tt-k2-baseline-"))
        try:
            files = checkpoint_files(root)
            with mirror(files) as server:
                code, log = run_in_process(root / "out", root / "work", server.base, bits)
            if code != 0:
                raise AssertionError(f"the clean baseline run failed:\n{log}")
            _BASELINES[bits] = {
                width: fingerprint(out) for width, out in output_paths(root / "out", bits).items()
            }
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return _BASELINES[bits]


# --- cases ------------------------------------------------------------------


@requires_deps
class DurableSourceTests(unittest.TestCase):
    """A source is dropped only when every writer has written past it."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)

    def source(self) -> pathlib.Path:
        path = self.dir / "model-00001-of-00004.safetensors"
        path.write_bytes(b"a checkpoint shard")
        return path

    @staticmethod
    def writer(total_bytes: int, pending_bytes: int = 0):
        return SimpleNamespace(total_bytes=total_bytes, pending_bytes=pending_bytes)

    def test_a_source_stays_while_either_width_is_behind_it(self) -> None:
        # `--bits 4 8` writes two outputs at different byte offsets; the wider
        # one holds more per tensor, so one writer can be past a source while
        # the other is not.
        source = self.source()
        pending = {source: {4: 1_500, 8: 2_000}}
        prepare.release_durable_sources(pending, {4: self.writer(2_000), 8: self.writer(1_000)})
        self.assertTrue(source.exists())
        self.assertEqual(pending, {source: {4: 1_500, 8: 2_000}})

    def test_a_source_is_dropped_once_every_writer_is_past_it(self) -> None:
        source = self.source()
        pending = {source: {4: 1_500, 8: 2_000}}
        prepare.release_durable_sources(pending, {4: self.writer(2_000), 8: self.writer(3_000)})
        self.assertFalse(source.exists())
        self.assertEqual(pending, {})

    def test_an_empty_block_is_not_a_flush(self) -> None:
        # The rule this replaced: a zero-length open block *looks* durable, but
        # `total_bytes` is the only count of finished shards. A writer that
        # flushed in the middle of the previous source has an empty block and
        # has still not written past this one.
        source = self.source()
        pending = {source: {4: 5_000}}
        prepare.release_durable_sources(
            pending, {4: self.writer(total_bytes=1_000, pending_bytes=0)}
        )
        self.assertTrue(source.exists())
        self.assertEqual(pending, {source: {4: 5_000}})


@requires_deps
class ReleasePointTests(unittest.TestCase):
    """Where the release happens, observed through the real conversion loop."""

    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-k2-release-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.files = checkpoint_files(self.root)
        self.mirror_ctx = mirror(self.files)
        self.server = self.mirror_ctx.__enter__()
        self.addCleanup(self.mirror_ctx.__exit__, None, None, None)
        self.work = self.root / "work"

    def present_at_each_conversion(self, patches: dict) -> list[int]:
        """How many source shards are in `--work` as each conversion starts."""
        counts: list[int] = []
        real = prepare.convert_shard

        def counting(local, outputs):
            counts.append(len(list(self.work.glob("*.safetensors"))))
            return real(local, outputs)

        with mock.patch.object(prepare, "convert_shard", counting):
            code, log = run_in_process(
                self.root / "out", self.work, self.server.base, patches=patches
            )
        self.assertEqual(code, 0, log)
        return counts

    def test_by_default_nothing_flushes_so_every_source_is_still_there(self) -> None:
        # The fixture is far below the 1 GB flush threshold: no shard is written
        # until `finish`, so every converted source must survive -- the state a
        # kill before the first flush resumes from.
        self.assertEqual(self.present_at_each_conversion({}), [1, 2, 3, 4])
        self.assertEqual(sorted(p.name for p in self.work.glob("*.safetensors")), [])

    def test_a_flush_releases_the_sources_behind_it(self) -> None:
        # `SHARD_LIMIT_BYTES = 1` flushes on every tensor, so each conversion
        # ends past the source it just read and the source goes immediately:
        # the converter never stages the checkpoint beside its output.
        self.assertEqual(self.present_at_each_conversion({"SHARD_LIMIT_BYTES": 1}), [1, 1, 1, 1])

    def test_a_clean_run_leaves_no_source_behind(self) -> None:
        code, log = run_in_process(self.root / "out", self.work, self.server.base)
        self.assertEqual(code, 0, log)
        self.assertEqual(sorted(p.name for p in self.work.glob("*.safetensors")), [])


@requires_deps
class ResumedSourceTests(unittest.TestCase):
    """What the next run does with a complete, partial or oversized source."""

    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-k2-resume-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.files = checkpoint_files(self.root)
        self.mirror_ctx = mirror(self.files)
        self.server = self.mirror_ctx.__enter__()
        self.addCleanup(self.mirror_ctx.__exit__, None, None, None)
        self.work = self.root / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.shard = shard_names(self.files)[0]
        self.argv = converter_argv(self.root / "out", self.work)

    def convert(self, **kwargs) -> tuple[int, str]:
        try:
            return run_in_process(self.root / "out", self.work, self.server.base, **kwargs)
        finally:
            self.server.release.set()

    def test_a_complete_source_is_converted_from_disk_without_a_request(self) -> None:
        (self.work / self.shard).write_bytes(self.files[self.shard])
        code, log = self.convert()
        self.assertEqual(code, 0, log)
        self.assertIn("resuming: 1 checkpoint shard(s) already in", log)
        self.assertIn(f"{self.shard} is complete, converting from disk", log)
        self.assertEqual(
            self.server.requests_for(self.shard),
            [],
            f"a complete source was probed or fetched again:\n{log}",
        )
        for width, out in output_paths(self.root / "out", (4, 8)).items():
            with self.subTest(width=width):
                self.assertEqual(fingerprint(out), baseline()[width], log)

    def test_a_partial_source_continues_from_its_last_byte(self) -> None:
        whole = self.files[self.shard]
        cut = len(whole) // 2
        (self.work / self.shard).write_bytes(whole[:cut])
        code, log = self.convert()
        self.assertEqual(code, 0, log)
        self.assertIn("resuming", log)
        ranges = [request["range"] for request in self.server.requests_for(self.shard)]
        # `curl -C -` asks for the remainder; a restart would begin at 0.
        self.assertIn((cut, len(whole) - 1), ranges, ranges)
        for width, out in output_paths(self.root / "out", (4, 8)).items():
            with self.subTest(width=width):
                self.assertEqual(fingerprint(out), baseline()[width], log)

    def test_an_oversized_source_is_discarded_and_downloaded_again(self) -> None:
        (self.work / self.shard).write_bytes(self.files[self.shard] + b"junk")
        code, log = self.convert()
        self.assertEqual(code, 0, log)
        self.assertIn(f"{self.shard}:", log)
        self.assertIn("discarding", log)
        for width, out in output_paths(self.root / "out", (4, 8)).items():
            with self.subTest(width=width):
                self.assertEqual(fingerprint(out), baseline()[width], log)


@requires_deps
class KilledBeforeFlushTests(unittest.TestCase):
    """The kill the resume exists for: no output shard yet, sources on disk."""

    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-k2-kill-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.files = checkpoint_files(self.root)
        self.mirror_ctx = mirror(self.files)
        self.server = self.mirror_ctx.__enter__()
        self.addCleanup(self.mirror_ctx.__exit__, None, None, None)
        self.work = self.root / "work"
        self.prefix = self.root / "out"
        self.shards = shard_names(self.files)
        # The third shard's payload: shards 1 and 2 have been converted by then,
        # and the loop is about to read the next one.
        self.held = self.shards[2]
        self.argv = converter_argv(self.prefix, self.work)

    def kill_while_fetching(self) -> str:
        """SIGKILL the converter while it waits for `self.held`'s payload."""
        self.server.hold_body_of(self.held)
        log_path = self.root / "killed.log"
        driver = driver_script(self.root, self.server.base, shard_limit=10 << 30)
        with log_path.open("w") as log_handle:
            process = subprocess.Popen(
                [sys.executable, str(driver), *self.argv],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            try:
                self.assertTrue(
                    self.server.parked.wait(120),
                    "the converter never asked for the held shard:\n" + log_path.read_text(),
                )
                self.assertIsNone(process.poll(), "the converter exited before the kill")
                for width, out in output_paths(self.prefix, (4, 8)).items():
                    with self.subTest(width=width):
                        self.assertFalse(
                            out.exists() and any(out.iterdir()),
                            "a flush happened before the kill, so this is not the case under test",
                        )
                process.send_signal(signal.SIGKILL)
            finally:
                process.wait(timeout=30)
                self.server.release.set()
                self.assertTrue(self.server.unwound.wait(30), "the held request did not unwind")
        return log_path.read_text()

    def test_a_kill_before_the_first_flush_keeps_its_sources(self) -> None:
        # `SHARD_LIMIT_BYTES` past the whole fixture, so no flush can make a
        # source droppable; the two converted shards must still be in `--work`
        # at the kill. The old code unlinked each one as its tensors were added.
        log = self.kill_while_fetching()
        for name in self.shards[:2]:
            source = self.work / name
            self.assertTrue(
                source.is_file(),
                f"{name} was dropped before its output reached disk:\n{log}",
            )
            self.assertEqual(source.stat().st_size, len(self.files[name]))

    def test_the_resume_converts_them_without_a_fetch_and_matches_a_clean_run(self) -> None:
        log = self.kill_while_fetching()
        retained = [
            name
            for name in self.shards
            if (self.work / name).is_file()
            and (self.work / name).stat().st_size == len(self.files[name])
        ]
        self.assertGreaterEqual(len(retained), 2, f"the kill left too little to reuse:\n{log}")
        present = sorted(path.name for path in self.work.glob("*.safetensors"))
        mark = len(self.server.requests)
        driver = driver_script(self.root, self.server.base)
        result = subprocess.run(
            [sys.executable, str(driver), *self.argv],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        log = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, log)
        self.assertIn(f"resuming: {len(present)} checkpoint shard(s) already in", log)
        # A complete source on disk is converted from disk: no probe and no body
        # request. Only what the killed run had not finished is fetched.
        fetched = {
            request["file"]
            for request in self.server.requests[mark:]
            if request["file"].endswith(".safetensors")
        }
        for name in retained:
            self.assertNotIn(name, fetched, f"the resume fetched {name} again:\n{log}")
        self.assertEqual(fetched, set(self.shards) - set(retained), log)
        for width, out in output_paths(self.prefix, (4, 8)).items():
            with self.subTest(width=width):
                self.assertEqual(fingerprint(out), baseline()[width], log)
        self.assertEqual(sorted(p.name for p in self.work.glob("*.safetensors")), [])


@requires_deps
class OutputContractTests(unittest.TestCase):
    """The streamed conversion writes what the single-file converter writes."""

    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="tt-k2-output-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.files = checkpoint_files(self.root)
        self.source = self.root / "checkpoint"
        self.mirror_ctx = mirror(self.files)
        self.server = self.mirror_ctx.__enter__()
        self.addCleanup(self.mirror_ctx.__exit__, None, None, None)

    def convert(self) -> tuple[int, str]:
        return run_in_process(self.root / "out", self.root / "work", self.server.base)

    def test_the_sharded_output_matches_the_single_file_conversion(self) -> None:
        code, log = self.convert()
        self.assertEqual(code, 0, log)
        for width, out in output_paths(self.root / "out", (4, 8)).items():
            direct = self.root / f"direct-{width}bit"
            prepare.convert(self.source, direct, width)
            with self.subTest(width=width):
                self.assertEqual(tensor_map(out), tensor_map(direct))

    def test_the_index_names_only_shards_that_exist(self) -> None:
        code, log = self.convert()
        self.assertEqual(code, 0, log)
        for width, out in output_paths(self.root / "out", (4, 8)).items():
            with self.subTest(width=width):
                index = json.loads((out / "model.safetensors.index.json").read_text())
                for name, shard in index["weight_map"].items():
                    self.assertTrue((out / shard).is_file(), f"{name} -> {shard}")
                total = 0
                for shard in set(index["weight_map"].values()):
                    with safe_open(out / shard, framework="np") as handle:
                        total += sum(handle.get_tensor(name).nbytes for name in handle.keys())
                self.assertEqual(index["metadata"]["total_size"], total)


if __name__ == "__main__":
    unittest.main()
