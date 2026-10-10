"""Shared production profile for TinyTitan benchmark launchers.

Specialized A/B scripts may override the one control they measure, but every
other setting should come from this module so a plain benchmark run matches
the user-facing launcher profile.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import statistics
import subprocess
from collections.abc import Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models/ornith-1.5_35B_A3B_8Bit"
DEFAULT_API_MODEL = "ornith-1.5-35b-a3b"
DEFAULT_CONTEXT_TOKENS = 262_144
DEFAULT_PROMPT_CACHE_MODE = "multi-prefix"
DEFAULT_PROMPT_CACHE_MEMORY_MIB = 256
# None means "do not pin a budget", so the runtime picks the family's measured
# default (RuntimeConfiguration.decodeTuning). Pinning 8G here would override the
# 12 GiB that Qwen3.8-Flash-Next now ships with, and the published protocol would
# stop measuring what a user actually gets. Pass ram_budget= explicitly to probe
# a specific size.
DEFAULT_EXPERT_CACHE_BUDGET = os.environ.get(
    "TINYTITAN_BENCH_RAM_BUDGET"
)  # None: the shipped default
DEFAULT_KV_BITS = 8
DEFAULT_CONCISE = False
DEFAULT_FAST_ALIAS = False
DEFAULT_MTP = False
SUPPORTED_THINKING_MODES = ("off", "on")


def pgrep_answer(argv: Sequence[str]) -> tuple[str, list[str]]:
    """Ask `pgrep` and keep its three answers apart. AUD-268.

    Returns `("busy", the lines it matched)`, `("clear", [])` only when pgrep
    itself said nothing matches, or `("unknown", why)`. AGENTS.md forbids starting
    a model process beside one that is already running, and every caller used to
    read only the answer it could act on — stdout, or the success status — so an
    erroring pgrep (status 2) or a missing one (127) reported "nothing running"
    and the run began anyway. Not knowing is not a clear.
    """
    try:
        proc = subprocess.run(["pgrep", *argv], capture_output=True, text=True, check=False)
    except OSError as exc:
        return "unknown", [
            f"pgrep could not be run ({exc}); nothing is known about a model process"
        ]
    if proc.returncode == 0:
        return "busy", [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode == 1:
        return "clear", []
    return "unknown", [
        f"pgrep exited {proc.returncode} and did not answer "
        f"{' '.join(argv)}: {proc.stderr.strip() or proc.stdout.strip() or 'no message'}"
    ]


def configured_thinking_mode(
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve the binary model switch used by every benchmark launcher."""
    source = os.environ if environment is None else environment
    value = source.get("TINYTITAN_THINKING_MODE", "off").lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "1": "on",
        "true": "on",
        "yes": "on",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_THINKING_MODES:
        raise ValueError(
            "TINYTITAN_THINKING_MODE must be off or on; "
            "Ornith does not expose low/medium/high effort levels"
        )
    return value


DEFAULT_THINKING_MODE = configured_thinking_mode()


def benchmark_log_path(name: str) -> str:
    """Return a git-ignored benchmark log path inside the checkout."""
    directory = ROOT / ".build/benchmark-logs"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / name)


def run_stamp(epoch: float | None = None) -> str:
    """`%Y%m%dT%H%M%S` in UTC for `epoch`, or for now -- local time repeats an hour on
    this zone's fall-back, and two runs that share a name share and overwrite a record.
    """
    moment = (
        datetime.datetime.now(datetime.timezone.utc)
        if epoch is None
        else datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc)
    )
    return moment.strftime("%Y%m%dT%H%M%S")


def parse_max_tokens(argv, default: int = 512) -> tuple[int | None, str | None]:
    """(tokens, error) for a probe's one length argument, read where it runs.

    Two probes held this as a module-level `int(sys.argv[1])`, which made every
    importer of those files run the cast against the *importing program's* argv --
    and silently re-lengthened the probe for an importer whose argument happened to
    parse.
    """
    if len(argv) < 2:
        return default, None
    try:
        value = int(argv[1])
    except ValueError:
        return None, f"max_tokens must be an integer, got {argv[1]!r}"
    if value < 1:
        return None, f"max_tokens must be at least 1, got {value}"
    return value, None


def catalog_id_for(model: str | os.PathLike[str]) -> str:
    """The catalog id of an install: `<modelID>_<bits>-Bit`.

    The launcher resolves a model key or a catalog id, not a directory, and the
    catalog id is exactly what the server advertises in `/v1/models` (it names
    the routed-expert width). A harness that knows its install directory can
    therefore name it precisely instead of guessing at a short key, and a new
    family needs no entry here -- it is read from the install's own manifest.
    """
    manifest = pathlib.Path(model) / "manifest.json"
    try:
        data = json.loads(manifest.read_text())
    except OSError as exc:
        raise ValueError(f"install manifest unreadable at {manifest}: {exc}") from exc
    model_id = data.get("modelID")
    bits = data.get("quant", {}).get("routedExpert", {}).get("weightBits")
    if not isinstance(model_id, str) or not isinstance(bits, int):
        raise ValueError(f"{manifest} declares no modelID / routedExpert.weightBits")
    return f"{model_id}_{bits}-Bit"


LAUNCHER = ROOT / "tools/server_launcher.sh"


def server_command(
    binary: str | os.PathLike[str],
    port: int,
    *,
    model: str | os.PathLike[str] = DEFAULT_MODEL_PATH,
    cache_mode: str = DEFAULT_PROMPT_CACHE_MODE,
    thinking_mode: str = DEFAULT_THINKING_MODE,
    ram_budget: str | None = DEFAULT_EXPERT_CACHE_BUDGET,
    mtp_model: str | os.PathLike[str] | None = None,
    engine: str = "gpu",
) -> list[str]:
    """Build the launcher invocation for the standard production profile.

    Every harness starts its server through `tools/server_launcher.sh`, so a
    benchmarked server is configured the way a user's server is: the catalog
    resolves the install, the model's own `ModelProfile` supplies the expert
    cache, prefetch and sampling, and the launcher refuses an engine the family
    does not implement. That single seam is also what keeps the harnesses inside
    the release policy -- the launcher only ever uses installs already under
    `models/`.

    `binary` is the release binary the caller resolved; the launcher starts that
    same path, so this only checks that a path-shaped argument is really there.
    `mtp_model` attaches a draft-head sidecar, the only way to reach the
    speculative path. `engine` is cpu or gpu.
    """
    if thinking_mode not in SUPPORTED_THINKING_MODES:
        raise ValueError("thinking_mode must be off or on")
    if engine not in ("cpu", "gpu"):
        raise ValueError("engine must be cpu or gpu")
    if cache_mode not in ("multi-prefix", "off"):
        raise ValueError("cache_mode must be multi-prefix or off")
    binary_path = pathlib.Path(binary)
    if ("/" in str(binary) or binary_path.is_absolute()) and not binary_path.exists():
        raise ValueError(f"release binary not found: {binary}")
    command = [
        "bash",
        str(LAUNCHER),
        "--client",
        "server",
        "--model",
        catalog_id_for(model),
        "--port",
        str(port),
        "--thinking",
        thinking_mode,
        "--engine",
        engine,
        "--kv",
        str(DEFAULT_KV_BITS),
    ]
    # multi-prefix is the launcher's default, so only the off arm is sent; both
    # are spelled out in the launcher and neither is a hidden default here.
    if cache_mode == "off":
        command += ["--prompt-cache", "off"]
    if mtp_model is not None:
        command += ["--mtp-model", str(mtp_model)]
    if ram_budget is not None and engine == "gpu":
        command += ["--ram", str(ram_budget)]
    return command


def server_environment(
    base: Mapping[str, str] | None = None,
    *,
    concise: bool = DEFAULT_CONCISE,
    thinking_mode: str = DEFAULT_THINKING_MODE,
) -> dict[str, str]:
    """Return an environment with concise and thinking modes selected."""
    if thinking_mode not in SUPPORTED_THINKING_MODES:
        raise ValueError("thinking_mode must be off or on")
    environment = dict(os.environ if base is None else base)
    if concise:
        environment["TINYTITAN_CONCISE_MODE"] = "1"
    else:
        environment.pop("TINYTITAN_CONCISE_MODE", None)
    environment["TINYTITAN_THINKING_MODE"] = thinking_mode
    return environment


def resolve_api_model(port, *, timeout=5):
    """Ask the server which model it serves.

    The id names the quantization now (`ornith-1.5-35b-a3b_4-Bit`), and it
    differs per model, so a hardcoded default silently restricts every harness
    to one install -- which is what it did.
    """
    import http.client

    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("GET", "/v1/models")
        data = json.loads(conn.getresponse().read().decode())
        conn.close()
        ids = [row["id"] for row in data.get("data", []) if not row["id"].endswith("-fast")]
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(
            f"asked 127.0.0.1:{port} which model it serves and could not read the answer "
            f"({type(error).__name__}: {error}); no measurement runs against a model id this "
            f"process guessed"
        ) from error
    if not ids:
        raise RuntimeError(
            f"127.0.0.1:{port} answered /v1/models with no model id, so which checkpoint this "
            f"server loaded is unknown; {DEFAULT_API_MODEL!r} is a default, not a measurement"
        )
    return ids[0]


def wait_for_health(proc, port, *, timeout: float = 120) -> bool:
    """Poll `/health` until the server answers.

    `False` means exactly one thing: the process exited, so nothing will ever
    answer. A timeout with the process still alive returns `True` — a large MoE
    streams its experts, and slow is not dead. Every driver that used to inline
    this loop disagreed about that, which is why it lives here now.
    """
    import http.client
    import time

    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            conn.request("GET", "/health")
            if "ok" in conn.getresponse().read().decode():
                conn.close()
                return True
            conn.close()
        except OSError:
            pass
        time.sleep(0.05)
    return True


def request_twice(prompt: str, max_tokens: int, port) -> None:
    """Two streamed greedy requests, content discarded — the server log is the
    product, and the second one is the warm measurement.

    The model id is asked of the running server rather than hardcoded, because a
    pinned id restricts every harness to the one install it names.
    """
    import http.client

    payload = json.dumps(
        {
            "model": resolve_api_model(port),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
    ).encode()
    for i in range(2):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        while resp.read(8192):
            pass
        conn.close()
        print(f"request {i + 1} done")


def request_model(*, fast: bool = DEFAULT_FAST_ALIAS, base: str | None = None) -> str:
    """Return the base API model unless an experiment explicitly asks for fast."""
    return (base or DEFAULT_API_MODEL) + ("-fast" if fast else "")


BENCH_MODEL_ENV = "TINYTITAN_BENCH_MODEL"


def bench_model() -> str:
    """The install a harness runs against.

    `TINYTITAN_BENCH_MODEL` is how the sweeps name their model. A driver that
    hardcodes `DEFAULT_MODEL_PATH` instead launches the shipped install whatever
    the operator exported, and prints a page that names neither.
    """
    return os.environ.get(BENCH_MODEL_ENV, str(DEFAULT_MODEL_PATH))


def _channel_count(noun: str, rows) -> str:
    return f"{noun} {len(rows)} {'line' if len(rows) == 1 else 'lines'}"


def channel_verdict(arms, channels):
    """(lines, exit status) for arms given as `(name, capture-or-None)`.

    A driver's product is the log it captures, so an arm whose capture is None
    (nothing to read) and an arm whose section is empty (a server that never
    printed the counter that channel exists to measure) are both runs that
    measured nothing, and neither may exit 0. The header carries each channel's
    own line count, because a section header over an empty body asserts a section
    that is not there -- which is how AUD-225's and AUD-227's drivers read as
    clean results.
    """
    lines, status = [], 0
    for name, sections in arms:
        if sections is None:
            lines.append(f"ARM FAILED: {name} -- the server never answered /health")
            status = 1
            continue
        columns = list(zip(channels, sections, strict=True))
        header = ", ".join(_channel_count(label, part) for (label, _), part in columns)
        lines.append(f"--- {name} ({header}) ---")
        for _, part in columns:
            lines.extend(part)
        for (_label, wanted), part in columns:
            if not part:
                lines.append(f"NOT MEASURED: {name} -- no log line contained {wanted!r}")
                status = 1
    return lines, status


def arm_metric(rows: Sequence[Mapping], key: str):
    """(median, counted, total) for one metric of one arm.

    The count travels with the value, because a median over only the rows that
    carry the key has a denominator the report used to hide: an mtp-on run whose
    server logged no MTP footer measured the plain scalar decode, so it entered
    the arm's rate median beside the runs that engaged while the acceptance
    median quietly used the one survivor, and nothing printed either count.
    A key no row carries answers `None` with a count of zero instead of raising
    `statistics.StatisticsError`, so a caller can refuse with the metric named
    rather than die after its headline lines have already printed.
    """
    values = [r[key] for r in rows if key in r]
    if not values:
        return None, 0, len(rows)
    return statistics.median(values), len(values), len(rows)


def arm_answered(rows: Sequence[Mapping]) -> bool:
    """Whether any run of an arm streamed content.

    Two arms that streamed nothing hash the same, so "identical" and "stable"
    both have to be earned by content before they say anything about a model.
    This is the rule `tinytitan_determinism_ab.py:156-165` applies to the streams
    it compares; the MTP and ANE A/B drivers compared digests without it.
    """
    return any(r.get("completion_tokens") for r in rows)


def metric_count(name: str, key: str, counted: int, total: int):
    """The reason line for an arm whose metric did not reach every run, or None."""
    if total == 0 or counted == total:
        return None
    if counted == 0:
        return f"NOT MEASURED: the {name} arm -- no run logged {key}"
    return f"PARTIAL: the {name} arm's {key} median is over {counted} of {total} runs"


def byte_claim(rows: Sequence[Mapping], off: str = "off", on: str = "on"):
    """(earned, identical, off digests, on digests) for a two-arm sweep.

    `earned` is False when either arm has no runs, or no run of it streamed
    content: two empty answers hash the same, so a sweep that generated nothing
    looks byte-identical, and `tinytitan_determinism_ab.py:156-165` already
    refuses that shape for the streams it compares. `identical` then requires one
    digest per arm with both arms at it -- an off arm that is not reproducible is
    a refusal rather than a pass, which is the conservatism the drivers already
    had in `len(off_digests) == 1`.
    """
    left = [r for r in rows if r["arm"] == off]
    right = [r for r in rows if r["arm"] == on]
    earned = bool(left) and bool(right) and arm_answered(left) and arm_answered(right)
    digests_a = sorted({r["sha256"] for r in left})
    digests_b = sorted({r["sha256"] for r in right})
    return earned, digests_a == digests_b and len(digests_a) == 1, digests_a, digests_b
