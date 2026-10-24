# Benchmark quality gates: logit noise floor and configuration comparison

Two model-free harnesses live under `benchmark/`. Neither loads a model, starts a
server, or reads the clock: they consume files another run produced and answer
with a status a caller can gate on.

## Logit noise floor

`benchmark/quality_floor.py` compares per-token logit captures from the same
prompts. It reports each candidate's maximum and mean absolute delta, every
position where the top-1 token differs, and a pass/fail verdict.

The floor is `2 x` the reference-vs-reference max absolute delta, measured from
two runs of the same build. Supply the delta with `--reference-delta`, or supply
the second reference run with `--reference-repeat` and let the harness measure
it. With neither, the reference is compared with itself: a zero floor, which is
the strictest reading rather than a claim that the build is deterministic.

A candidate passes only when its maximum absolute delta is at most the floor
(inclusive) and no prompt's top-1 token sequence diverges. A reference repeat
that is not top-1 stable is refused, because a candidate cannot be judged
against a reference that changes between identical runs.

### Capture format

JSONL (one record per line) or JSON (an array, one object, or an object with a
`captures` array). A record is:

```json
{"prompt_id": "basic-fact", "position": 0, "logits": [0.125, -2.5, 1.75]}
```

`prompt_id` names the prompt and the exact token sequence scored, `position` is
contiguous from 0 per prompt, and `logits` is a non-empty finite array with the
same vocabulary length throughout a capture. Numbers are parsed as IEEE-754
binary64 and never rounded. A producer that stores fp16 records the fp16 grid,
so the floor must be measured from the same precision class and the file must
not truncate digits more coarsely than they were measured.

The existing `TINYTITAN_ACT_DUMP` facility records named activation slices as
raw fp16; it is not a logit capture. A logit capture is experiment output, and
this harness reads it as an input.

### Commands

```sh
python3 benchmark/quality_floor.py REFERENCE.jsonl CANDIDATE.jsonl
python3 benchmark/quality_floor.py REFERENCE.jsonl CANDIDATE.jsonl \
    --reference-repeat REFERENCE_REPEAT.jsonl
python3 benchmark/quality_floor.py REFERENCE.jsonl CANDIDATE.jsonl \
    --reference-delta 0.000122
python3 benchmark/quality_floor.py REFERENCE.jsonl CANDIDATE.jsonl --json
```

Status: `0` pass, `1` fail, `2` not measured. A refusal prints
`NOT MEASURED: <reason>` and no traceback: missing or empty captures, prompt
sets or lengths that do not match, duplicate positions, position gaps,
NaN/Infinity, non-numeric logits, or an invalid floor.

## Configuration comparison

`benchmark/tinytitan_bench_compare.py` ingests saved `aggregate.json` results for
several configurations or models and emits a stable Markdown table, or `--json`.
It validates that the results share a workload: the same configured prompt
count, the same prompt capability names, and the same `prompt_tokens` for each
prompt. Completion tokens may differ, because two models can decode different
lengths over the same input.

Older aggregates may omit `summary.measured` and `summary.prompts_configured`.
The comparator then derives completeness from the recorded rows and prints a
`Notes:` line saying so; it never treats the absence as a measurement. Commands
that produce the inputs, from the repository root:

```sh
cd benchmark
python3 tinytitan_benchmark.py                 # shipped default install
python3 tinytitan_benchmark.py ../models/<install>
python3 tinytitan_benchmark.py --matrix --mtp-model ../models/<draft-head>
```

Each run writes `benchmark/benchmark-results/bench-<config>-<stamp>/aggregate.json`.
Compare two or more of those files (a directory holding `aggregate.json` is also
accepted):

```sh
cd benchmark
python3 tinytitan_bench_compare.py \
    benchmark-results/bench-cache_off_mtp_off_4bit-.../aggregate.json \
    benchmark-results/bench-cache_on_mtp_off_4bit-.../aggregate.json
python3 tinytitan_bench_compare.py <result> <result> --json
```

The report prints provenance for every row: git commit, chip/hw.model, macOS,
flags/config and repeats. A result file may record these in a `provenance`
object, with legacy top-level `commit`, `chip`, `macos` and `model` keys also
accepted. Flags are derived from `cache_mode`, `mtp_config` and
`runtime_profile` when the file does not record them. A field the result does
not record prints as `not recorded` (and is `null` in JSON); the comparator never
fills it from the machine it happens to run on.

Status: `0` the report was produced and every run is complete, `1` the report was
produced but a run is incomplete or failed, `2` no comparison exists. A missing
or unreadable result, an unreadable schema, or workloads with different prompt
sets or prompt-token counts prints `NOT MEASURED: <reason>` and writes no table.

## Tests

```sh
cd benchmark
python3 -m unittest test_quality_floor test_tinytitan_bench_compare
```

Both suites build synthetic fixtures in temporary directories. They start no
model and make no wall-clock assertion.
