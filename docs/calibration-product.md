# Calibration product (T15-T18)

Status: T15 (schema + CLI), T16 (decision engine + measurement driver), T17
(load and precedence) and T18 (test matrix) are implemented.

## Profile

`CalibrationProfile` is the v1 JSON schema: machine (`hw.model`, memory,
macOS), target (modelID, family, weight bits, context) and six optional knobs,
with `measured`/`measuredAt`/`source` provenance. A draft has no knobs and
`measured: false`; a calibration run merges only kept knobs and sets
`source: "calibrate/1"`.

Validation refuses an unknown schema version, an empty machine or model, an
unsupported context, weights other than 4/8, out-of-range chunk/slot/thread/
tile values and malformed timestamps.

## Decision rule

Samples are interleaved (`baseline, arm A, arm B, baseline, ...`) so drift hits
every arm equally. An arm is kept only when its median beats the baseline by
more than both 3% and three median-absolute-deviations of the samples. The
band is measured on the machine, not assumed.

## Arms

| Arm | Switch | Measurable in-process |
| --- | --- | --- |
| prefill chunk | `EngineConfiguration.prefillChunkTokens` | yes |
| cache slots | `EngineConfiguration.expertCacheSlots` | yes |
| bounded I/O | `TINYTITAN_BOUNDED_IO` at reader creation | no, needs a fresh process |
| prefill tile depth | none (kernel picks from head geometry) | no |
| q8 attention | none (selected from thread width) | no |
| reader threads | none (fixed in the C reader) | no |

`tinytitan calibrate --plan` prints this split.

## Measurement protocol

One engine per sample, loaded with the arm's configuration and unloaded after
it. The sample is the engine's own `completionTokens / decodeSeconds`, with an
8-token warmup request before a 32-token measured request at temperature 0.
Load time is outside the number. Interleaved repetitions cancel drift.

## First real run (Ornith 4-bit, 2026-10-22)

`tinytitan calibrate --model models/ornith-1.5_35B_A3B_4Bit --max-context 8192
--measure --tokens 32 --repeats 2`

| Arm | Baseline | Arm | Improvement | Noise band | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| prefill-chunk-128 | 7.91 tok/s | 7.95 | +0.5% | 3.0% floor | rejected |
| cache-slots-96 | 7.91 tok/s | 8.23 | +4.1% | 6.3% | rejected (inside noise) |

The produced profile is `measured: true`, `source: calibrate/1`, and carries
no knobs: no arm cleared both gates. That is the correct outcome for a
two-repeat run on a host carrying swap; a longer run is the way to settle the
4.1%, not a looser rule. The profile is preserved at
`.build/t16-ornith-calibration.json` (git-ignored).

## Loading and precedence (T17)

The CLI and the server accept `--calibration-profile <path>`, and
`tools/server_launcher.sh` forwards it unchanged:

- The profile is read with the bounded metadata reader and validated; a corrupt
  or unsupported profile fails the launch with the validator's error.
- Precedence is explicit flag > profile > runtime default. The profile only
  fills `prefillChunkTokens` and `expertCacheSlots` when the corresponding flag
  was not given; an explicit flag always wins.
- The startup log names exactly what was applied and lists profile knobs that
  have no runtime switch (`boundedIO`, `q8Attention`, `readerThreads`,
  `prefillTileDepth`) under "not applied", so a profile is never silently
  ignored or partially applied without a line.
- CLI precedence lives in `Args.parse` (a bad profile is an `ArgsError`); the
  server's lives in `ServerArguments.parse` (a bad profile is a
  `ServerArgumentError`) and the log line is written before the ready banner.

Tests: CLI `profileFillsOnlyUnsetKnobs` / `explicitFlagsWinOverTheProfile` /
`profileKnobsWithoutSwitchesAreReported` / `corruptProfileIsRefused`; server
`CalibrationProfileArgumentsTests` mirrors the same four cases.

## Compatibility and the test matrix (T18)

A profile is only applied when it describes the machine and run it is applied
to:

- machine identifier and physical memory must match `currentMachine()`; a
  profile from another Mac is refused, not reinterpreted;
- `target.contextTokens` must equal the context the run was launched with;
- the CLI also checks `target.modelID` against the install it is loading.

All failures are launch errors (`ArgsError` / `ServerArgumentError`) that name
the mismatch. Covered cases: schema round-trip and every validation refusal
(profile tests), the 3 % floor and noise band (runner tests), precedence
fill/explicit-wins at both front ends, corrupt profile, foreign machine, wrong
model, wrong context, switchless knobs reported, and the no-profile path.
Decision tests inject samples, so none of them depends on wall-clock luck.
