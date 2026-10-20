# Expert routing profile: a persisted warm start

**Status (2026-10-20): implemented, measured, on by default.** Rebuilt from
Project Maya's `expert-profile.bin` idea (see
`docs/project-maya-scan-2026-10-20.md`, item D): Maya saves how often each
expert was routed after every answer and reads the counts back at the next
start, where the most-routed experts are placed first.

## What was wrong

TinyTitan's expert cache is aging-LFU, but its score lived only in memory and
was **zeroed at every prefill-to-decode handover**
(`PreadExpertStreamer.resetExpertUseCounts`). That reset exists for a good
reason — a prefill chunk plans every expert it touches hundreds of times, so
the leftovers outrank anything decode has used — but it also threw away the one
thing the engine had just learned about the request. A restart began with every
slot at zero and the first tokens re-read experts the last session had already
found.

## What changed

- `ExpertUsageProfile` (Infrastructure/Streaming) stores `layers × experts`
  routing counts in a small binary file beside the install
  (`expert-profile.bin`). On load the counts are **halved and capped** (512) so
  the profile follows a workload that moves on instead of freezing the first
  session.
- `PreadExpertStreamer` takes the profile as `seedUseCounts`; the handover
  reset now restores that prior instead of zero. No profile keeps the old
  zeroing.
- `Model` loads the profile at open, passes each layer's slice to its streamer,
  and folds the open layers' counts back into the file after every request.
  The same call adopts the counts as the **next** request's prior, so a
  long-lived server keeps learning without a restart.
- `TINYTITAN_EXPERT_PROFILE=0` disables load and save; any other value is used
  as the path. The default is `<install>/expert-profile.bin`.

The format checks the layer/expert geometry, refuses foreign or truncated
files, and writes atomically. A read-only install simply gets no profile.

## Measurements

M3, 24 GiB, `models/ornith-1.5_35B_A3B_4Bit`, `--max-context 4096`,
`--prompt-cache-mode off`, `TINYTITAN_RUNNER_STATS=1`, greedy, 64 new tokens,
one server per arm. Arms alternated off/on; the machine carried ~8.4 GiB of
swap, so absolute numbers move, but every pair below was measured back to back.
A 2.2K-token prompt; the profile starts from the previous run of the same
prompt unless stated.

| workload | profile | decode tok/s (median) | expert reads | reloads | io ms/token | body ms/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| same 2.2K prompt, 4 pairs | off | 12.20 | 18,319 MiB | 1,844 | 18.1 | 73.3 |
| | **on** | **17.15 (+40.6 %)** | 16,312 MiB | 28 | 2.3 | 50.1 |
| different prompt, same domain, 3 pairs | off | 12.46 | 17,958 MiB | 1,690 | 24.1 | 73.0 |
| | **on** | **16.07 (+29.0 %)** | 16,784 MiB | 629 | 7.6 | 54.4 |
| different domain (cooking), 2 pairs | off | 13.55 | 17,481 MiB | 1,601 | 15.7 | 65.3 |
| | **on** | **15.90 (+17.4 %)** | 16,430 MiB | 621 | 7.9 | 56.0 |

Prefill is unchanged in every pair (20–22 s), so the whole gain is decode. The
mechanism is visible in the counters: the prior keeps the prompt's experts in
their slots, so decode stops evicting and re-reading them — reloads fall 63–98 %
and the decode-side read traffic collapses, while total reads fall 6–11 %
because prefill dominates the total. `io_ms` falls from 16–24 ms to 2–8 ms per
token, and `body_ms` — the CPU's wait on the layer's command buffer — falls
14–32 %.

The three rows are also the honest scope: the more the new prompt looks like
what the profile saw, the more it helps. A brand-new install has no profile;
the first start writes one, and the gain begins with the next start (or the
next request, through the in-process adoption).

## Reproducibility caveat

The profile decides **which experts occupy which slots**, and the routed-MoE
reduction is ordered by slot. A different slot order therefore rounds
differently, and greedy text can differ between a profile run and a cold run.
This is the same class of effect as changing the cache size, which the engine
already documents as producing valid but different tokens. The stored goldens
are one exact text per (machine, build, model), so
`tools/golden-baseline.sh` exports `TINYTITAN_EXPERT_PROFILE=0`; a
profile-enabled run differing from a golden is cache state, not a regression.

## Reproduction

The harness is `/tmp/profile-ab.sh`, `/tmp/profile-ab2.sh` and
`/tmp/profile-cross2.sh` (not committed): each starts one server per arm on
port 18100, sends the same JSON body, records the `TinyTitan generation` and
`runner cb1_ms` lines, and alternates the arms. The profile file for the
cross-prompt arms is copied from `/tmp/profile-A.bin` before each run so the
history does not drift between measurements.

## What it cost

A 40 KB file per install, one `pread`-free load at start, and an atomic 40 KB
write after each request. No engine math changed: the kernels still sum the
same experts in the same slot order for a given slot assignment; only the
assignment now starts from history.
