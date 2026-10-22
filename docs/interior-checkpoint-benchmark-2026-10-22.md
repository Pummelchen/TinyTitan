# Interior checkpoints: edited-turn benchmark (T24)

Status: measured on a 24.5K-token conversation. The literal 100K read is
**host-blocked**: a full prefill of the scaled 100K request (about 61K tokens)
did not finish inside the 3600 s request timeout, and a real 100K prefix would
take longer. The 24.5K read is the largest the 24 GB M3 completes in a
practical session.

## Method

`benchmark/maya_checkpoint_ab.py`, Ornith 1.5 35B-A3B 4-bit, one server per
arm, default shipped settings (`prefill_chunk=4096`, `expert_slots=160`,
multi-prefix prompt cache, `max-context=131072`). The prompt is a long
alternating conversation ending in a short user turn. The first request
prefills it and captures the arm's checkpoints; the measured request is the
same conversation with that final user turn edited. `max_tokens=1`, greedy.

The target-token argument sizes characters, and repeated prose tokenizes at
about 5 characters per token at this length. `--target-tokens 40000` produced
**24,551 prompt tokens** (24,555 for the edited turn), read back from the
server's own usage.

The 100K attempt used `--target-tokens 100000` (about 61K tokens at the same
character ratio). Its `none` request was still prefilling when the HTTP client
timed out at 3600 s; the server log's last line is `generating`. There is no
100K number to report rather than a fabricated one.

## 24.5K result

One measured edited request per arm. `prefill_s` is the server's own line,
which excludes the snapshot load/restore that happens before prefill.

| arm | capture prefill | edited prefill | edited wall | checkpoint hit | speedup vs none |
| --- | ---: | ---: | ---: | --- | ---: |
| `none` (no checkpoints) | 997.7 s | **1119.3 s** | 1119.8 s | — | 1.0x |
| `message` (system end + last-turn start) | 1059.8 s | **3.07 s** | 3.58 s | `last-turn-start`, 24,529 tokens | **365x** |
| `16k` grid only | 949.0 s | **465.2 s** | 465.5 s | `grid`, 16,384 tokens | 2.41x |
| `32k` grid only | — | — | — | — | not applicable: 32,768 > 24,551 |

The edited text is byte-identical across all three measured arms:
`b344d80e24a36799…` for the one-token completion, so the restore path changes
the time, not the answer. The 16K grid restores two thirds of the prompt and
still spends 465 s re-prefilling the last third; the message boundary restores
24,529 of 24,555 tokens and leaves only the 26-token edited turn.

Capture overhead is inside the noise of this host: the message arm's capture
prefill is 1059.8 s against the none arm's 997.7 s (+6%), while the none arm's
own two full prefills differ by 12% (997.7 s and 1119.3 s). One run cannot
separate the snapshot copy from thermal/run variance; the feature's win is two
orders of magnitude larger than either.

## Decision

Keep the two message boundaries as the default and keep the token grid opt-in
(`TINYTITAN_CHECKPOINT_GRID`, `TINYTITAN_CHECKPOINT_GRID_ONLY`). A 16K/32K grid
on a 100K read would restore at 96K and still re-prefill the last 4K + the
edited turn; `lastTurnStart` restores at the edited turn itself. T20 rejected a
grid as the default and this measurement is what supports keeping that default.
The 32K arm is not run at 24.5K because no grid point fits; a 32K grid on the
61K-token attempt would have had one point at 32K and is strictly worse than
the 16K point at the same read, which is already 150x behind the message
boundary.

## 2K smoke

The same harness at `--target-tokens 2000` (2,563 prompt tokens, two measured
edits): `none` 35.7 s edited prefill, `message` 0.94 s, byte-identical text. It
is the functional smoke that preceded the 24.5K run; the 24.5K table is the
placement evidence.

## Artifact

`.build/t24-checkpoints-ornith4-40k.json` (git-ignored) carries the per-arm
rows, server prefill lines and text hashes. The command is the module docstring
of `benchmark/maya_checkpoint_ab.py`.
