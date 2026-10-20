# The PLE n-gram table at 4 bits

**Status (2026-10-20): implemented, measured, opt-in per install.** This is
Project Maya's PLE-table idea (see `docs/project-maya-scan-2026-10-20.md`,
item B): Maya's engine carries the same Qwen3.8-Flash-Next table at 28.8 GB,
which is a ~4.5-bit-per-value footprint. TinyTitan now has the reader and the
converter for that class of table, with the fp16 install left byte-for-byte as
it was.

## Why

On the installed 125B, `ngram_table.bin` is **102,400,491,520 bytes**
(95.4 GiB) — 63 % of the 162 GB install — and it is identical in every
quantisation of the same checkpoint (`docs/ngram-table-sharing-plan.md`). It
is read one 320-byte row at a time, 16 rows a token; a 4-bit row is 100 bytes,
so the gather moves 3.2× fewer bytes and the file fits a far smaller disk.

## Format

40-byte header (little-endian) then rows:

    magic "TTQ4" (0x34515454), version 1,
    rowDim, bits=4, groupSize=32, rowBytes=100, rowCount (u64),
    headerBytes=40, reserved

    row: packed nibbles, low first   (rowDim / 2 bytes, 80 at 160 dims)
         bf16 scale per group of 32  (groups * 2 bytes)
         bf16 bias per group of 32   (groups * 2 bytes)

`tools/quantize_ngram_table.py` writes it. The reader
(`NgramTableReader.format == .affine4Group32`) validates the header against
`ple_constants.json`'s geometry, dequantises each row into the fp16 gather
destination, and keeps `rowBytes` as the **decoded** stride callers pass to
`gatherConcurrently` — the stored stride stays private, so a q4 table cannot
shorten the PLE prefill chunks' destination spacing.

`ple_constants.json` decides: `table_file` (default `ngram_table.bin`) and
`table_dtype` (`float16` or `int4_affine_group32`). Absent fields are the fp16
install, unchanged.

## Measured

M3, 24 GiB, `qwen3.8-flash-next_125B_A6B_4Bit`, 1,691-token prompt, 32 greedy
tokens, `--max-context 4096 --prompt-cache-mode off`,
`TINYTITAN_RUNNER_STATS=1`, one server per arm, arms alternated:

| arm | prefill s | decode tok/s | gather ms/token |
| --- | ---: | ---: | ---: |
| fp16, run 1 | 57.88 | 3.514 | 15.02 |
| **4-bit, run 1** | 56.68 | **3.818** | 14.48 |
| fp16, run 2 | 56.48 | 3.853 | 14.71 |
| **4-bit, run 2** | 57.16 | **3.757** | 14.22 |

Prefill is unchanged, decode is +3 % at the median (3.68 → 3.79 tok/s, inside
this machine's run-to-run noise), and the gather is ~3 % cheaper. The output
is **byte-identical** across all four arms on this prompt (175 chars, sha256
`cfb0db0faaf6…`), which is the first quality evidence; a broader battery is
the shipping gate, which is why the converter keeps the source by default.

The table itself:

| | fp16 | 4-bit |
| --- | ---: | ---: |
| file | 102,400,491,520 B (95.4 GiB) | **32,000,144,640 B (29.8 GiB)** |
| share of the fp16 file | 100 % | **31.2 %** |
| install (with the source removed) | 162 GB | **~92 GB** |

Verification on the converted file sampled 8 rows: mean absolute error
≤ 5.97e-4, max ≤ 1.54e-3 against the fp16 rows.

## Using it

```bash
# 1. convert (keeps ngram_table.bin; ~10 minutes for the 125B)
python3 tools/quantize_ngram_table.py \
  --install models/qwen3.8-flash-next_125B_A6B_4Bit \
  --out models/qwen3.8-flash-next_125B_A6B_4Bit/ngram_table.q4 --verify

# 2. point the sidecar and the manifest at it, and reclaim the fp16 copy
python3 tools/quantize_ngram_table.py \
  --install models/qwen3.8-flash-next_125B_A6B_4Bit \
  --out models/qwen3.8-flash-next_125B_A6B_4Bit/ngram_table.q4 \
  --update-metadata --delete-source
```

`--update-metadata` rewrites `ple_constants.json` (`table_file`,
`table_dtype`), moves the manifest's file entry to the q4 file with its
sha256/size, and updates the `ple_constants.json` entry. The install receipt
(`verified-install.json`) is bound to the old manifest; the default
`.fullSha256` policy does not read it, but a run under
`.sizeCheckTrustedReceipt` would need a re-install. `--delete-source` is
refused for a partial (`--rows`) table.

To go back: restore `table_file`/`table_dtype` to `ngram_table.bin`/`float16`
(and the manifest entry) before the source is deleted; after deletion, convert
again from the checkpoint's table or restore the install.

## What this does not do

It is a footprint win with a small gather win, not a decode rewrite: the
routed experts still dominate the 125B's token, and the PLE gather is 14–15 ms
of a ~250 ms body. The reason to do it is the 70 GB and the 3.2× smaller
random reads, which matter on a 24 GiB host and on any disk that is also
feeding the expert stream.
