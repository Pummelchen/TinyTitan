# TinyTitan 6.2 — bf16 dense blocks, Ornith 1.5 122B, and one DSH plugin

This release makes the always-on parts of a 35B-MoE model runnable at the
checkpoint's own bf16 precision, adds Ornith 1.5 122B as a supported install,
puts the bf16-dense Ornith 35B in the installer's catalogue, and moves the
DeepSeek Harness plugin to `0.2.2-alpha.2`. The LAN manager plugin and its
`ttlanmanager` client are removed.

### bf16 weights for attention, GDN and the shared expert

The runtime used to choose its kernels from the manifest's slot width, so a
tensor stored at bf16 inside a 4-bit slot was read as packed nibbles. That is
fixed: the fused-QKV choice, the out-projections, the shared expert and the GDN
input projections all dispatch on the tensor's own dtype through `BF16GEMV`.
The repacker records such tensors in the manifest's `promoted` list instead of
widening a slot, and `VerifiedInstallTool` refuses a promoted tensor the list
does not name. A promoted tensor also disables the fused kernels that assume one
width, through `FusedKernelPolicy`.

The bf16 prefill path now issues one multirow GEMV dispatch per chunk instead of
one per token. `BF16GEMVTests` checks the multirow and single-row kernels
against each other and against a float64 reference, so the grid math is covered
where a golden cannot see it.

### The bf16-dense 35B is an installer model

`tools/install_models.sh ornith15-bf16dense` builds Ornith 1.5 35B with
attention, GDN and the shared expert at the checkpoint's bf16 and the routed
experts at 4-bit. It is one width, has its own model id and catalog id, and its
own ~70 GB conversion — it is not a second width of the 4-bit snapshot. The
launcher defaults the bare name to 4-bit and refuses `--bits 8` by name. The
installer's `--dense-bits 16` is the converter policy behind it.

### Ornith 1.5 122B

The 122B member of the Ornith family is supported at 4-bit and 8-bit: hidden
3072 over 48 layers, 256 experts top-8, served by the same `qwen36` graph. Its
installs are one width per run (`convert_qwen35moe_solo`) because a 250 GB source
cannot stage both snapshots beside a 120 GB install; the repacker consumes each
snapshot shard as it commits it. The profile holds it to the 12 GB process target
with 48/24 expert slots, and it has its own golden baselines.

### DeepSeek Harness `0.2.2-alpha.2`

`dsh-tinytitan` now pins exactly `0.2.2-alpha.2`, requires Node 24.9, and moves
its cordis dependencies to the versions that release resolves. It was verified
against a real 0.2.2 profile: the route block is written through the settings
service, the `tinytitan` preset registers from 0.2.2's own standard composition,
and `selectedDefault` is set. The GitHub Actions runners now use Node 24 for the
plugin and lint jobs.

The second bundle, `dsh-lan-manager`, and the `ttlanmanager` fleet client it
drove are removed. This is a breaking change for anyone using that API; the
release-notes entries for 5.8 still describe the versions that shipped it.

### Also in this release

- The README's benchmark section was cleaned up and the inventory table gained
  dense/MoE and parameter columns.
- `TinyTitanRepack --consume-snapshot` is listed in `--help`.
- The installer sizes the solo models volume for a snapshot that retires as the
  install is written.

### Checks not run

No model is installed under `models/` on the machine this release was built on,
so **every** golden baseline was reported not checked. The targets are
`ornith-4`, `ornith-8`, `ornith-bf16dense-4`, `ornith122-4`, `ornith122-8`,
`qwen36-4`, `qwen36-8`, `qwen38-4`, `qwen38-8`, `agentworld-4`, `agentworld-8`,
`katcoder-4`, `katcoder-8`, `qwen35-2b-4`, `qwen35-2b-8`, `qwen35-4b-4`,
`qwen35-4b-8`, `qwen35-9b-4`, `qwen35-9b-8`, `k2-4` and `k2-8`.

No performance measurement was re-run for this cut. Figures quoted for the
122B in the README, and the bf16 measurements in the commit history, were taken
on the development tree and are not re-certified here.

### Checksum

`tinytitan-6.2-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-6.2-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-6.2-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-6.2-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-6.2-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-6.2-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
