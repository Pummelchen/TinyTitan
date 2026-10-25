#!/usr/bin/env python3
"""Build a toy Nemotron-shaped `.ssdai` install and make the real verifier accept it.

The point is the *format*, not the model: a three-layer toy (mamba, MoE,
attention) exercises the same writer as the 74B conversion -- resident index,
affine group-64 packing, a per-expert two-role (up/down) packed blob, per-layer
expert counts, and the manifest -- in seconds, before 150 GB is downloaded.

    python3 tools/test_nemotron_writer.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import prepare_nemotron as nemotron  # noqa: E402

VERIFIER = Path(
    os.environ.get(
        "TINYTITAN_REPACK",
        str(ROOT / ".build/release/TinyTitanRepack")))

HIDDEN = 128
LATENT = 64
EXPERT_WIDTH = 128
EXPERTS = 8
VOCAB = 128


def fake_headers() -> dict[str, dict]:
    def bf16(shape):
        return {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, 0]}

    def f32(shape):
        return {"dtype": "F32", "shape": list(shape), "data_offsets": [0, 0]}

    headers: dict[str, dict] = {
        nemotron.EMBEDDING: bf16((VOCAB, HIDDEN)),
        nemotron.HEAD: bf16((VOCAB, HIDDEN)),
        "language_model.backbone.norm_f.weight": bf16((HIDDEN,)),
        # layer 0: mamba
        "language_model.backbone.layers.0.norm.weight": bf16((HIDDEN,)),
        "language_model.backbone.layers.0.mixer.in_proj.weight": bf16((2 * HIDDEN + 8, HIDDEN)),
        "language_model.backbone.layers.0.mixer.out_proj.weight": bf16((HIDDEN, HIDDEN)),
        "language_model.backbone.layers.0.mixer.conv1d.weight": bf16((HIDDEN + 8, 1, 4)),
        "language_model.backbone.layers.0.mixer.conv1d.bias": bf16((HIDDEN + 8,)),
        "language_model.backbone.layers.0.mixer.A_log": bf16((4,)),
        "language_model.backbone.layers.0.mixer.D": bf16((4,)),
        "language_model.backbone.layers.0.mixer.dt_bias": bf16((4,)),
        "language_model.backbone.layers.0.mixer.norm.weight": bf16((HIDDEN,)),
        # layer 1: MoE
        "language_model.backbone.layers.1.norm.weight": bf16((HIDDEN,)),
        "language_model.backbone.layers.1.mixer.fc1_latent_proj.weight": bf16((LATENT, HIDDEN)),
        "language_model.backbone.layers.1.mixer.fc2_latent_proj.weight": bf16((HIDDEN, LATENT)),
        "language_model.backbone.layers.1.mixer.gate.weight": bf16((EXPERTS, HIDDEN)),
        "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias": f32((EXPERTS,)),
        "language_model.backbone.layers.1.mixer.shared_experts.up_proj.weight": bf16((64, HIDDEN)),
        "language_model.backbone.layers.1.mixer.shared_experts.down_proj.weight": bf16((HIDDEN, 64)),
        # layer 2: attention
        "language_model.backbone.layers.2.norm.weight": bf16((HIDDEN,)),
        "language_model.backbone.layers.2.mixer.q_proj.weight": bf16((HIDDEN, HIDDEN)),
        "language_model.backbone.layers.2.mixer.k_proj.weight": bf16((64, HIDDEN)),
        "language_model.backbone.layers.2.mixer.v_proj.weight": bf16((64, HIDDEN)),
        "language_model.backbone.layers.2.mixer.o_proj.weight": bf16((HIDDEN, HIDDEN)),
    }
    for expert in range(EXPERTS):
        headers[f"language_model.backbone.layers.1.mixer.experts.{expert}.up_proj.weight"] = bf16(
            (EXPERT_WIDTH, LATENT))
        headers[f"language_model.backbone.layers.1.mixer.experts.{expert}.down_proj.weight"] = bf16(
            (LATENT, EXPERT_WIDTH))
    return headers


def fake_text() -> dict:
    return {
        "hidden_size": HIDDEN,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "vocab_size": VOCAB,
        "mamba_num_heads": 4,
        "mamba_head_dim": 32,
        "ssm_state_size": 8,
        "n_groups": 2,
        "conv_kernel": 4,
        "moe_latent_size": LATENT,
        "moe_shared_expert_intermediate_size": 64,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 5.0,
        "norm_topk_prob": True,
        "layer_norm_epsilon": 1e-5,
        "mamba_hidden_act": "silu",
        "time_step_min": 1e-4,
        "use_conv_bias": True,
        "layers_block_type": ["mamba", "moe", "attention"],
        "block_configs": [
            {"block_type": "mamba"},
            {"block_type": "moe", "moe_intermediate_size": EXPERT_WIDTH,
             "n_routed_experts": EXPERTS, "num_experts_per_tok": 2},
            {"block_type": "attention"},
        ],
    }


def fill(install: nemotron.Install, headers: dict[str, dict], bits: int) -> None:
    rng = np.random.default_rng(7)
    for name, entry in install.resident.items():
        if entry.bits is None:
            if entry.source_dtype == "F32":
                value = rng.normal(0, 0.02, entry.shape).astype(np.float32)
            else:
                value = rng.normal(0, 0.02, entry.shape)
        else:
            value = rng.normal(0, 0.02, entry.shape)
        install.write_resident(name, value)
    for layer, plan in install.layer_plans.items():
        for expert in range(plan.experts):
            for tensor in plan.tensors:
                install.write_expert(
                    layer, expert, tensor.role,
                    rng.normal(0, 0.02, tensor.shape))


def build(directory: Path, bits: int) -> Path:
    headers = fake_headers()
    text = fake_text()
    layers = {1: EXPERTS}
    plan = nemotron.build_install_plan(text, headers, layers, bits)
    install = nemotron.Install(directory, bits, "nemotron-toy", plan)
    fill(install, headers, bits)

    def write_tokenizer(_config):
        target = directory / "tokenizer"
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokenizer.json").write_text("{}")
        (target / "tokenizer_config.json").write_text("{}")
        (target / "chat_template.jinja").write_text("{{ messages }}")

    install._tokenizer = write_tokenizer
    install.finish(text, "0" * 64, {})
    return directory


def main() -> int:
    if not VERIFIER.exists():
        print(f"skip: {VERIFIER} is not built")
        return 0
    root = Path(tempfile.mkdtemp(prefix="nemotron-writer-"))
    try:
        failures = 0
        for bits in (4, 8):
            directory = build(root / f"toy-{bits}bit", bits)
            manifest = json.loads((directory / "manifest.json").read_text())
            result = subprocess.run(
                [str(VERIFIER), "--verify-install", "--input-ssdai", str(directory)],
                capture_output=True, text=True)
            status = "ok" if result.returncode == 0 else "FAILED"
            print(
                f"{bits}-bit: {status}: {manifest['numLayers']} layers, "
                f"{manifest['expertsPerLayer']} experts declared, "
                f"stride {manifest['expertStride']}")
            if result.returncode != 0:
                failures += 1
                print(result.stdout[-4000:])
                print(result.stderr[-4000:])
            elif not (directory / "verified-install.json").exists():
                failures += 1
                print("verify-install accepted the install but wrote no receipt")
        return 1 if failures else 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
