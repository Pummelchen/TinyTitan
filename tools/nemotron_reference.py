#!/usr/bin/env python3
"""A NumPy forward of a Nemotron 3.5 Bolt `.ssdai` install, for checking the engine.

This is a transcription of `modeling_nemotron_h.py` against the *install's own*
weights and layout, not against the checkpoint: it reads the resident index and
the packed expert blobs exactly as the Swift engine does, and runs one token at
a time. It is slow by design (plain NumPy, no BLAS blocking), so it is the
reference for the toy three-layer fixture and for a prefix of the real model,
never a benchmark.

    tools/nemotron_reference.py <install> <token,token,...> [--logits-out FILE]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

GROUP = 64


def bf16(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def read_index(path: Path) -> dict[str, dict]:
    raw = path.read_bytes()
    index_size, resident_size, count = struct.unpack_from("<QQQ", raw, 0)
    tensors: dict[str, dict] = {}
    for i in range(count):
        offset = 24 + i * 72
        name_offset, name_length, dtype = struct.unpack_from("<IHB", raw, offset)
        file_offset, size = struct.unpack_from("<QQ", raw, offset + 8)
        shape = struct.unpack_from("<IIII", raw, offset + 24)
        scale_offset, scale_size, bias_offset, bias_size = struct.unpack_from(
            "<QQQQ", raw, offset + 40)
        name = raw[name_offset:name_offset + name_length].decode()
        logical = [value for value in shape if value]
        bits = None
        if dtype == 0:
            cells = 1
            for value in logical:
                cells *= value
            bits = size * 8 // cells
        tensors[name] = {
            "dtype": dtype, "shape": logical, "offset": file_offset,
            "size": size, "scale_offset": scale_offset, "bias_offset": bias_offset,
            "bits": bits,
        }
    return tensors


class Install:
    def __init__(self, directory: Path):
        self.directory = directory
        manifest = json.loads((directory / "manifest.json").read_text())
        self.arch = manifest["arch"]
        self.meta = self.arch["nemotronH"]
        self.tensors = read_index(directory / "model_weights.bin")
        self.weights = np.memmap(
            directory / "model_weights.bin", dtype=np.uint8, mode="r")
        layout = json.loads((directory / "packed_experts/layout.json").read_text())
        self.stride = layout["expertStride"]
        self.layers: dict[int, dict] = {}
        for layer in layout["layers"]:
            index = layer["layer"]
            experts = layer["experts"]
            self.layers[index] = {
                "count": layer["expertsPerLayer"],
                "roles": experts[0]["tensors"] if experts else {},
                "file": layer["file"],
            }
        self.expert_maps: dict[int, np.memmap] = {}

    # -- tensors ------------------------------------------------------------

    def vector(self, name: str) -> np.ndarray:
        entry = self.tensors[name]
        start = entry["offset"]
        data = self.weights[start:start + entry["size"]]
        if entry["dtype"] == 3:
            return np.frombuffer(data, dtype=np.float32).copy()
        return bf16(np.frombuffer(data, dtype=np.uint16))

    def matrix(self, name: str) -> np.ndarray:
        entry = self.tensors[name]
        rows, columns = entry["shape"]
        start = entry["offset"]
        if entry["bits"] is None:
            data = bf16(np.frombuffer(
                self.weights[start:start + entry["size"]], dtype=np.uint16))
            return data.reshape(rows, columns).astype(np.float32)
        bits = entry["bits"]
        packed = np.frombuffer(
            self.weights[start:start + entry["size"]], dtype=np.uint8)
        if bits == 4:
            low = (packed & 0x0F).astype(np.float32)
            high = (packed >> 4).astype(np.float32)
            q = np.empty(packed.size * 2, dtype=np.float32)
            q[0::2] = low
            q[1::2] = high
        else:
            q = packed.astype(np.float32)
        q = q.reshape(rows, columns)
        groups = columns // GROUP
        scales = bf16(np.frombuffer(
            self.weights[entry["scale_offset"]:entry["scale_offset"] + rows * groups * 2],
            dtype=np.uint16)).reshape(rows, groups)
        biases = bf16(np.frombuffer(
            self.weights[entry["bias_offset"]:entry["bias_offset"] + rows * groups * 2],
            dtype=np.uint16)).reshape(rows, groups)
        return (q.reshape(rows, groups, GROUP)
                * scales[:, :, None] + biases[:, :, None]).reshape(rows, columns)

    def expert_block(self, layer: int, expert: int) -> np.memmap:
        if layer not in self.expert_maps:
            info = self.layers[layer]
            self.expert_maps[layer] = np.memmap(
                self.directory / "packed_experts" / info["file"],
                dtype=np.uint8, mode="r")
        base = expert * self.stride
        return self.expert_maps[layer][base:base + self.stride]

    def expert_matrix(self, layer: int, expert: int, role: str) -> np.ndarray:
        blob = self.expert_block(layer, expert)
        tensors = self.layers[layer]["roles"]
        weight = tensors[role]
        rows, columns = weight["shape"]
        bits = weight["bits"]
        packed = np.frombuffer(
            blob[weight["offset"]:weight["offset"] + weight["size"]], dtype=np.uint8)
        if bits == 4:
            low = (packed & 0x0F).astype(np.float32)
            high = (packed >> 4).astype(np.float32)
            q = np.empty(packed.size * 2, dtype=np.float32)
            q[0::2] = low
            q[1::2] = high
        else:
            q = packed.astype(np.float32)
        groups = columns // GROUP
        scales = bf16(np.frombuffer(
            blob[tensors[role + "_scales"]["offset"]:
                 tensors[role + "_scales"]["offset"] + rows * groups * 2],
            dtype=np.uint16)).reshape(rows, groups)
        biases = bf16(np.frombuffer(
            blob[tensors[role + "_biases"]["offset"]:
                 tensors[role + "_biases"]["offset"] + rows * groups * 2],
            dtype=np.uint16)).reshape(rows, groups)
        return (q.reshape(rows, groups, GROUP)
                * scales[:, :, None] + biases[:, :, None]).reshape(rows, columns)


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x) + eps) * weight


def rmsnorm_gated(x: np.ndarray, gate: np.ndarray, weight: np.ndarray,
                  groups: int, eps: float) -> np.ndarray:
    out = np.empty_like(x)
    width = x.size // groups
    for g in range(groups):
        part = x[g * width:(g + 1) * width]
        out[g * width:(g + 1) * width] = (
            part / np.sqrt(np.mean(part * part) + eps) * weight[g * width:(g + 1) * width])
    return out * (gate / (1.0 + np.exp(-gate)))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def relu2(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0) ** 2


class Forward:
    def __init__(self, install: Install):
        self.install = install
        self.meta = install.meta
        self.kinds = self.meta["layerKinds"]
        heads = self.meta["mambaNumHeads"]
        head_dim = self.meta["mambaHeadDim"]
        state = self.meta["ssmStateSize"]
        self.layers = len(self.meta["layerKinds"])
        self.conv_state = np.zeros(
            (self.layers, self.conv_channels(), self.meta["mambaConvKernel"]),
            dtype=np.float32)
        self.ssm_state = np.zeros((self.layers, heads, head_dim, state), dtype=np.float32)
        self.kv = {}
        self.attention_heads = install.arch["numHeads"]
        self.attention_kv_heads = install.arch["numKVHeads"]
        self.attention_head_dim = install.arch["headDim"]

    def conv_channels(self) -> int:
        return self.meta["mambaIntermediate"] + 2 * self.meta["mambaGroups"] * \
            self.meta["ssmStateSize"]

    def embedding(self, token: int) -> np.ndarray:
        name = "language_model.backbone.embeddings.weight"
        entry = self.install.tensors[name]
        rows, columns = entry["shape"]
        groups = columns // GROUP
        start = entry["offset"] + token * (columns if entry["bits"] == 8 else columns // 2)
        packed = np.frombuffer(self.install.weights[start:start + columns], dtype=np.uint8)
        q = packed.astype(np.float32)
        scales = bf16(np.frombuffer(
            self.install.weights[entry["scale_offset"] + token * groups * 2:
                                 entry["scale_offset"] + (token + 1) * groups * 2],
            dtype=np.uint16))
        biases = bf16(np.frombuffer(
            self.install.weights[entry["bias_offset"] + token * groups * 2:
                                 entry["bias_offset"] + (token + 1) * groups * 2],
            dtype=np.uint16))
        return (q.reshape(groups, GROUP) * scales[:, None] + biases[:, None]).reshape(columns)

    def step(self, token: int, position: int) -> np.ndarray:
        x = self.embedding(token).astype(np.float32)
        for layer, kind in enumerate(self.kinds):
            norm = self.install.vector(
                f"language_model.backbone.layers.{layer}.norm.weight")
            normed = rmsnorm(x, norm, self.meta["normEpsilon"]).astype(np.float32)
            mixer = f"language_model.backbone.layers.{layer}.mixer"
            if kind == "mamba":
                out = self.mamba(layer, mixer, normed)
            elif kind == "moe":
                out = self.moe(layer, mixer, normed)
            else:
                out = self.attention(layer, mixer, normed, position)
            x = x + out
        final = self.install.vector("language_model.backbone.norm_f.weight")
        normed = rmsnorm(x, final, self.meta["normEpsilon"]).astype(np.float32)
        head = self.install.matrix("language_model.lm_head.weight")
        return head @ normed

    def mamba(self, layer: int, mixer: str, x: np.ndarray) -> np.ndarray:
        meta = self.meta
        heads = meta["mambaNumHeads"]
        head_dim = meta["mambaHeadDim"]
        state_size = meta["ssmStateSize"]
        groups = meta["mambaGroups"]
        intermediate = meta["mambaIntermediate"]
        conv_dim = self.conv_channels()
        projected = self.install.matrix(f"{mixer}.in_proj.weight") @ x
        gate = projected[:intermediate]
        conv_in = projected[intermediate:intermediate + conv_dim]
        dt = projected[intermediate + conv_dim:]
        conv_weight = self.install.vector(f"{mixer}.conv1d.weight").reshape(conv_dim, 4)
        conv_bias = self.install.vector(f"{mixer}.conv1d.bias")
        history = self.conv_state[layer]
        history = np.roll(history, -1, axis=1)
        history[:, -1] = conv_in
        self.conv_state[layer] = history
        conv_out = np.sum(history * conv_weight, axis=1) + conv_bias
        conv_out = silu(conv_out)
        values = conv_out[:intermediate]
        b = conv_out[intermediate:intermediate + groups * state_size]
        c = conv_out[intermediate + groups * state_size:]
        a_log = self.install.vector(f"{mixer}.A_log")
        skip_d = self.install.vector(f"{mixer}.D")
        dt_bias = self.install.vector(f"{mixer}.dt_bias")
        head_states = self.ssm_state[layer]
        y = np.empty(intermediate, dtype=np.float32)
        for h in range(heads):
            group = h // (heads // groups)
            a = -math.exp(float(a_log[h]))
            dt_value = max(
                math.log1p(math.exp(float(dt[h] + dt_bias[h]))), meta["timeStepMin"])
            decay = math.exp(dt_value * a)
            state = head_states[h]
            state *= decay
            state += (dt_value * b[group * state_size:(group + 1) * state_size])[None, :] \
                * values[h * head_dim:(h + 1) * head_dim][:, None]
            y[h * head_dim:(h + 1) * head_dim] = state @ c[
                group * state_size:(group + 1) * state_size]
            y[h * head_dim:(h + 1) * head_dim] += float(skip_d[h]) * values[
                h * head_dim:(h + 1) * head_dim]
        weight = self.install.vector(f"{mixer}.norm.weight")
        normalized = rmsnorm_gated(
            y, gate, weight, groups, meta["mambaEpsilon"])
        return self.install.matrix(f"{mixer}.out_proj.weight") @ normalized

    def top_k(self, values: np.ndarray, k: int) -> list[int]:
        return list(np.argsort(-values, kind="stable")[:k])

    def moe(self, layer: int, mixer: str, x: np.ndarray) -> np.ndarray:
        meta = self.meta
        count = self.install.layers[layer]["count"]
        gate_weights = self.install.matrix(f"{mixer}.gate.weight")
        logits = gate_weights @ x
        scores = 1.0 / (1.0 + np.exp(-logits))
        bias = self.install.vector(f"{mixer}.gate.e_score_correction_bias")
        selected = self.top_k(scores + bias, self.install.arch["topKExperts"])
        weights = scores[selected]
        if self.install.arch.get("routerNormTopK", True):
            weights = weights / weights.sum()
        weights *= meta["routedScalingFactor"]
        latent = self.install.matrix(f"{mixer}.fc1_latent_proj.weight") @ x
        accumulator = np.zeros(meta["moeLatentSize"], dtype=np.float32)
        for expert, weight in zip(selected, weights):
            up = self.install.expert_matrix(layer, int(expert), "up") @ latent
            down = self.install.expert_matrix(layer, int(expert), "down") @ relu2(up)
            accumulator += weight * down
        out = self.install.matrix(f"{mixer}.fc2_latent_proj.weight") @ accumulator
        shared_up = self.install.matrix(f"{mixer}.shared_experts.up_proj.weight") @ x
        shared = self.install.matrix(f"{mixer}.shared_experts.down_proj.weight") @ relu2(shared_up)
        return out + shared

    def attention(self, layer: int, mixer: str, x: np.ndarray, position: int) -> np.ndarray:
        meta = self.meta
        heads = self.attention_heads
        kv_heads = self.attention_kv_heads
        head_dim = self.attention_head_dim
        q = (self.install.matrix(f"{mixer}.q_proj.weight") @ x).reshape(heads, head_dim)
        k = (self.install.matrix(f"{mixer}.k_proj.weight") @ x).reshape(kv_heads, head_dim)
        v = (self.install.matrix(f"{mixer}.v_proj.weight") @ x).reshape(kv_heads, head_dim)
        cache = self.kv.setdefault(layer, {
            "k": np.zeros((1024, kv_heads, head_dim), dtype=np.float32),
            "v": np.zeros((1024, kv_heads, head_dim), dtype=np.float32),
        })
        cache["k"][position] = k
        cache["v"][position] = v
        scale = 1.0 / math.sqrt(head_dim)
        out = np.empty((heads, head_dim), dtype=np.float32)
        for h in range(heads):
            kv_head = h // (heads // kv_heads)
            scores = np.einsum("d,td->t", q[h], cache["k"][:position + 1, kv_head]) * scale
            scores = np.exp(scores - scores.max())
            scores /= scores.sum()
            out[h] = np.einsum("t,td->d", scores, cache["v"][:position + 1, kv_head])
        return self.install.matrix(f"{mixer}.o_proj.weight") @ out.reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("install")
    parser.add_argument("tokens")
    parser.add_argument("--logits-out", type=Path)
    args = parser.parse_args()
    install = Install(Path(args.install))
    forward = Forward(install)
    tokens = [int(value) for value in args.tokens.split(",")]
    logits = None
    for position, token in enumerate(tokens):
        logits = forward.step(token, position)
    assert logits is not None
    if args.logits_out:
        np.save(args.logits_out, logits.astype(np.float32))
        print(f"wrote {logits.size} logits to {args.logits_out}")
    else:
        print(",".join(f"{value:.6f}" for value in logits[:16]))
        print(f"argmax {int(np.argmax(logits))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
