#!/usr/bin/env python3
"""Validate the chunked form of the gated-DeltaNet recurrence.

The CPU prefill spends about two thirds of a long prompt in the token-ordered
delta rule: every token reads and writes the head's whole `dv * dk` state, so
a 2B prompt streams ~1 MB per layer per token. A chunked scan would read the
state once per chunk instead of once per token. This file derives and checks
the algebra on a tiny case before any C is written.

Scalar recurrence (what `tinytitan_delta_rule` does, one token at a time):

    S_t = alpha_t * S_{t-1}
    c_t = beta_t * (v_t - S_t @ k_t)
    S_t = S_t + outer(c_t, k_t)
    o_t = S_t @ q_t

Chunked form, with `A_t = prod_{i<=t} alpha_i`:

    S_t = A_t * S_0 + sum_{s<=t} (A_t / A_s) * outer(c_s, k_s)

so the only products that touch `S_0` are `S_0 K^T` and `S_0 Q^T`, computed
once per chunk, and the within-chunk corrections need only the Gram matrices
`K K^T` and `K Q^T` plus a triangular pass over the chunk's vectors.

    python3 benchmark/gdn_chunked_scan.py            # random case
    python3 benchmark/gdn_chunked_scan.py --chunk 64
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

import numpy as np


class Case(NamedTuple):
    state: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    queries: np.ndarray
    alphas: np.ndarray
    betas: np.ndarray


def make_case(dv: int, dk: int, chunk: int, seed: int) -> Case:
    rng = np.random.default_rng(seed)
    return Case(
        state=rng.normal(size=(dv, dk)),
        keys=rng.normal(size=(chunk, dk)),
        values=rng.normal(size=(chunk, dv)),
        queries=rng.normal(size=(chunk, dk)),
        alphas=rng.uniform(0.5, 1.0, size=chunk),
        betas=rng.uniform(0.1, 0.9, size=chunk),
    )


def scalar(case: Case) -> tuple[np.ndarray, np.ndarray]:
    state = case.state.copy()
    readouts = []
    for t in range(case.alphas.size):
        state = case.alphas[t] * state
        stored = state @ case.keys[t]
        correction = case.betas[t] * (case.values[t] - stored)
        state = state + np.outer(correction, case.keys[t])
        readouts.append(state @ case.queries[t])
    return np.asarray(readouts), state


def chunked(case: Case) -> tuple[np.ndarray, np.ndarray]:
    chunk = case.alphas.size
    decay = np.cumprod(case.alphas)
    # The only two products that touch S_0.
    state_keys = np.array([case.state @ case.keys[t] for t in range(chunk)])
    state_queries = np.array([case.state @ case.queries[t] for t in range(chunk)])
    gram_keys = case.keys @ case.keys.T
    gram_queries = case.keys @ case.queries.T

    corrections = np.zeros((chunk, case.state.shape[0]))
    readouts = np.zeros((chunk, case.state.shape[0]))
    for t in range(chunk):
        previous = decay[t - 1] if t > 0 else 1.0
        # S_{t-1} k_t and S_{t-1} q_t, from S_0 and the earlier corrections.
        stored = previous * state_keys[t]
        read = previous * state_queries[t]
        for s in range(t):
            weight = previous / decay[s]
            stored = stored + weight * corrections[s] * gram_keys[s, t]
            read = read + weight * corrections[s] * gram_queries[s, t]
        corrections[t] = case.betas[t] * (case.values[t] - case.alphas[t] * stored)
        readouts[t] = case.alphas[t] * read + corrections[t] * (case.keys[t] @ case.queries[t])

    # S_chunk = A_chunk * S_0 + sum_s (A_chunk / A_s) * outer(c_s, k_s).
    state = decay[-1] * case.state
    for s in range(chunk):
        state = state + (decay[-1] / decay[s]) * np.outer(corrections[s], case.keys[s])
    return readouts, state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dv", type=int, default=8)
    parser.add_argument("--dk", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    case = make_case(args.dv, args.dk, args.chunk, args.seed)
    scalar_readouts, scalar_state = scalar(case)
    chunked_readouts, chunked_state = chunked(case)
    readout_error = float(np.abs(scalar_readouts - chunked_readouts).max())
    state_error = float(np.abs(scalar_state - chunked_state).max())
    print(f"dv={args.dv} dk={args.dk} chunk={args.chunk}")
    print(f"readout max |scalar - chunked| = {readout_error:.3e}")
    print(f"state   max |scalar - chunked| = {state_error:.3e}")
    return 0 if max(readout_error, state_error) < args.tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
