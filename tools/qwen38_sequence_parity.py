"""Compare a whole prompt, token by token, against the stateful reference.

Usage:  python3 tools/qwen38_sequence_parity.py <model-dir> <dump-dir>

The dump directory holds one `posN` subdirectory per position, written by a
run with `TINYTITAN_ACT_DUMP` and `TINYTITAN_ACT_DUMP_POSITIONS`.

Exit status: 0 every position was profiled, 2 the directory held nothing to
profile. The cosines are a profile, not a verdict -- see the note in the loop --
so this script never reports that the two implementations disagree.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from qwen38_reference import Reference, NUM_LAYERS  # noqa: E402


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> int:
    model_dir, root = sys.argv[1], Path(sys.argv[2])
    # Asked before the reference is built: a dump root holding nothing is the one
    # answer that needs no weights, and printing nothing and exiting 0 was the
    # report of a comparison that never happened.
    positions = sorted(int(p.name[3:]) for p in root.glob("pos*"))
    if not positions:
        print(f"nothing was compared: no posN directories under {root}")
        return 2
    reference = Reference(model_dir)
    for position in positions:
        directory = root / f"pos{position}"
        token = int((directory / "token.txt").read_text().strip())
        entries, stack_out, logits = reference.step(token)
        print(f"position {position}, token {token}")
        # Print the whole profile rather than stopping at the first layer
        # under threshold: fp16 through 48 layers drifts a little on its own,
        # and what distinguishes drift from a bug is the shape of the curve --
        # gradual decay versus a step at one layer.
        line = []
        for layer in range(NUM_LAYERS):
            path = directory / f"L{layer}_entry.f16"
            if not path.exists():
                continue
            dumped = np.fromfile(path, dtype=np.float16).astype(np.float32)
            line.append((layer, cosine(entries[layer], dumped)))
        if not line:
            print(f"  nothing to profile: no L<N>_entry.f16 in {directory}")
            return 2
        for layer, c in line:
            if layer % 4 == 0 or c < 0.99:
                print(f"  L{layer:02d} entry  cos={c:.5f}")
        drops = [(layer, line[i - 1][1] - c) for i, (layer, c) in enumerate(line) if i > 0]
        if drops:
            worst_layer, worst_drop = max(drops, key=lambda d: d[1])
            print(f"  largest single-layer drop: L{worst_layer:02d} -{worst_drop:.5f}")
        stack_path = directory / "stack_out.f16"
        if not stack_path.is_file():
            print(f"  nothing to compare: no {stack_path.name} in {directory}")
            return 2
        dumped = np.fromfile(stack_path, dtype=np.float16).astype(np.float32)
        top = np.argsort(-logits)[:5]
        print(f"  stack out cos={cosine(stack_out, dumped):.5f}, reference top-5 {top.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
