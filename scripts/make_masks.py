"""Generate the Bernoulli(0.5) IN/OUT mask for shadow training.

Shape: [N_shadows, N_pool] where N_pool = len(pub) + len(priv) (~28k).
Each entry is independent Bernoulli(0.5). Shadow `i` trains on
`pool[mask[i] == 1]`. Each sample appears IN for ~N_shadows/2 shadows on
average — exactly what LiRA's per-example IN/OUT Gaussian fit needs.

The pool indexing convention:
    indices [0 .. len(pub)-1]                    -> pub samples (in pub.ids order)
    indices [len(pub) .. len(pub)+len(priv)-1]   -> priv samples (in priv.ids order)

The signal-extraction step MUST follow this same convention when stacking
target/shadow signals.

Run locally OR on the cluster (both produce identical output given the seed):
    python scripts/make_masks.py --n_shadows 64 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import load_pub, load_priv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path,
                   default=Path(__file__).resolve().parents[1] / "data")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "masks" / "in_mask.npy")
    p.add_argument("--n_shadows", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p_in", type=float, default=0.5,
                   help="Bernoulli p (probability sample is IN any given shadow). 0.5 is standard.")
    args = p.parse_args()

    pub = load_pub(args.data_dir, attach_transform=False)
    priv = load_priv(args.data_dir, attach_transform=False)
    n_pub, n_priv = len(pub), len(priv)
    n_pool = n_pub + n_priv
    print(f"[make_masks] pool size: {n_pool} ({n_pub} pub + {n_priv} priv)")
    print(f"[make_masks] N_shadows: {args.n_shadows}, p_in: {args.p_in}, seed: {args.seed}")

    rng = np.random.default_rng(args.seed)
    mask = (rng.random((args.n_shadows, n_pool)) < args.p_in).astype(np.uint8)

    # Sanity stats
    n_in_per_shadow = mask.sum(axis=1)
    n_in_per_sample = mask.sum(axis=0)
    print(f"[make_masks] training-set sizes per shadow: "
          f"min={n_in_per_shadow.min()}, mean={n_in_per_shadow.mean():.1f}, max={n_in_per_shadow.max()}")
    print(f"[make_masks] IN-shadows per sample:         "
          f"min={n_in_per_sample.min()}, mean={n_in_per_sample.mean():.1f}, max={n_in_per_sample.max()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, mask)
    print(f"[make_masks] saved to {args.out}  (shape {mask.shape}, dtype {mask.dtype})")

    # Persist pool layout for reproducibility (so signal extraction stacks consistently)
    layout = {"n_pub": n_pub, "n_priv": n_priv, "seed": args.seed,
              "n_shadows": args.n_shadows, "p_in": args.p_in}
    layout_path = args.out.parent / "pool_layout.json"
    import json
    layout_path.write_text(json.dumps(layout, indent=2))
    print(f"[make_masks] pool layout: {layout}")
    print(f"[make_masks] saved to {layout_path}")


if __name__ == "__main__":
    main()
