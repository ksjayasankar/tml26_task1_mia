#!/usr/bin/env python
"""Per-(model, sample, augmented-view) signal extraction.

For each of the 64 trained shadow ResNet-18s and for the target model:
  for each sample j in pool (pub then priv, concatenated; ~28k):
    for each dihedral D_4 view k in 0..7:
      logit-scaled phi_ijk = log(p_y / (1 - p_y))     -> for LiRA
      softmax_y_ijk        = softmax(logits)[true_y]  -> for RMIA

Outputs (.npz, in `signals/`):
  shadow_signals.npz   keys: phi  (64, N_pool, 8) float32
                              p_y  (64, N_pool, 8) float32
  target_signals.npz   keys: phi  (N_pool, 8)     float32
                              p_y  (N_pool, 8)     float32

Compute on a Tesla P100: ~5-15 minutes wall-clock (dominated by 64 shadow
checkpoint loads + 64 * 28k * 8 forward passes). Total output disk: ~115 MB.

Pool indexing is anchored to `masks/pool_layout.json` (n_pub=14000,
n_priv=14000); the downstream LiRA/RMIA scoring uses the same convention.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import load_pub, load_priv, default_transform  # noqa: E402
from src.model import build_resnet18, load_target  # noqa: E402
from src.augmentations import forward_with_augs, DIHEDRAL_OPS  # noqa: E402


def _softmax_torch(logits_np: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    m = logits_np.max(axis=-1, keepdims=True)
    e = np.exp(logits_np - m)
    return e / e.sum(axis=-1, keepdims=True)


def _phi_and_p_y_from_logits(logits_NKC: np.ndarray, labels_N: np.ndarray):
    """Convert per-view logits to (phi, p_y) arrays of shape [N, K].

    phi   = log(p_y / (1 - p_y))  via stable log(p_y) - logsumexp(logits[!= y])
    p_y   = softmax(logits)[true label]
    """
    N, K, C = logits_NKC.shape
    rows = np.arange(N)[:, None]
    cols = np.arange(K)[None, :]
    target_logit = logits_NKC[rows, cols, labels_N[:, None]]            # [N, K]
    other = logits_NKC.copy()
    other[rows, cols, labels_N[:, None]] = -np.inf
    m = other.max(axis=-1, keepdims=True)
    lse = m.squeeze(-1) + np.log(np.exp(other - m).sum(axis=-1))         # [N, K]
    phi = (target_logit - lse).astype(np.float32)
    p = _softmax_torch(logits_NKC.astype(np.float64))                    # [N, K, C]
    p_y = p[rows, cols, labels_N[:, None]].astype(np.float32)            # [N, K]
    return phi, p_y


def make_pool(data_dir: Path):
    """Return (concat_dataset, n_pub, n_priv) with eval transform attached."""
    pub = load_pub(data_dir, attach_transform=False)
    priv = load_priv(data_dir, attach_transform=False)
    eval_tfm = default_transform()
    pub.transform = eval_tfm
    priv.transform = eval_tfm
    return ConcatDataset([pub, priv]), len(pub), len(priv)


def extract_for_one_model(
    model: torch.nn.Module, pool, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run a model on the entire pool with K=8 dihedral views.

    Returns (phi[N,K], p_y[N,K]) both float32.
    """
    model.eval()
    _, logits_NKC, labels_N = forward_with_augs(
        model, pool, ops=DIHEDRAL_OPS, device=device, batch_size=256
    )
    return _phi_and_p_y_from_logits(logits_NKC, labels_N)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=Path, required=True,
                   help="Repo root (e.g. $HOME/tml26-mia).")
    p.add_argument("--n_shadows", type=int, default=64)
    p.add_argument("--target_only", action="store_true",
                   help="Skip shadow extraction, run target only (debug).")
    p.add_argument("--shadow_subset", type=int, nargs="*", default=None,
                   help="Optional list of shadow_ids to process (debug).")
    args = p.parse_args()

    base = Path(args.base_dir)
    data_dir = base / "data"
    weights_dir = base / "weights"
    signals_dir = base / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract] device: {device}", flush=True)

    print("[extract] building pool ...", flush=True)
    pool, n_pub, n_priv = make_pool(data_dir)
    n_pool = n_pub + n_priv
    print(f"  pool: {n_pool} ({n_pub} pub + {n_priv} priv)", flush=True)

    # Sanity: pool layout matches what make_masks.py wrote
    layout_path = base / "masks" / "pool_layout.json"
    if layout_path.exists():
        layout = json.loads(layout_path.read_text())
        if (layout["n_pub"], layout["n_priv"]) != (n_pub, n_priv):
            sys.exit(
                f"pool_layout.json mismatch: expected n_pub={layout['n_pub']} "
                f"n_priv={layout['n_priv']}, got {n_pub}/{n_priv}"
            )
        print(f"  pool layout matches masks/pool_layout.json", flush=True)

    K = len(DIHEDRAL_OPS)

    # ----- target -----
    print("\n[extract] target ...", flush=True)
    t0 = time.time()
    target = load_target(data_dir, device=device)
    target_phi, target_p = extract_for_one_model(target, pool, device)
    print(f"  target_phi: {target_phi.shape}  target_p_y: {target_p.shape}  "
          f"({time.time()-t0:.1f}s)", flush=True)
    np.savez(signals_dir / "target_signals.npz", phi=target_phi, p_y=target_p)
    print(f"  saved -> {signals_dir / 'target_signals.npz'}", flush=True)
    del target

    if args.target_only:
        print("[extract] --target_only set; done.", flush=True)
        return

    # ----- shadows -----
    shadow_ids = args.shadow_subset if args.shadow_subset else list(range(args.n_shadows))
    print(f"\n[extract] shadows ({len(shadow_ids)} models): {shadow_ids[:5]}...", flush=True)

    # Pre-allocate (N_shadows, N_pool, K) float32 arrays. 64*28000*8*4 ~= 57 MB each.
    shadow_phi = np.zeros((args.n_shadows, n_pool, K), dtype=np.float32)
    shadow_p_y = np.zeros((args.n_shadows, n_pool, K), dtype=np.float32)

    t_all = time.time()
    for sid in shadow_ids:
        wpath = weights_dir / f"shadow_{sid:03d}.pt"
        if not wpath.exists():
            sys.exit(f"shadow weight not found: {wpath}")
        t0 = time.time()
        model = build_resnet18().to(device)
        state = torch.load(wpath, map_location=device, weights_only=True)
        model.load_state_dict(state)
        phi, p_y = extract_for_one_model(model, pool, device)
        shadow_phi[sid] = phi
        shadow_p_y[sid] = p_y
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        if sid % 8 == 0 or sid == shadow_ids[-1]:
            print(f"  shadow {sid:>3d}: {time.time()-t0:.1f}s "
                  f"  cumulative {(time.time()-t_all)/60:.1f} min", flush=True)

    out_path = signals_dir / "shadow_signals.npz"
    np.savez(out_path, phi=shadow_phi, p_y=shadow_p_y)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n[extract] saved -> {out_path}  ({size_mb:.1f} MB)", flush=True)
    print(f"[extract] total wall: {(time.time()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
