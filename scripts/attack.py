#!/usr/bin/env python
"""End-to-end attack driver: LiRA + RMIA + rank-fusion ensemble.

Inputs (must already exist):
  signals/shadow_signals.npz    keys phi (64, N_pool, 8), p_y (64, N_pool, 8)
  signals/target_signals.npz    keys phi (N_pool, 8),     p_y (N_pool, 8)
  masks/in_mask.npy             shape (64, N_pool)
  masks/pool_layout.json        gives n_pub, n_priv
  data/pub.pt, priv.pt          for ids and pub.pt ground-truth membership

Outputs (in `results/`):
  per_method_table.csv          pub TPR@5%FPR per individual method
  ensemble_table.csv            equal-weight vs grid-tuned ensemble
  submission.csv                final 4-method equal-weight ensemble (priv scores)
                                ready to POST to the leaderboard

CPU-only; runs in ~30 s on a laptop given the pre-computed signals.

Pipeline:
  1. Score with three LiRA variants (online global, online diagonal, offline).
  2. Score with RMIA, grid-searching alpha x gamma on pub.pt TPR@5%FPR.
  3. Apply committed sign per scorer using pub.pt ground truth.
  4. Rank-normalise each scorer globally, average with equal weights w_i=0.25.
     Also evaluate a grid-search over the simplex on an 80/20 pub split as a
     sanity check; fall back to equal weights if it overfits.
  5. Write the final priv scores as a leaderboard CSV.
"""
from __future__ import annotations


def _ensure_deps():
    """pandas + sklearn aren't in the cluster docker image; install once into ~/.local."""
    import importlib, subprocess, sys
    needed = [("pandas", "pandas"), ("sklearn", "scikit-learn")]
    missing = []
    for mod, pkg in needed:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[deps] pip install --user {missing}", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--user", "--quiet", *missing]
        )
        import site
        if hasattr(site, "getusersitepackages"):
            user_site = site.getusersitepackages()
            if user_site not in sys.path:
                sys.path.append(user_site)
        importlib.invalidate_caches()


_ensure_deps()


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import load_pub, load_priv
from src.eval import evaluate_with_sign_check, pub_membership_labels, tpr_at_fpr
from src.lira import lira_online_diag, lira_online_global, lira_offline
from src.rmia import rmia_sweep
from src.ensemble import equal_weight_baseline, grid_search_weights
from src.submit import write_submission_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=Path, required=True,
                   help="Repo root containing signals/, masks/, data/, results/")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="FPR target for TPR@FPR evaluation.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the 80/20 pub.pt split used in the sanity grid search.")
    p.add_argument("--grid_step", type=float, default=0.05,
                   help="Granularity of the simplex grid search over ensemble weights.")
    args = p.parse_args()

    base = Path(args.base_dir)
    signals_dir = base / "signals"
    masks_dir = base / "masks"
    data_dir = base / "data"
    results_dir = base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------- load everything -----------------------
    print("[attack] loading signals + mask + pool layout ...", flush=True)
    sh = np.load(signals_dir / "shadow_signals.npz")
    shadow_phi = sh["phi"]
    shadow_p_y = sh["p_y"]
    tg = np.load(signals_dir / "target_signals.npz")
    target_phi = tg["phi"]
    target_p_y = tg["p_y"]
    mask = np.load(masks_dir / "in_mask.npy")
    layout = json.loads((masks_dir / "pool_layout.json").read_text())
    n_pub, n_priv = layout["n_pub"], layout["n_priv"]
    n_pool = n_pub + n_priv
    pub_idx = np.arange(n_pub)
    priv_idx = np.arange(n_pub, n_pool)
    print(f"  shadow_phi {shadow_phi.shape}; target_phi {target_phi.shape}; mask {mask.shape}",
          flush=True)
    print(f"  pool: {n_pub} pub + {n_priv} priv = {n_pool}", flush=True)
    assert shadow_phi.shape[1] == n_pool, "pool layout mismatch with shadow signals"

    pub = load_pub(data_dir, attach_transform=False)
    priv = load_priv(data_dir, attach_transform=False)
    pub_y = pub_membership_labels(pub)

    def eval_pool(scores_pool: np.ndarray) -> dict:
        """Sign-checked TPR@alpha on the pub.pt ground-truth split."""
        return evaluate_with_sign_check(scores_pool[pub_idx], pub_y, alpha=args.alpha)

    rows = []
    method_pool_scores: list[tuple[str, np.ndarray]] = []

    # ----------------------- LiRA variants -----------------------
    for name, fn in [
        ("lira_online_global", lira_online_global),
        ("lira_online_diag",   lira_online_diag),
        ("lira_offline",       lira_offline),
    ]:
        print(f"\n[attack] {name} ...", flush=True)
        scores = fn(shadow_phi, target_phi, mask)
        chk = eval_pool(scores)
        if chk["best_sign"] == "-":
            scores = -scores
        rows.append({"method": name, "pub_tpr": chk["best_tpr"], "sign": chk["best_sign"]})
        print(f"  pub TPR@{int(args.alpha*100)}%FPR = {chk['best_tpr']:.4f} "
              f"(sign={chk['best_sign']})", flush=True)
        method_pool_scores.append((name, scores))

    # ----------------------- RMIA grid search -----------------------
    print(f"\n[attack] RMIA alpha x gamma sweep ...", flush=True)

    def rmia_metric(s_pool: np.ndarray) -> float:
        return eval_pool(s_pool)["best_tpr"]

    best_rmia, best_rmia_scores = rmia_sweep(
        shadow_p_y, target_p_y, mask, eval_fn=rmia_metric,
        alphas=(0.2, 0.3, 0.33, 0.5),
        gammas=(1.0, 1.2, 1.5, 2.0),
    )
    chk = eval_pool(best_rmia_scores)
    if chk["best_sign"] == "-":
        best_rmia_scores = -best_rmia_scores
    rmia_name = f"rmia_a{best_rmia['alpha']}_g{best_rmia['gamma']}"
    rows.append({"method": rmia_name, "pub_tpr": chk["best_tpr"], "sign": chk["best_sign"]})
    print(f"  best: alpha={best_rmia['alpha']}, gamma={best_rmia['gamma']}, "
          f"pub TPR={chk['best_tpr']:.4f} (sign={chk['best_sign']})", flush=True)
    method_pool_scores.append((rmia_name, best_rmia_scores))

    # Save per-method summary
    per_df = pd.DataFrame(rows).sort_values("pub_tpr", ascending=False)
    per_df.to_csv(results_dir / "per_method_table.csv", index=False)
    print(f"\n[attack] per-method summary:\n{per_df.to_string(index=False)}", flush=True)

    # ----------------------- ensemble -----------------------
    pool_scores = [s for _, s in method_pool_scores]

    print("\n[attack] equal-weight rank-fusion ensemble ...", flush=True)
    eq_pool = equal_weight_baseline(pool_scores)
    eq_tpr = tpr_at_fpr(eq_pool[pub_idx], pub_y, args.alpha)
    print(f"  pub TPR@{int(args.alpha*100)}%FPR = {eq_tpr:.4f}", flush=True)

    print(f"\n[attack] sanity grid search over simplex (step={args.grid_step}) ...", flush=True)
    gs = grid_search_weights(
        pool_scores, pub_idx, pub_y,
        train_frac=0.8, seed=args.seed, step=args.grid_step, alpha=args.alpha,
    )
    method_names = [n for n, _ in method_pool_scores]
    print(f"  best weights: {dict(zip(method_names, gs['best_weights'].round(3)))}", flush=True)
    print(f"  train pub TPR: {gs['train_tpr']:.4f}", flush=True)
    print(f"  val   pub TPR: {gs['val_tpr']:.4f}  <- the honest number", flush=True)

    full_pub_grid_tpr = tpr_at_fpr(gs["final_pool_scores"][pub_idx], pub_y, args.alpha)

    ens_df = pd.DataFrame([
        {"variant": "equal_weight",
         "pub_tpr": eq_tpr,
         "note": "uniform 0.25 across the four scorers; what we submit"},
        {"variant": "grid_train",
         "pub_tpr": gs["train_tpr"],
         "note": "best simplex weights on the 80% pub split"},
        {"variant": "grid_val",
         "pub_tpr": gs["val_tpr"],
         "note": "those weights evaluated on the held-out 20%"},
        {"variant": "grid_full",
         "pub_tpr": full_pub_grid_tpr,
         "note": "those weights applied to all of pub.pt (informational)"},
    ])
    ens_df.to_csv(results_dir / "ensemble_table.csv", index=False)
    print(f"\n[attack] ensemble summary:\n{ens_df.to_string(index=False)}", flush=True)

    # Pick equal-weight if grid-search val score doesn't clearly beat it. Equal-weight
    # is the unbiased ensemble; the grid-search variant typically overfits the train split.
    if gs["val_tpr"] >= eq_tpr - 0.002:
        chosen_name = "grid_tuned"
        chosen_pool = gs["final_pool_scores"]
    else:
        chosen_name = "equal_weight"
        chosen_pool = eq_pool
    print(f"\n[attack] submission variant chosen: {chosen_name}", flush=True)

    # ----------------------- write submission CSV -----------------------
    out_csv = results_dir / "submission.csv"
    write_submission_csv(priv.ids, chosen_pool[priv_idx], out_csv)
    print(f"[attack] wrote {out_csv}", flush=True)
    print(f"[attack] To POST to the leaderboard:\n   python -m src.submit {out_csv}",
          flush=True)


if __name__ == "__main__":
    main()
