"""Rank-fusion ensemble.

Given several score vectors (each `[N_pool]`, higher = more member-like),
rank-normalize them and form a weighted convex combination:

    final[j] = sum_i w_i * (rank(s_i)[j] / N_pool)
    with     sum_i w_i = 1,  w_i >= 0.

Why ranks: different scorers have different scales (LiRA log-LR can reach the
hundreds, RMIA scores live in [0, 1]). Rank-normalization removes scale
mismatches and makes a convex combination meaningful. Carlini et al. (2022)
recommend rank-averaging when ensembling MIA scores.

We grid-search the weight simplex on a held-out 20% pub.pt split to detect
overfitting to pub idiosyncrasies and fall back to equal weights when the
grid-tuned weights don't generalize.
"""
from __future__ import annotations

from itertools import product
import numpy as np

from .eval import tpr_at_fpr


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Rank-normalize a [N] score array to [0, 1]; ties broken by stable sort."""
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(s, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return ranks / len(s)


def _enumerate_simplex(n_methods: int, step: float):
    """Yield all weight vectors on the (n_methods-1)-simplex with step granularity.

    For step=0.1 and n_methods=5 there are C(14, 4) = 1001 points, fast.
    """
    n_steps = int(round(1.0 / step))
    for combo in product(range(n_steps + 1), repeat=n_methods):
        if sum(combo) == n_steps:
            yield np.asarray(combo, dtype=np.float64) / n_steps


def grid_search_weights(
    pool_scores_list: list[np.ndarray],
    pub_idx: np.ndarray,
    pub_membership: np.ndarray,
    *,
    train_frac: float = 0.8,
    seed: int = 0,
    step: float = 0.1,
    alpha: float = 0.05,
):
    """Grid-search the simplex on a pub TRAIN split; report TRAIN + VAL TPR.

    Args:
        pool_scores_list: list of [N_pool] arrays, each with higher = member.
        pub_idx:          [N_pub] indices into pool that correspond to pub.pt.
        pub_membership:   [N_pub] binary {0, 1} from pub.pt's GT labels.
        train_frac:       fraction of pub used to fit weights; rest is held-out val.
        step:             grid spacing on the simplex (0.1 → 1001 points for 5 scorers).
        alpha:            FPR target (0.05 = TPR@5%FPR).

    Returns:
        dict with: best_weights, train_tpr, val_tpr,
                   final_pool_scores [N_pool] (rank-normalized convex combination),
                   train_indices_within_pub, val_indices_within_pub.
    """
    n_pool = pool_scores_list[0].shape[0]
    n_methods = len(pool_scores_list)
    assert all(s.shape == (n_pool,) for s in pool_scores_list)
    assert pub_membership.shape == pub_idx.shape

    # Rank-normalize each scorer GLOBALLY across the pool, so train/val share a scale.
    # NOTE: ranking across the full pool (not just pub) means priv samples influence
    # the rank of pub samples. That's fine — at submission time we score priv on the
    # same global rank, and we don't peek at priv's *labels* (which are None anyway).
    rank_pool = np.stack([rank_normalize(s) for s in pool_scores_list], axis=0)  # [M, N_pool]

    # Pub-only views for fitting / validating
    rank_pub = rank_pool[:, pub_idx]                                            # [M, N_pub]

    # Stratified 80/20 split on pub by membership label
    rng = np.random.default_rng(seed)
    pos_idx = np.where(pub_membership == 1)[0]
    neg_idx = np.where(pub_membership == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_train_pos = int(round(train_frac * len(pos_idx)))
    n_train_neg = int(round(train_frac * len(neg_idx)))
    train_local = np.concatenate([pos_idx[:n_train_pos], neg_idx[:n_train_neg]])
    val_local = np.concatenate([pos_idx[n_train_pos:], neg_idx[n_train_neg:]])
    rng.shuffle(train_local)
    rng.shuffle(val_local)

    y_train = pub_membership[train_local]
    y_val = pub_membership[val_local]
    rank_train = rank_pub[:, train_local]                                      # [M, N_train]
    rank_val = rank_pub[:, val_local]                                          # [M, N_val]

    best = {"w": None, "train_tpr": -1.0, "val_tpr": -1.0}
    for w in _enumerate_simplex(n_methods, step):
        train_score = (w[:, None] * rank_train).sum(axis=0)
        train_tpr = tpr_at_fpr(train_score, y_train, alpha)
        if train_tpr > best["train_tpr"]:
            val_score = (w[:, None] * rank_val).sum(axis=0)
            val_tpr = tpr_at_fpr(val_score, y_val, alpha)
            best.update(w=w, train_tpr=train_tpr, val_tpr=val_tpr)

    # Recompute final pool-wide scores under the best weights (rank-normalize again
    # to ensure the final array is monotone in [0, 1] for the leaderboard CSV).
    final_pool = (best["w"][:, None] * rank_pool).sum(axis=0)
    final_pool = rank_normalize(final_pool)

    return {
        "best_weights": best["w"],
        "train_tpr": best["train_tpr"],
        "val_tpr": best["val_tpr"],
        "final_pool_scores": final_pool,
        "train_indices_within_pub": train_local,
        "val_indices_within_pub": val_local,
    }


def equal_weight_baseline(pool_scores_list: list[np.ndarray]) -> np.ndarray:
    """Simple rank-average of all scorers (uniform weights)."""
    M = len(pool_scores_list)
    rank_pool = np.stack([rank_normalize(s) for s in pool_scores_list], axis=0)
    final = rank_pool.mean(axis=0)
    return rank_normalize(final)
