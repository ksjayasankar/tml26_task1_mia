"""RMIA scoring (Zarifzadeh, Liu, Shokri 2024 ICML; arXiv 2312.03262).

Robust Membership Inference Attack. For each challenge sample x and a random
population sample z drawn from the pool, compute the pairwise likelihood ratio:

    LR_theta(x, z) = (Pr(x | theta) / Pr(x))  /  (Pr(z | theta) / Pr(z))

where:
    Pr(x | theta) = softmax probability the target gives x its true class.
    Pr(x)         = "natural" probability for x; estimated as a weighted
                    blend of OUT-shadow probabilities (and an alpha-biased
                    IN estimate when no IN shadows are used).

The RMIA score for x is then `Pr_z [LR_theta(x, z) > gamma]`, the fraction
of population samples that x dominates. Higher = more member-like.

We aggregate over the K=8 augmented views by taking the mean per-shadow
probability across views first (per the paper's standard approach).

Hyperparameters:
    alpha — interpolation weight between OUT-only and IN+OUT estimates of
            Pr(x). The paper recommends ~0.3 for the offline setting and
            sweeps {0.2, 0.3, 0.33, 0.5}.
    gamma — threshold on the pairwise LR. Default 1.0 means "x at least as
            member-like as z." Sweep {1.0, 1.2, 1.5, 2.0}.

This module is pure CPU + numpy.
"""
from __future__ import annotations

import numpy as np


def _per_shadow_p_x(p_y_NSK: np.ndarray) -> np.ndarray:
    """Aggregate softmax_y across the K augmented views per (shadow, sample).

    Input:  p_y[N_shadows, N_pool, K]
    Output: p_per_shadow[N_shadows, N_pool] (mean over views)
    """
    return p_y_NSK.mean(axis=2)


def _natural_p_x(
    p_per_shadow: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """RMIA's estimator of Pr(x), the "natural" probability.

    p_OUT(x) = mean of p_per_shadow over OUT shadows of x.
    p_IN (x) = mean of p_per_shadow over IN  shadows of x.
    Pr(x)    = (1 - alpha) * p_OUT(x) + alpha * p_IN(x)

    When alpha = 0 this is the offline OUT-only estimator.
    The paper's offline-RMIA uses alpha in {0.2, 0.3, 0.33}.

    Returns: [N_pool] float64.
    """
    mask_b = mask.astype(bool)
    in_count = mask_b.sum(axis=0).astype(np.float64)
    out_count = (~mask_b).sum(axis=0).astype(np.float64)

    sum_out = (p_per_shadow * (~mask_b)).sum(axis=0)
    sum_in = (p_per_shadow * mask_b).sum(axis=0)
    p_out = sum_out / np.maximum(out_count, 1)
    p_in = sum_in / np.maximum(in_count, 1)
    return (1.0 - alpha) * p_out + alpha * p_in


def rmia_scores(
    shadow_p_y: np.ndarray,
    target_p_y: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = 0.33,
    gamma: float = 1.0,
    n_population: int = 4096,
    population_indices: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Compute RMIA scores for every sample in the pool.

    Args:
        shadow_p_y: [N_shadows, N_pool, K] softmax_y per (shadow, sample, view).
        target_p_y: [N_pool, K]            softmax_y from the target.
        mask:       [N_shadows, N_pool]    Bernoulli IN/OUT.
        alpha:      RMIA blending weight for Pr(x).
        gamma:      Threshold on pairwise LR.
        n_population: Number of random z samples to evaluate against per x.
                    Set to N_pool to use the full pool.
        population_indices: Optional array of pool indices to use as the
                    population. If None, drawn uniformly at random with `seed`.
        seed:       RNG seed for population sampling (when indices is None).

    Returns: [N_pool] float64 RMIA score (higher = more member-like).
    """
    N_shadows, N_pool, K = shadow_p_y.shape
    assert target_p_y.shape == (N_pool, K)

    # Per-shadow per-sample aggregate over views
    p_per_shadow = _per_shadow_p_x(shadow_p_y)            # [N_shadows, N_pool]
    target_p = target_p_y.mean(axis=1)                    # [N_pool]

    # Natural probability estimate
    p_x = _natural_p_x(p_per_shadow, mask, alpha=alpha)   # [N_pool]
    eps = 1e-12

    # Per-sample target ratio
    ratio = target_p / np.maximum(p_x, eps)               # [N_pool]

    # Population subsample of z's
    if population_indices is None:
        rng = np.random.default_rng(seed)
        if n_population >= N_pool:
            pop_idx = np.arange(N_pool)
        else:
            pop_idx = rng.choice(N_pool, size=n_population, replace=False)
    else:
        pop_idx = np.asarray(population_indices, dtype=np.int64)
    pop_ratio = ratio[pop_idx]                             # [n_pop]

    # For each x: fraction of z in population with ratio[x] / ratio[z] > gamma.
    # Equivalent to: fraction of z with ratio[z] < ratio[x] / gamma.
    # Vectorize via sort + searchsorted (faster than broadcasting for large pools).
    sorted_pop = np.sort(pop_ratio)
    threshold = ratio / gamma
    # Number of z with ratio[z] strictly < threshold[x]
    counts = np.searchsorted(sorted_pop, threshold, side="left")
    return counts.astype(np.float64) / max(len(sorted_pop), 1)


def rmia_sweep(
    shadow_p_y: np.ndarray,
    target_p_y: np.ndarray,
    mask: np.ndarray,
    eval_fn,
    alphas=(0.2, 0.3, 0.33, 0.5),
    gammas=(1.0, 1.2, 1.5, 2.0),
    n_population: int = 4096,
    seed: int = 0,
) -> tuple[dict, np.ndarray]:
    """Grid-search alpha x gamma; return best (config, scores) by `eval_fn`.

    `eval_fn(scores: [N_pool]) -> float` — typically pub TPR@5%FPR over the
    pub indices. Higher is better.
    """
    best = {"alpha": None, "gamma": None, "score": -np.inf}
    best_scores = None
    for a in alphas:
        for g in gammas:
            s = rmia_scores(
                shadow_p_y, target_p_y, mask,
                alpha=a, gamma=g, n_population=n_population, seed=seed,
            )
            metric = eval_fn(s)
            if metric > best["score"]:
                best.update(alpha=a, gamma=g, score=metric)
                best_scores = s
    return best, best_scores
