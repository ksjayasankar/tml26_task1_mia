"""LiRA scoring (Carlini, Chien, Nasr, Song, Terzis, Tramer 2022; arXiv 2112.03570).

Given:
  - shadow_phi[N_shadows, N_pool, K]   logit-scaled scores per (shadow, sample, view)
  - target_phi[N_pool, K]              logit-scaled scores from the target model
  - mask[N_shadows, N_pool]            Bernoulli IN/OUT (1 = sample IN shadow's training)

For each sample j we collect:
  IN[j]  = shadow_phi[mask[:,j] == 1, j, :]  shape ~[N_in_j, K]
  OUT[j] = shadow_phi[mask[:,j] == 0, j, :]  shape ~[N_out_j, K]

Fit per-example Gaussians to IN[j] and OUT[j] (online), or only to OUT[j]
(offline). Score the target's K-dim phi vector under both densities and take
the log likelihood ratio.

Variance handling — three flavours; we run all three and pick the best on
the pub.pt held-out:

1. "global"   pool the K-dim covariance across ALL samples; per-example just
   replaces mu. Most stable when N_in / N_out are small (typical for our
   N_shadows=64 -> ~32 IN samples for an 8-D Gaussian).

2. "diag"     per-example diagonal covariance (assume independence across
   the K augmented views). Cheap, often best.

3. "per_example" full per-example KxK covariance with Ledoit-Wolf shrinkage.
   Most expressive but noisy at small N_in.
"""
from __future__ import annotations

import numpy as np


_LOG_2PI = float(np.log(2 * np.pi))


def _log_gaussian_density(x_NK: np.ndarray, mu_NK: np.ndarray, var_NK: np.ndarray) -> np.ndarray:
    """log N(x; mu, diag(var)) for diagonal-covariance Gaussian.

    All inputs are [N, K]; returns [N].
    """
    var = np.clip(var_NK, 1e-12, None)
    diff_sq = (x_NK - mu_NK) ** 2
    log_norm = -0.5 * (np.log(var) + _LOG_2PI)
    return (log_norm - 0.5 * diff_sq / var).sum(axis=1)


def lira_online_diag(
    shadow_phi: np.ndarray,
    target_phi: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """LiRA-online with diagonal per-example covariance.

    Returns: [N_pool] log-likelihood-ratio scores. Higher = more likely member.
    """
    N_shadow, N_pool, K = shadow_phi.shape
    assert target_phi.shape == (N_pool, K)
    assert mask.shape == (N_shadow, N_pool)
    mask_b = mask.astype(bool)
    in_count = mask_b.sum(axis=0)
    out_count = (~mask_b).sum(axis=0)
    if in_count.min() < 2 or out_count.min() < 2:
        raise RuntimeError(
            f"some samples have fewer than 2 IN or 2 OUT shadows; "
            f"min in={in_count.min()} min out={out_count.min()}. Increase N_shadows."
        )

    # Compute per-example mean and var via masked sums.
    # Trick: use mask as float weights so we can broadcast.
    w_in = mask_b.astype(np.float64)                     # [N_shadow, N_pool]
    w_out = 1.0 - w_in
    # E[phi]_in[j, k] = sum_i w_in[i,j] * phi[i,j,k] / in_count[j]
    sum_in = (w_in[:, :, None] * shadow_phi).sum(axis=0)            # [N_pool, K]
    sum_out = (w_out[:, :, None] * shadow_phi).sum(axis=0)
    mean_in = sum_in / in_count[:, None]
    mean_out = sum_out / out_count[:, None]
    sq_in = (w_in[:, :, None] * (shadow_phi - mean_in[None]) ** 2).sum(axis=0)
    sq_out = (w_out[:, :, None] * (shadow_phi - mean_out[None]) ** 2).sum(axis=0)
    var_in = sq_in / np.maximum(in_count[:, None] - 1, 1)
    var_out = sq_out / np.maximum(out_count[:, None] - 1, 1)

    log_p_in = _log_gaussian_density(target_phi, mean_in, var_in)
    log_p_out = _log_gaussian_density(target_phi, mean_out, var_out)
    return (log_p_in - log_p_out).astype(np.float64)


def lira_online_global(
    shadow_phi: np.ndarray,
    target_phi: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """LiRA-online with GLOBAL diagonal variance shared across samples.

    Per-example mean (depends on j); ONE global variance per view k pooled
    across all samples. This is Carlini's "fixed variance" variant — most
    stable when per-example variance estimates are noisy from small N.
    """
    N_shadow, N_pool, K = shadow_phi.shape
    mask_b = mask.astype(bool)
    in_count = mask_b.sum(axis=0)
    out_count = (~mask_b).sum(axis=0)

    w_in = mask_b.astype(np.float64)
    w_out = 1.0 - w_in
    mean_in = (w_in[:, :, None] * shadow_phi).sum(axis=0) / in_count[:, None]
    mean_out = (w_out[:, :, None] * shadow_phi).sum(axis=0) / out_count[:, None]
    # Pool squared deviations across all samples for IN and OUT separately
    sq_in_all = (w_in[:, :, None] * (shadow_phi - mean_in[None]) ** 2).sum()
    sq_out_all = (w_out[:, :, None] * (shadow_phi - mean_out[None]) ** 2).sum()
    var_in_global = sq_in_all / max(in_count.sum() * K - K, 1)
    var_out_global = sq_out_all / max(out_count.sum() * K - K, 1)
    var_in = np.full((N_pool, K), var_in_global)
    var_out = np.full((N_pool, K), var_out_global)

    log_p_in = _log_gaussian_density(target_phi, mean_in, var_in)
    log_p_out = _log_gaussian_density(target_phi, mean_out, var_out)
    return (log_p_in - log_p_out).astype(np.float64)


def lira_offline(
    shadow_phi: np.ndarray,
    target_phi: np.ndarray,
    mask: np.ndarray,
    fixed_variance: bool = False,
) -> np.ndarray:
    """LiRA-offline: only fit OUT-distribution; one-sided z-score.

    Returns: [N_pool] scores. Higher = more likely member.
    """
    N_shadow, N_pool, K = shadow_phi.shape
    mask_b = mask.astype(bool)
    out_count = (~mask_b).sum(axis=0)
    w_out = (~mask_b).astype(np.float64)
    mean_out = (w_out[:, :, None] * shadow_phi).sum(axis=0) / out_count[:, None]
    sq_out = (w_out[:, :, None] * (shadow_phi - mean_out[None]) ** 2).sum(axis=0)
    if fixed_variance:
        var_out_global = sq_out.sum() / max(out_count.sum() * K - K, 1)
        var_out = np.full((N_pool, K), var_out_global)
    else:
        var_out = sq_out / np.maximum(out_count[:, None] - 1, 1)

    # One-sided density: how likely is target_phi under the OUT model?
    # Members will be UNlikely under OUT, so we return -log p_out.
    log_p_out = _log_gaussian_density(target_phi, mean_out, var_out)
    return (-log_p_out).astype(np.float64)
