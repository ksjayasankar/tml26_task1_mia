"""Evaluation harness.

Computes TPR@FPR=alpha (default 5%) on pub.pt using its ground-truth
membership labels. Uses sklearn for the ROC curve.

matplotlib is lazy-imported inside `plot_roc` so the rest of this module
works even when matplotlib isn't installed (it's only needed for ROC PNGs).

Includes a sign-direction safety check: for any score vector, evaluates
both `score` and `-score`. This is a debugging tool only and applies on
pub.pt where ground truth is known. Sign decisions are committed in code
before any priv.pt submission — never auto-flip on the leaderboard side.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, alpha: float = 0.05) -> float:
    """Return the TPR at the largest threshold with FPR <= alpha.

    `labels` is binary {0, 1} where 1 = member. `scores` higher => member.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = fpr <= alpha
    if not valid.any():
        return 0.0
    return float(tpr[valid].max())


def evaluate_with_sign_check(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Evaluate score and -score; return both TPRs plus a flag for the chosen sign.

    Returns:
        {
          "tpr_pos": float,         # TPR@alpha for `scores`
          "tpr_neg": float,         # TPR@alpha for `-scores`
          "best_sign": "+" or "-",  # whichever has higher TPR@alpha
          "best_tpr":  float,
          "alpha": float,
          "n_members": int,
          "n_nonmembers": int,
        }
    """
    labels = np.asarray(labels).astype(int)
    if set(np.unique(labels).tolist()) - {0, 1}:
        raise ValueError("labels must be binary {0, 1}")
    tpr_pos = tpr_at_fpr(scores, labels, alpha=alpha)
    tpr_neg = tpr_at_fpr(-scores, labels, alpha=alpha)
    best_sign = "+" if tpr_pos >= tpr_neg else "-"
    return {
        "tpr_pos": tpr_pos,
        "tpr_neg": tpr_neg,
        "best_sign": best_sign,
        "best_tpr": max(tpr_pos, tpr_neg),
        "alpha": alpha,
        "n_members": int((labels == 1).sum()),
        "n_nonmembers": int((labels == 0).sum()),
    }


def pub_membership_labels(pub_dataset) -> np.ndarray:
    """Extract the binary membership label vector from a MembershipDataset (pub.pt)."""
    m = np.asarray(pub_dataset.membership)
    if m.dtype == object or any(x is None for x in m):
        raise ValueError("pub.pt should have membership populated; got None entries.")
    return m.astype(int)


def plot_roc(scores_dict: dict[str, np.ndarray], labels: np.ndarray, out_path: Path | str):
    """Plot multi-curve ROC for sanity inspection. scores_dict: {name: score_array}.

    matplotlib is lazy-imported here so the rest of this module works in
    minimal environments where matplotlib isn't installed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, s in scores_dict.items():
        fpr, tpr, _ = roc_curve(labels, s)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, label="random")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ROC curves on pub.pt")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
