"""Dihedral group D_4 augmented-query forward pass.

For each sample x and each of the 8 dihedral transformations
(identity, h-flip, v-flip, 90/180/270 rotations, and h-flip composed with
rot90/rot270), query the model on g_k(x) and stack the per-view logits.
This vector of K=8 augmented-view scores is the input that the LiRA Gaussian
fits and the RMIA pairwise comparison consume downstream.

Histopathology images (PathMNIST and similar) have no canonical orientation,
so D_4 is a biologically valid invariance to exploit.

Note: applying dihedral symmetries AFTER the standard
`Resize(32) + Normalize` transform is mathematically correct because per-channel
normalization commutes with horizontal/vertical flips and 90-degree rotations
(spatial layout doesn't matter for per-channel mean/std subtraction).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader


# Dihedral group D_4 (8 elements). Each op acts on a tensor batch [B, 3, H, W].
DIHEDRAL_OPS: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
    ("id", lambda x: x),
    ("hflip", TF.hflip),
    ("vflip", TF.vflip),
    ("rot90", lambda x: TF.rotate(x, 90)),
    ("rot180", lambda x: TF.rotate(x, 180)),
    ("rot270", lambda x: TF.rotate(x, 270)),
    ("hflip_rot90", lambda x: TF.rotate(TF.hflip(x), 90)),
    ("hflip_rot270", lambda x: TF.rotate(TF.hflip(x), 270)),
]


def _collate_drop_membership(batch):
    """DataLoader collate that ignores the membership field.

    `priv.pt` has membership=None for every sample; the default collate chokes
    on None. We don't need membership inside the forward pass anyway -- callers
    pull it from the dataset directly when needed.
    """
    ids = [b[0] for b in batch]
    imgs = torch.stack([b[1] for b in batch], dim=0)
    labels = [b[2] for b in batch]
    return ids, imgs, labels


@torch.no_grad()
def forward_with_augs(
    model: torch.nn.Module,
    dataset,
    ops: list[tuple[str, Callable]] = DIHEDRAL_OPS,
    device: str = "cpu",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward each sample through `model` once per op in `ops`.

    Returns:
        ids:    [N] sample-id array (matches dataset.ids order).
        logits: [N, K, C] float32, where K=len(ops) augmented views.
        labels: [N] true-class labels.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=_collate_drop_membership,
    )
    ids_chunks, logits_chunks, labels_chunks = [], [], []
    for ids, imgs, labels in loader:
        imgs = imgs.to(device)
        per_op = []
        for _, op_fn in ops:
            aug = op_fn(imgs)
            logits = model(aug)
            per_op.append(logits.detach().cpu())
        stacked = torch.stack(per_op, dim=1).numpy().astype(np.float32)  # [B, K, C]
        ids_chunks.append(np.asarray(ids))
        logits_chunks.append(stacked)
        labels_chunks.append(np.asarray(labels))
    ids = np.concatenate(ids_chunks, axis=0)
    logits = np.concatenate(logits_chunks, axis=0)
    labels = np.concatenate(labels_chunks, axis=0)
    return ids, logits, labels
