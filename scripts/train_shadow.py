#!/usr/bin/env python
"""Train one shadow ResNet-18 for the LiRA / RMIA pipeline.

Architecture is identical to the target (3x3 conv1, no maxpool, fc=512->9),
loaded via `src.model.build_resnet18()`. Training data is the IN-subset of
`pool = pub + priv` defined by row `--shadow_id` of `masks/in_mask.npy`.

Designed to be invoked as the executable in an HTCondor `queue 64` array job:
each Process gets a different shadow_id. Reads + writes happen against the
mounted home directory (`+WantGPUHomeMounted = true`), so there's no need to
transfer datasets/weights via condor's file transfer.

Standard CIFAR-style augmentation during training: random crop pad=4 +
horizontal flip. PathMNIST being rotation-invariant we *could* add 90/180/270
rotations, but doing so risks mismatching the target's training recipe — the
default (crop + h-flip) is the conservative starting point. If shadow accuracy
diverges materially from target accuracy after the first few jobs, retune.

Mixed precision (autocast + GradScaler) is enabled for memory headroom; on a
P100 (no tensor cores) the speedup is modest but the memory savings let us
push batch size higher if needed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision.transforms as T

# Make `from src.* import` work when this file runs as the executable inside
# the docker container with cwd = scratch dir.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import load_pub, load_priv, MEAN, STD  # noqa: E402
from src.model import build_resnet18  # noqa: E402


def make_train_transform() -> T.Compose:
    """Augmentation pipeline for shadow training.

    Input: a tensor (3, 28, 28) per-pixel in [0, 1] (the format inside pub/priv .pt).
    Output: a tensor (3, 32, 32), augmented and normalized.
    """
    return T.Compose([
        T.Resize(32),
        T.Pad(4),                    # 32 + 8 = 40
        T.RandomCrop(32),            # crop back to 32 with shift
        T.RandomHorizontalFlip(p=0.5),
        T.Normalize(mean=MEAN, std=STD),
    ])


def make_eval_transform() -> T.Compose:
    """Same as src.dataset.default_transform — no augmentation."""
    return T.Compose([
        T.Resize(32),
        T.Normalize(mean=MEAN, std=STD),
    ])


def _collate_drop_membership(batch):
    """Drop the membership field; priv has None for it (unstackable by default collate)."""
    ids = [b[0] for b in batch]
    imgs = torch.stack([b[1] for b in batch], dim=0)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return ids, imgs, labels


def train_one_shadow(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[shadow] WARN: no CUDA; training on CPU will be very slow.")
    torch.manual_seed(args.seed + args.shadow_id)
    np.random.seed(args.seed + args.shadow_id)

    base = Path(args.base_dir)
    data_dir = base / "data"
    weights_dir = base / "weights"
    masks_dir = base / "masks"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets WITHOUT a pre-attached transform; we set transforms manually.
    pub = load_pub(data_dir, attach_transform=False)
    priv = load_priv(data_dir, attach_transform=False)
    pool = ConcatDataset([pub, priv])
    n_pub, n_priv = len(pub), len(priv)
    n_pool = n_pub + n_priv
    print(f"[shadow {args.shadow_id}] pool: {n_pool} ({n_pub} pub + {n_priv} priv)")

    # Apply train-time transform on both halves of the concat.
    train_transform = make_train_transform()
    pub.transform = train_transform
    priv.transform = train_transform

    # Load IN/OUT mask
    mask_path = masks_dir / "in_mask.npy"
    mask = np.load(mask_path)
    if mask.shape[0] <= args.shadow_id:
        sys.exit(f"shadow_id {args.shadow_id} out of range for mask shape {mask.shape}")
    in_row = mask[args.shadow_id].astype(bool)
    in_indices = np.flatnonzero(in_row)
    print(f"[shadow {args.shadow_id}] training on {len(in_indices)} samples "
          f"(IN-mask sum) out of {n_pool}")

    train_subset = Subset(pool, in_indices)
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=(device == "cuda"),
        collate_fn=_collate_drop_membership,
    )

    model = build_resnet18().to(device)
    model.train()

    optim = torch.optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=False,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    use_amp = (device == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start = time.time()
    losses, accs = [], []
    for ep in range(args.epochs):
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        ep_start = time.time()
        for _ids, imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss = F.cross_entropy(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optim.step()
            ep_loss += float(loss.detach()) * imgs.size(0)
            ep_correct += int((logits.argmax(-1) == labels).sum())
            ep_total += imgs.size(0)
        sched.step()
        ep_time = time.time() - ep_start
        losses.append(ep_loss / max(1, ep_total))
        accs.append(ep_correct / max(1, ep_total))
        if ep == 0 or (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            print(f"[shadow {args.shadow_id}] epoch {ep+1:>3}/{args.epochs}  "
                  f"loss={losses[-1]:.4f}  acc={accs[-1]:.4f}  ({ep_time:.1f}s)")

    elapsed = time.time() - start
    print(f"[shadow {args.shadow_id}] training done in {elapsed/60:.2f} min")

    # Save weights
    out_path = weights_dir / f"shadow_{args.shadow_id:03d}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"[shadow {args.shadow_id}] weights -> {out_path}")

    # Quick OUT-set sanity check: accuracy on samples not in this shadow's training set
    # using the eval transform. This is the "shadow vs target" diagnostic.
    pub.transform = make_eval_transform()
    priv.transform = make_eval_transform()
    out_indices = np.flatnonzero(~in_row)
    out_subset = Subset(pool, out_indices)
    out_loader = DataLoader(
        out_subset, batch_size=512, shuffle=False, num_workers=0,
        collate_fn=_collate_drop_membership,
    )
    model.eval()
    out_correct, out_total = 0, 0
    with torch.no_grad():
        for _ids, imgs, labels in out_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
            else:
                logits = model(imgs)
            out_correct += int((logits.argmax(-1) == labels).sum())
            out_total += imgs.size(0)
    out_acc = out_correct / max(1, out_total)
    print(f"[shadow {args.shadow_id}] OUT-set accuracy: {out_acc:.4f} (n={out_total})")

    summary = {
        "shadow_id": args.shadow_id,
        "n_train": int(in_row.sum()),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "final_train_loss": losses[-1],
        "final_train_acc": accs[-1],
        "out_set_acc": out_acc,
        "elapsed_min": elapsed / 60.0,
        "out_path": str(out_path),
    }
    summary_path = weights_dir / f"shadow_{args.shadow_id:03d}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shadow_id", type=int, required=True,
                   help="Row index into masks/in_mask.npy")
    p.add_argument("--base_dir", type=str, required=True,
                   help="Repo root on the cluster (e.g. /home/$USER/tml26-mia)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_amp", action="store_true",
                   help="Disable mixed precision (fall back to fp32).")
    args = p.parse_args()
    train_one_shadow(args)


if __name__ == "__main__":
    main()
