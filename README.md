# tml26_task1_mia

Membership Inference Attack on a pretrained ResNet-18 image classifier.
The pipeline trains 64 shadow ResNet-18s, extracts logit-scaled scores under
8 dihedral augmentations, runs three LiRA variants and an RMIA grid search,
and combines the four scores via equal-weight rank-fusion.

**Best leaderboard result reproduced by this code: TPR@5%FPR = 0.063765 on
the 30% public split (rank #1 at submission time, margin +0.0019 over the
prior leader).**

References:
- Carlini et al. 2022, *Membership Inference Attacks From First Principles* — arXiv:2112.03570
- Zarifzadeh et al. 2024, *Low-Cost High-Power Membership Inference Attacks* (RMIA) — arXiv:2312.03262
- Shokri et al. 2017, *Membership Inference Attacks against Machine Learning Models*

---

## Repository layout

```
tml26_task1_mia/
├── README.md                       this file
├── requirements.txt
├── .gitignore
├── src/                            importable modules
│   ├── __init__.py
│   ├── dataset.py                  pub.pt / priv.pt loaders + transforms
│   ├── model.py                    modified ResNet-18 builder + target loader
│   ├── augmentations.py            dihedral D_4 group + augmented forward pass
│   ├── lira.py                     LiRA online-global, online-diag, offline
│   ├── rmia.py                     RMIA + alpha/gamma sweep
│   ├── ensemble.py                 rank-fusion + simplex grid search
│   ├── eval.py                     TPR@5%FPR with sign-direction safety check
│   └── submit.py                   POST a CSV to the leaderboard server
├── scripts/                        entry points
│   ├── make_masks.py               generates masks/in_mask.npy
│   ├── train_shadow.py             trains one shadow (called per Process by HTCondor)
│   ├── extract_signals.py          forwards target + shadows -> signals/*.npz
│   └── attack.py                   LiRA + RMIA + ensemble -> results/submission.csv
└── condor/                         HTCondor submit specs
    ├── make_masks.sub
    ├── train_shadows.sub           queue 64
    ├── extract_signals.sub
    └── attack.sub
```

Runtime artefacts (`data/`, `weights/`, `masks/`, `signals/`, `results/`, `logs/`)
are gitignored and regenerable from the scripts.

---

## Setup

### Dependencies

```bash
pip install -r requirements.txt
```

On the cluster's `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` docker image,
torch + numpy ship by default and the remaining packages are installed lazily
into `~/.local/...` by `scripts/attack.py`'s `_ensure_deps()` bootstrap.

### Data

Download the three task files from HuggingFace into `data/`:

```bash
mkdir -p data
cd data
wget https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt
wget https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt
wget https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt
cd ..
```

`pub.pt` and `priv.pt` are pickled instances of `MembershipDataset` and require
the `__main__` namespace to contain that class definition; `src/dataset.py`
registers it on import, so any of the provided scripts loads the .pt files
without further setup.

### Leaderboard API key

The submission server (`http://34.63.153.158/submit/01-mia`) requires an
`X-API-Key` header. Put your key in `.env` at the repo root:

```bash
echo 'TML_API_KEY=<your-key-from-CMS>' > .env
chmod 600 .env
```

`.env` is gitignored. `src/submit.py` loads it via `python-dotenv`.

---

## Reproducing the best result end-to-end

The pipeline is GPU-bound. We ran it on the Saarland HPC cluster (HTCondor
+ docker, Tesla P100). The four cluster jobs below correspond exactly to the
four `condor/*.sub` files; on a different scheduler (Slurm, single workstation)
the entry points are the four `scripts/*.py` files with the same arguments.

Total wall-clock from a clean checkout: roughly 1 hour with 16 parallel GPU
slots; ~3 hours with 4 slots. Total compute: ~12 GPU-hours.

### Step 1 — generate the IN/OUT mask (~5 s, CPU)

```bash
condor_submit condor/make_masks.sub
# OR locally:
python scripts/make_masks.py --n_shadows 64 --seed 0
```

Produces `masks/in_mask.npy` (shape `[64, 28000]`, Bernoulli(0.5), seed 0)
and `masks/pool_layout.json` (records `n_pub=14000`, `n_priv=14000`).

### Step 2 — train the 64 shadow ResNet-18s (~30-45 min wall-clock with 16 parallel slots)

```bash
condor_submit condor/train_shadows.sub      # queue 64 array job
```

Each shadow trains for 80 epochs with SGD(lr=0.1, momentum=0.9, weight_decay=5e-4),
cosine LR schedule to 0, batch 128, AMP, on its IN-mask subset (~14k of the
28k pool). Single-shadow wall: ~9-10 min on a Tesla P100. Output:
`weights/shadow_000.pt` … `weights/shadow_063.pt` plus matching `.json`
metadata files.

### Step 3 — extract per-(model, sample, view) signals (~5-15 min, 1 GPU)

```bash
condor_submit condor/extract_signals.sub
```

Runs the target model and all 64 shadows on the 28k-sample pool under each
of the 8 dihedral D_4 augmentations. Saves `signals/shadow_signals.npz`
(~115 MB) and `signals/target_signals.npz` (~1.8 MB) with both the
logit-scaled `phi = log(p_y / (1-p_y))` and raw softmax `p_y` arrays.

### Step 4 — score, ensemble, write submission CSV (~30 s, CPU)

```bash
condor_submit condor/attack.sub
# OR locally (after pulling signals down):
python scripts/attack.py --base_dir .
```

Runs the three LiRA variants (online global, online diagonal, offline) and
the RMIA alpha/gamma grid search, sign-checks each on `pub.pt`, rank-normalises,
and equal-weight averages. Sanity-checks against a simplex grid search on an
80/20 pub split — if grid-tuned weights overfit (validation TPR drops vs.\
the train fit), the driver falls back to equal weights. Writes
`results/submission.csv`.

### Step 5 — submit to the leaderboard

```bash
python -m src.submit results/submission.csv
```

Expected response: `{'submission_id': N, 'status': 'success', ...}`. The
server enforces a 60-min cooldown after a successful submission, 2 min after
a failed one. The leaderboard (`http://34.63.153.158/leaderboard_page`)
keeps each team's best score, so a worse-than-best resubmission is harmless.

---

## What was done in each step (brief)

- **Bernoulli(0.5) IN/OUT mask** randomises which half of the combined
  `pub.pt ∪ priv.pt` pool each shadow trains on. Each sample lands IN for
  ~32 of the 64 shadows and OUT for the other ~32, automatically yielding
  per-sample IN/OUT calibrators without sample-specific shadow training.
- **Shadow architecture matches the target exactly** (modified ResNet-18:
  3×3 conv1, no maxpool, fc=512→9). Shadow training recipe matches a
  conservative CIFAR-style ResNet-18 setup; the resulting OUT-set accuracy
  (mean ≈0.977) sits within ~1.2 pp of the target's non-member accuracy
  (0.989), supporting the recipe-match assumption. We include `priv.pt`
  samples in the shadow training pool — LiRA's threat model explicitly
  permits the attacker to train shadows on challenge points (Carlini et
  al. 2022); excluding them would degrade LiRA-online to LiRA-offline for
  every priv sample.
- **Dihedral D_4 augmentations (K=8 views)** at signal-extraction time
  tighten LiRA's per-example Gaussian fits. Histopathology-style images
  have no canonical orientation, so flips and 90-degree rotations are
  label-preserving.
- **Logit-scaling** `phi = log(p_y / (1 - p_y)) = logit_y - logsumexp(logit_others)`
  makes per-sample softmax confidences approximately Gaussian, which is
  what LiRA's likelihood-ratio test expects.
- **LiRA-online with global variance** pools the squared residuals across
  all (sample, IN-shadow, view) triples to estimate one shared
  `sigma_in^2`, `sigma_out^2`. This is the most stable variant when the
  per-example IN/OUT counts (~32 each) are too small for reliable
  per-example covariance estimation.
- **LiRA-online with per-example diagonal variance** lets `sigma^2[n]` vary
  per sample but assumes independence across the K views.
- **LiRA-offline** uses only the OUT distribution and scores each sample
  by its log-density under that one Gaussian. Its natural sign is
  inverted vs. the online variants; we verify and correct it on
  `pub.pt` ground truth before ensembling.
- **RMIA** (Zarifzadeh et al. 2024) compares the target's softmax_y(x_n)
  against an alpha-blended IN/OUT shadow average, then scores x_n by the
  fraction of pool samples z it dominates by margin gamma. We grid-search
  alpha ∈ {0.2, 0.3, 0.33, 0.5} and gamma ∈ {1.0, 1.2, 1.5, 2.0}; the
  empirical best on `pub.pt` is alpha=0.5, gamma=1.2.
- **Equal-weight rank-fusion ensemble** rank-normalises each scorer
  globally over the pool, averages with uniform weights w_i=0.25, and
  rank-normalises again. We retain equal weights because a simplex grid
  search (step 0.05, 1771 candidate weight vectors) on an 80/20 pub.pt
  split overfit (0.071 train vs.\ 0.062 validation TPR — a clear 0.009
  generalisation gap).

---

## Per-step pub-set TPR@5%FPR (for reference)

| Stage / method                     | pub.pt TPR@5%FPR |
|------------------------------------|------------------|
| Random floor                        | 0.0500            |
| LiRA-online, global variance        | 0.0629            |
| LiRA-online, per-example diagonal   | 0.0487            |
| LiRA-offline, global variance       | 0.0531            |
| RMIA (alpha=0.5, gamma=1.2)         | 0.0626            |
| **Equal-weight rank-fusion (4)**    | **0.0656**        |
| **Public leaderboard (30% priv)**   | **0.0638**        |

---

## Notes on cluster vs.\ laptop execution

- The four `condor/*.sub` files correspond one-to-one with the four
  pipeline scripts. Adapt the `arguments` line for a different scheduler.
- The HTCondor docker image ships only `torch + numpy`; `attack.py` and
  `make_masks.py` invoke a `pip install --user` of `pandas` and
  `scikit-learn` on first run, persisting into NFS-mounted `~/.local`.
- The submission step (`python -m src.submit ...`) requires only
  `requests` and `python-dotenv` and is typically run from a laptop; the
  cluster has no outbound network restrictions, so it also runs there.
