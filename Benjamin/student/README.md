# Summer School Challenge — Predictive Uncertainty on iWildCam

You are given a starter codebase, a labeled training and validation set, and an
unlabeled `test_public` set. You will train a classifier, produce a
`submission.csv`, and upload it. The server scores your submission on four
metrics; the leaderboard aggregates them via Borda count.

## 1. Install

```bash
pip install -r requirements.txt
```

This installs `torch`, `torchvision`, `pandas`, `numpy`, `pillow`, `tqdm`.
No `wilds`, no `scikit-learn` — the local evaluator is self-contained.

## 2. Where to put the data

Unzip the release so the layout is:

```
challenge_data/
├── train/
│   ├── images/<uid>.jpg
│   └── labels.csv
├── val/
│   ├── images/<uid>.jpg
│   └── labels.csv
├── test_public/
│   └── images/<uid>.jpg
├── test_private/
│   └── images/<uid>.jpg
├── sample_submission.csv
└── class_mapping.json
```

- `train/labels.csv` and `val/labels.csv` have columns `uid, y, domain`.
- `val`'s `domain` is `id` (same camera locations as train, different days)
  or `ood` (held-out locations). Useful for split-conditional analysis.
- `test_public/` is the live leaderboard set; `test_private/` is the
  held-out final-score set. **Neither has labels** — predict on both and
  submit a single CSV. The server uses `test_public` rows for the
  leaderboard and `test_private` rows for final scoring.
- `sample_submission.csv` covers every uid across both test splits with
  uniform `1/K` probabilities — a working format reference.
- `class_mapping.json` carries `num_classes` (`K`) and the mapping between
  contiguous `[0, K-1]` labels and the original WILDS class ids.

## 3. Train the baseline

```bash
python -m student.train \
    --data-root /path/to/challenge_data \
    --output-dir runs/baseline \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --head-lr 1e-3 \
    --patience 3
```

This runs ResNet-50 with AdamW (lower LR on the pretrained backbone, higher
on the new classification head), a cosine LR schedule, and early stopping on
val NLL. After training, a single-scalar temperature `T` is fit on `val` via
LBFGS on NLL. Two checkpoints are written:

- `runs/baseline/model.pt`              — early-stopped best weights (`T = 1.0`)
- `runs/baseline/model_temp_scaled.pt`  — same weights, with the learned `T`

Both checkpoints share the same format — a dict containing `state_dict`,
`num_classes`, `temperature`, `backbone` (timm model id), and the resolved
`hyperparameters` dict. Pass `--backbone <timm-id>` to swap (e.g.
`--backbone resnet18` or `--backbone vit_base_patch16_224`); pass
`--pretrained` to start from timm's published weights.

## 4. Evaluate locally

```bash
python -m student.eval \
    --checkpoint runs/baseline/model_temp_scaled.pt \
    --data-root /path/to/challenge_data
```

Prints the same five numbers the server will compute:

- `accuracy`
- `ece` (15 equal-width bins on max-softmax)
- `nll` (mean negative log-likelihood)
- `brier` (mean Brier score)
- `misclassification_auroc` (score = max-softmax confidence, target = correct)

The local metrics are bit-for-bit equivalent to the server. Use them to
iterate before submitting.

## 5. Generate a submission

```bash
python -m student.predict \
    --checkpoint runs/baseline/model_temp_scaled.pt \
    --data-root /path/to/challenge_data \
    --output submission.csv
```

## 6. Submission format

**One row per `uid` across `test_public/images/` and `test_private/images/`.**
Columns:

```
uid, p_0, p_1, ..., p_{K-1}
```

- `p_k` is the predicted probability of class `k` (the remapped contiguous
  label space — see `class_mapping.json`).
- Probabilities must sum to ~1 per row (tolerance `1e-3`).
- Row order does not matter; the server joins on `uid`.
- One combined submission for both splits — `student.predict` does this by
  default. The server uses `test_public` uids for the live leaderboard and
  `test_private` uids for final scoring.

`challenge_data/sample_submission.csv` is a working template with uniform
`1/K` probabilities — it will be accepted by the server (just very badly
scored).

## 7. What to try

The proposal in `docs/summerschool_challenge.pdf` discusses what each metric
captures and points to families of methods worth trying. Concrete starting
points:

- **Temperature scaling** is already in the baseline; try alternative
  calibration objectives (Brier instead of NLL).
- **Deep ensembles** — train N models, average their probabilities.
- **MC dropout** — enable dropout at eval time, average over T forward passes.
- **Class re-weighting / focal loss** — iWildCam is long-tailed.

Most experimentation lives in `student/model.py` — the `Classifier` class
exposes `self.backbone`, `self.head`, and an `embed(x)` method, so you can
swap backbones, add projection heads, or override `forward` for MC sampling
without touching `train.py`. Keep `forward(x) → logits` and the `backbone /
head` split, and the rest of the pipeline still works.

Don't change the metric formulas in `student/metrics.py` — the server uses
the same ones.
