# Solution: Traffic Sign Label-Flipping Poisoning Assessment

This solution completes the label-flipping poisoning workflow for the traffic sign classifier.

The completed notebook:

- Prepares a compact GTSRB subset (200 train / 100 val per class, six classes).
- Trains a clean baseline classifier for 8 epochs.
- Flips 75% of `Stop` training labels to `Yield`.
- Retrains a poisoned model with the same recipe.
- Compares aggregate accuracy, source-class accuracy, and targeted misclassification behavior.
- Saves plots, checkpoints, and result tables under `results/` and `models/`.

Downloaded datasets and executed notebooks are ignored by git. The reference metrics (`results/label_flip_metrics_baseline.csv`), comparison image (`results/label_flip_examples_baseline.png`), and trained solution checkpoints are shareable so instructors can inspect the completed answer without rerunning the notebook. Notebook runs write timestamped outputs (`*_YYYYMMDD_HHMMSS.*`) alongside the baselines so reruns never overwrite the reference.

## Expected results

The notebook is seeded (`torch.manual_seed(7)`, `np.random.seed(7)`), so the default run is reproducible. These are the exact figures it produces, matching `results/label_flip_metrics_baseline.csv` — verified on the classroom GPU image pins (Python 3.12.13, torch 2.5.1) on Apple Silicon/MPS, where a fresh run also regenerated both checkpoints byte-identically.

| Model | Clean Accuracy | Stop-Class Accuracy | Stop → Yield Misclassification | Mean Confidence |
| --- | :---: | :---: | :---: | :---: |
| Clean baseline | 0.952 | 0.910 | 0.000 | 0.966 |
| Label-flipped model | 0.802 | 0.000 | 0.990 | 0.919 |

Change the seed or the hardware and the figures move. Across three different seeds the same recipe lands at 0.94 ± 0.03 clean accuracy, 0.03 ± 0.05 `Stop` accuracy and 0.89 ± 0.08 targeted misclassification — use those bands as the tolerance, not the exact numbers above.

The label-flipped model collapses to 0% accuracy on `Stop` while preserving most of its aggregate accuracy. The targeted misclassification rate (99%) reflects how often clean `Stop` validation images are predicted as `Yield` — no trigger needed at inference, because the corruption is in the labels alone.

The clean and label-flipped checkpoints are intentionally shareable in git for the solution path.

## Optional instructor prep

To pre-generate the compact dataset and clean baseline checkpoint, run:

```bash
bash scripts/prepare_traffic_sign_assets.sh
```

Or run:

```bash
python scripts/prepare_traffic_sign_assets.py
```

