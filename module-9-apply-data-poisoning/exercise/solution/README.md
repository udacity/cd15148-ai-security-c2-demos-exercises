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

The published reference numbers are means and standard deviations across three seeds. A fresh run on a clean machine should land inside each band.

| Model | Clean Accuracy | Stop-Class Accuracy | Stop → Yield Misclassification | Mean Confidence |
| --- | :---: | :---: | :---: | :---: |
| Clean baseline | 0.94 ± 0.03 | ~0.92 | 0.00 ± 0.01 | high |
| Label-flipped model | 0.80 ± 0.01 | 0.03 ± 0.05 | 0.89 ± 0.08 | high |

The label-flipped model collapses to ~0% accuracy on `Stop` while preserving most of its aggregate accuracy. The targeted misclassification rate (~89%) reflects how often clean `Stop` validation images are predicted as `Yield` — no trigger needed at inference, because the corruption is in the labels alone.

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

