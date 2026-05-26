# Solution: Traffic Sign Label-Flipping Poisoning Assessment

This solution completes the label-flipping poisoning workflow for the traffic sign classifier.

The completed notebook:

- Prepares a compact GTSRB subset.
- Trains a clean baseline classifier.
- Flips a portion of `Stop` training labels to `Yield`.
- Retrains a poisoned model.
- Compares aggregate accuracy, source-class accuracy, and targeted misclassification behavior.
- Saves plots, checkpoints, and result tables under `results/` and `models/`.

Downloaded datasets and executed notebooks are ignored by git. The reference metrics, comparison image, and trained solution checkpoints are shareable so instructors can inspect the completed answer without rerunning the notebook.

The reference run included in this solution produced:

| Model | Clean Accuracy | Stop Accuracy | Stop-to-Yield Misclassification |
| --- | ---: | ---: | ---: |
| Clean baseline | 0.7667 | 0.8667 | 0.0000 |
| Label-flipped model | 0.7222 | 0.0000 | 0.8000 |

The clean and label-flipped checkpoints are intentionally shareable in git for the solution path.

## Optional Instructor Prep

To pre-generate the compact dataset and clean baseline checkpoint, run:

```bash
bash scripts/prepare_traffic_sign_assets.sh
```

Or run:

```bash
python scripts/prepare_traffic_sign_assets.py
```
