# Demo: Build a Robustness Evaluation Pipeline for an Image Classification Model

This demo evaluates CIFAR-10 ResNet-18 image classifiers under clean, environmental, and adversarial test conditions. It produces a comparative scorecard with accuracy, confidence degradation, perturbation tolerance, attack success rate, and an operational robustness score.

The first run downloads CIFAR-10 and creates a balanced 1,000-image validation subset. Trained ResNet-18 classroom checkpoints ship in `models/` and are loaded on every run, so no run trains by default. Set `RUN_FROM_CHECKPOINT = False` in the notebook to retrain from scratch on the full 50,000-image CIFAR-10 train split. Retraining overwrites the checkpoints committed to git.

> **Pre-cached in the classroom workspace.** When `C2_ASSET_CACHE` is set, this download is
> read from that shared cache and nothing is fetched at run time. Unset — a plain `git clone` —
> everything downloads into this module's own folders exactly as described above.

## Run the Demo

```bash
pip install -r requirements.txt
python scripts/run_robustness_demo.py
```

For a faster smoke test, run:

```bash
python scripts/run_robustness_demo.py --train-per-class 10 --val-per-class 10 --epochs 1 --max-eval 20 --skip-art
```

## Outputs

Each run writes timestamped files to `results/`, so reruns don't overwrite each other:

- `results/robustness_scorecard_<timestamp>.csv`: accessible scorecard table.
- `results/robustness_scorecard_<timestamp>.png`: visual comparison of robustness scores.
- `results/sample_adversarial_examples_<timestamp>.png`: clean and adversarial sample comparison when ART attacks run.
