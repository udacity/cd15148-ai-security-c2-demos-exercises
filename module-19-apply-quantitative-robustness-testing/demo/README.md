# Demo: Build a Robustness Evaluation Pipeline for an Image Classification Model

This demo evaluates CIFAR-10 ResNet-18 image classifiers under clean, environmental, and adversarial test conditions. It produces a comparative scorecard with accuracy, confidence degradation, perturbation tolerance, attack success rate, and an operational robustness score.

The first run downloads CIFAR-10, creates a balanced 1,000-image validation subset, and trains compact classroom checkpoints. Later runs reuse the checkpoints in `models/`.

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

- `results/robustness_scorecard.csv`: accessible scorecard table.
- `results/robustness_scorecard.png`: visual comparison of robustness scores.
- `results/sample_adversarial_examples.png`: clean and adversarial sample comparison when ART attacks run.

## Counterfit Extension

The `targets/` and `configs/counterfit_attack_plan.json` files extend the Module 17 Counterfit pattern. Microsoft Counterfit is optional here because it is typically installed in a supported Linux or WSL environment. The primary quantitative pipeline uses Adversarial Robustness Toolbox so the scorecard remains repeatable.
