# Demo: Build a Poisoned Image Classification Training Pipeline

## Overview

This demo walks through two data poisoning attacks against a CIFAR-10 image classifier: a **backdoor attack** (visual trigger + relabel) and a **label-flipping attack** (relabel only, no trigger). Learners train a clean ResNet-18 baseline, then retrain on each poisoned variant, then compare the three models side by side on aggregate accuracy, targeted attack behavior, and per-class accuracy.

The core lesson is that data poisoning compromises the training process itself, and that different attacks hide from different defender-side checks. A backdoored model passes aggregate validation but fails on triggered inputs. A label-flipped model passes aggregate validation but fails on per-class accuracy. Neither attack is caught by standard top-line metrics alone.

## Scenario

A manufacturing company trains a computer vision model to classify component categories on an automated assembly line processing approximately 100,000 parts per day. During a security assessment, researchers investigate whether an attacker with limited access to the training pipeline could manipulate training data to reduce reliability or create hidden backdoor behavior that bypasses quality assurance checks.

CIFAR-10 classes are used as lightweight stand-ins for manufacturing component categories.

## Demo Materials

| File | Purpose |
|------|---------|
| `notebooks/poisoned_image_classification_training_pipeline.ipynb` | Main demo notebook (backdoor + label-flip + three-way comparison) |
| `src/poisoning_utils.py` | CIFAR-10 subset prep, ResNet-18 model, poisoning helpers, metrics, and plots |
| `scripts/train_clean_baseline.py` | Regenerates the shipped clean baseline checkpoint |
| `models/baseline_resnet18.pt` | Shipped clean baseline checkpoint (loaded by the notebook) |
| `docs/instructor_notes.md` | Timing and teaching notes |
| `docs/results_template.md` | Results table and discussion prompts |
| `docs/references.md` | Dataset and runtime notes |
| `requirements.txt` | Python dependencies |
| `data/` | Downloaded CIFAR-10 and generated subsets appear here |
| `models/` | Live-trained backdoor and label-flipped checkpoints appear here |
| `results/` | Generated plots and metrics appear here. Notebook runs write timestamped files (e.g., `poisoning_metrics_YYYYMMDD_HHMMSS.csv`); committed reference copies use a `_baseline` suffix and are never overwritten. |

## Training Recipe

The demo trains ResNet-18 **from scratch** (no pretrained ImageNet weights) on a 20,000-image CIFAR-10 subset (2,000 images per class) for 15 epochs with light data augmentation (random crop + horizontal flip). The clean baseline reaches roughly 83% validation accuracy under this recipe — credibly "model passes basic validation" rather than a state-of-the-art CIFAR-10 result.

The shipped `models/baseline_resnet18.pt` checkpoint was produced by `scripts/train_clean_baseline.py` using this exact recipe. The notebook loads it by default. Set `RETRAIN_FROM_SCRATCH = True` in the notebook to retrain the baseline yourself. The backdoor-poisoned and label-flipped models are always trained live in the notebook so learners observe the training step.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Execute the notebook from this folder:

```bash
python -m nbconvert --to notebook --execute notebooks/poisoned_image_classification_training_pipeline.ipynb --output executed_poisoned_image_classification_training_pipeline.ipynb --output-dir results
```

The notebook downloads CIFAR-10 through TorchVision if the dataset is not already present.

> **Pre-cached in the classroom workspace.** When `C2_ASSET_CACHE` is set, this download is
> read from that shared cache and nothing is fetched at run time. Unset — a plain `git clone` —
> everything downloads into this module's own folders exactly as described above.

To regenerate the clean baseline checkpoint:

```bash
python scripts/train_clean_baseline.py
```

## Key Takeaway

Data poisoning attacks demonstrate that AI systems can be compromised long before deployment. Different attacks evade different defender-side checks: aggregate accuracy catches neither backdoors nor label flips, per-class accuracy catches label flips but not backdoors, and trigger-aware validation is needed to catch backdoors. Upstream protection — training data provenance, label-pipeline integrity, dataset hashing, controlled annotation access — is the only durable defense.