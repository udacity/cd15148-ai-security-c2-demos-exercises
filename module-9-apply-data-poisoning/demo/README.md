# Demo: Build a Poisoned Image Classification Training Pipeline

Estimated time: 13 minutes

## Overview

This demo shows a backdoor-style data poisoning workflow against an image classification model. Learners modify a small portion of the CIFAR-10 training subset with a visible trigger pattern, relabel those images to a target class, retrain a ResNet-18 classifier, and compare normal validation accuracy with targeted backdoor behavior.

The core lesson is that data poisoning compromises the training process itself. A poisoned model may still perform normally on clean validation data while behaving incorrectly when a trigger appears at inference time.

## Scenario

A manufacturing company trains a computer vision model to classify component categories on an automated assembly line processing approximately 100,000 parts per day. During a security assessment, researchers investigate whether an attacker with limited access to the training pipeline could manipulate training data to reduce reliability or create hidden backdoor behavior that bypasses quality assurance checks.

CIFAR-10 classes are used as lightweight stand-ins for manufacturing component categories.

## Demo Materials

| File | Purpose |
|------|---------|
| `notebooks/poisoned_image_classification_training_pipeline.ipynb` | Main demo notebook |
| `src/poisoning_utils.py` | CIFAR-10 subset prep, ResNet-18 model, poisoning helpers, metrics, and plots |
| `docs/instructor_notes.md` | Timing and teaching notes |
| `docs/results_template.md` | Results table and discussion prompts |
| `docs/references.md` | Dataset, library, and runtime notes |
| `requirements.txt` | Python dependencies |
| `data/` | Generated CIFAR-10 subsets appear here |
| `models/` | Generated baseline and poisoned checkpoints appear here |
| `results/` | Generated plots and metrics appear here |

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

The notebook trains ResNet-18 on a 5,000-image subset for classroom runtime. The clean accuracy is not meant to represent a production CIFAR-10 benchmark; the important comparison is clean validation behavior versus triggered backdoor behavior.

## Key Takeaway

Data poisoning attacks demonstrate that AI systems can be compromised long before deployment. Protecting training pipelines and validating training data integrity are critical components of AI security.
