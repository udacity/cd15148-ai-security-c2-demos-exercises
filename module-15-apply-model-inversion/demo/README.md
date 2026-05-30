# Demo: Reconstruct Sensitive Features from a Facial Recognition Model

Estimated time: 14 minutes

## Overview

This demo walks through a model inversion attack workflow against a facial recognition classifier. Learners train or load a small PyTorch face classifier on the AT&T Laboratories Cambridge face dataset, inspect rich confidence score outputs, and then reconstruct representative facial feature approximations using repeated inference-style optimization.

The script first looks for the original AT&T/ORL folder layout at `data/att_faces/s1/1.pgm` through `data/att_faces/s40/10.pgm`. If that folder is not present, it uses `sklearn.datasets.fetch_olivetti_faces`, which provides the same AT&T/Olivetti face dataset and caches it under `data/generated/sklearn_cache`.

## Scenario

A security technology company deploys a facial recognition system for controlled facility access across multiple office locations, processing approximately 60,000 authentication requests per day. During an internal privacy assessment, researchers test whether attackers can reconstruct sensitive facial characteristics from model outputs alone.

## Demo Materials

| File | Purpose |
|------|---------|
| `scripts/run_model_inversion_demo.py` | End-to-end runnable demo |
| `src/model_inversion_demo_utils.py` | AT&T/Olivetti dataset loading, CNN training, inversion attack, metrics, and plots |
| `notebooks/model_inversion_facility_access_demo.ipynb` | Jupyter walkthrough for instruction |
| `docs/instructor_notes.md` | Timing and discussion guidance |
| `docs/results_template.md` | Learner-facing result prompts |
| `requirements.txt` | Python dependencies |
| `data/` | Optional local AT&T folder and generated scikit-learn cache |
| `models/` | Generated facial recognition checkpoint |
| `results/` | Generated confidence outputs, metrics, and visualizations |

## Run the Demo

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full script:

```bash
python scripts/run_model_inversion_demo.py
```

The default run trains a compact classroom model for 50 epochs on 280 AT&T/Olivetti training images, then evaluates 500 labeled validation queries produced from the held-out AT&T/Olivetti split.

For a faster smoke test:

```bash
python scripts/run_model_inversion_demo.py --train-per-identity 6 --val-per-identity 4 --validation-queries 80 --epochs 2 --attack-steps 20 --attack-restarts 1 --targets 2
```

Execute the notebook:

```bash
python -m nbconvert --to notebook --execute notebooks/model_inversion_facility_access_demo.ipynb --output executed_model_inversion_facility_access_demo.ipynb --output-dir results
```

## Outputs

- `results/sample_confidence_outputs.csv`: sample labels, predictions, confidence scores, and top probability vectors.
- `results/model_inversion_metrics.csv`: reconstruction confidence, query count, and prototype similarity metrics.
- `results/model_inversion_summary.json`: scenario summary and mitigation notes.
- `results/reconstructed_feature_approximations.png`: representative class prototypes compared with reconstructed approximations.
- `results/recovered_faces_from_model_inversion.png`: nearest training image, class prototype, and face-prior model inversion recovery examples.
- `results/confidence_leakage_comparison.png`: leakage comparison for rich versus rounded probability outputs.

## Key Takeaway

Model inversion attacks demonstrate that trained models can unintentionally reveal sensitive information about training data. The reconstruction is usually a representative face-like approximation rather than a guaranteed exact copy of one training image, but overfit models and detailed probability outputs can make the approximation much more revealing. Privacy protection in AI systems must include thoughtful control of model outputs, rate limits on inference access, overfitting checks, and privacy-aware training practices.
