# Poisoned Training Dataset Assessment Report

## Configuration

- Source class: Stop
- Target class: Yield
- Poison fraction: 0.75 of source-class training samples
- Training subset size: 1,200 samples
- Validation subset size: 600 samples

## Quantitative Results

| Model | Clean Accuracy | Source-Class Accuracy | Targeted Misclassification Rate |
| --- | ---: | ---: | ---: |
| Clean baseline | 0.933 | 0.870 | 0.000 |
| Poisoned model | 0.808 | 0.000 | 0.990 |

## Findings

The label-flipping attack intentionally corrupted 75% of the Stop training labels by relabeling them as Yield. The poisoned model still retained a relatively high aggregate clean accuracy, but it completely lost Stop-class recognition on clean validation data and showed a 99.0% targeted Stop-to-Yield misclassification rate. This demonstrates that aggregate accuracy alone can conceal a serious class-specific failure in a safety-critical setting.

## Operational Risk

If a poisoned model learns to confuse Stop and Yield signs, an autonomous shuttle or traffic-monitoring system may fail to recognize stop signs correctly even when the input images are clean. This is a training pipeline integrity failure rather than an inference-time perturbation problem, and it can create immediate safety risk in transportation applications.

## Mitigations

Recommended mitigations include dataset provenance controls, label review for safety-critical classes, class-conditional validation, anomaly checks for label distribution shifts, protected training pipelines, and model comparison gates before deployment.
