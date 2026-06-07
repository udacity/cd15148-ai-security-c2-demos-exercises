# Traffic Sign Robustness Assessment Report

## Summary

The highest-risk condition was `art_fgsm_eps_0.06` with accuracy `0.2333`, confidence drop `-0.0605`, and attack success rate `0.8182`. This condition should be reviewed before expanding the shuttle fleet to new operating environments.

## Operational Risk Analysis

Environmental testing covered 3 degraded camera conditions. Adversarial testing covered 3 perturbation strengths. Conditions marked `high` risk fall below the demo safety gate of 0.70 stressed-condition accuracy or exceed 0.35 attack success rate.

Confidence degradation should be treated as an early warning signal. A condition with acceptable accuracy but a large confidence drop can still create unstable downstream behavior if the shuttle planning system depends on confidence thresholds.

## Recommended Mitigations

- Add degraded-image validation gates for blur, noise, and compression before model release.
- Add confidence-threshold review for low-confidence traffic sign predictions.
- Retrain or fine-tune with targeted environmental augmentation for the highest-risk condition.
- Add periodic ART-based adversarial regression tests to detect robustness regressions after model updates.
- Route safety-critical low-confidence frames to redundant perception checks when latency budgets allow.
