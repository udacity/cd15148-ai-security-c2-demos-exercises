# Instructor Notes

This demo extends the Module 17 Counterfit workflow into a quantitative robustness evaluation pipeline. Counterfit remains available as an optional target interface, while ART provides the repeatable adversarial attack suite used in the scorecard.

## Suggested Flow

1. Load the CIFAR-10 validation subset and compact ResNet-18 checkpoints.
2. Show that clean accuracy and confidence are only the baseline.
3. Run blur, compression, and Gaussian noise evaluations.
4. Run ART FGSM and PGD attacks at configurable perturbation strengths.
5. Compare the two model variants using the scorecard.
6. Tie the pass/fail interpretation to operational thresholds instead of benchmark accuracy alone.

## Talking Points

- Similar clean accuracy can hide different stressed-condition behavior.
- Confidence degradation can be an early warning before full misclassification.
- Attack success rate should be computed against examples the model initially classified correctly.
- Environmental distortions and adversarial perturbations are different stressors and should be reported separately.
- Operational release gates should be explicit, measurable, and tied to deployment requirements.
