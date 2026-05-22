# Robustness Assessment

## Selected Attack

This solution uses Projected Gradient Descent (PGD), a white-box evasion attack that uses model gradients to search for adversarial examples within an epsilon-bounded perturbation budget. PGD is relevant to the restricted-airspace monitoring scenario because it estimates how vulnerable the aerial object classifier could be if an evaluator or attacker has strong knowledge of the model internals.

## Quantitative Findings

The workflow evaluates three epsilon values: `0.01`, `0.03`, and `0.06`. As epsilon increases, perturbation magnitude increases and adversarial accuracy is expected to decrease. Attack success rate is measured only on examples that the model classified correctly before attack, which avoids overstating attack effectiveness.

Key metrics to review in `results/adversarial_metrics.csv`:

- Clean accuracy versus adversarial accuracy
- Attack success rate
- Mean clean confidence versus mean adversarial confidence
- Mean `Linf` and `L2` perturbation magnitude
- False negatives on true airplane images

## Operational Risk

The highest-risk configuration is the smallest epsilon that creates meaningful false negatives while remaining difficult to notice visually. In the security monitoring scenario, false negatives are more dangerous than false positives because they can suppress alerts for unauthorized aerial objects near restricted airspace.

## Recommended Next Steps

Recommended mitigations include adversarial training, confidence-threshold review, monitoring for sudden confidence degradation, human review of low-confidence frames, sensor fusion across multiple cameras, and periodic robustness testing during model updates.
