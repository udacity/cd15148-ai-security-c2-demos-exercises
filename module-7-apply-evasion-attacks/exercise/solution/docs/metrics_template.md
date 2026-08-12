# Metrics Template

| Attack | Epsilon | Clean Accuracy | Adversarial Accuracy | Attack Success Rate | Mean Clean Confidence | Mean Adversarial Confidence | Mean Linf | Mean L2 |
|--------|---------|----------------|----------------------|---------------------|-----------------------|-----------------------------|-----------|---------|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## Comparison Grid

Inline the comparison grid your notebook saved for the highest epsilon you evaluated. The
notebook writes it to `results/` as `<attack>_epsilon_<epsilon>_comparison.png`, so link to it
relative to this file and replace the filename with your own:

![TODO clean versus adversarial comparison grid](../results/TODO_epsilon_TODO_comparison.png)

## Notes

- Attack success rate should be measured on examples that were correctly classified before attack.
- For this security scenario, false negatives are especially important because missed aerial object detections can suppress alerts.
