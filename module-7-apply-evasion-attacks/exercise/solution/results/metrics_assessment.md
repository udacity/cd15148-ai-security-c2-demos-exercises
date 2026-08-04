# Metrics Filled Out

| Attack | Epsilon | Clean Accuracy | Adversarial Accuracy | Attack Success Rate | Mean Clean Confidence | Mean Adversarial Confidence | Mean Linf | Mean L2 |
|--------|---------|----------------|----------------------|---------------------|-----------------------|-----------------------------|-----------|---------|
| pgd | 0.01 | 0.7833 | 0.7917 | 0.0532 | 0.7277 | 0.6908 | 0.01 | 0.522 | 
| pgd | 0.03 | 0.7833 | 0.6917 | 0.2872 | 0.7277 | 0.6567 | 0.03 | 1.5254 | 
| pgd | 0.06 | 0.7833 | 0.525 | 0.5532 | 0.7277 | 0.6781 | 0.06 | 3.0098

## Notes

- Attack success rate should be measured on examples that were correctly classified before attack.
- For this security scenario, false negatives are especially important because missed aerial object detections can suppress alerts.
