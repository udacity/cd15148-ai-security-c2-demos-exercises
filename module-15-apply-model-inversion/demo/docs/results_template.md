# Results Template: Facial Model Inversion

## Baseline Output Review

| Metric | Value |
|--------|-------|
| Validation accuracy | |
| Mean confidence | |
| P95 confidence | |

## Inversion Findings

| Attack setting | Target identity | Queries | Target confidence | Prototype similarity |
|----------------|-----------------|---------|-------------------|----------------------|
| Rich probability | | | | |
| Rounded probability | | | | |

## Privacy Interpretation

What did the reconstructed feature approximation reveal about the target identity?

How similar is the face-prior reconstruction to the nearest training image? Does it look like an exact sample recovery or a representative reconstruction?

## Mitigation Plan

Which controls should be prioritized?

- Output restriction:
- Confidence rounding:
- Query rate limiting:
- Overfitting reduction:
- Differential privacy or privacy-preserving training:
