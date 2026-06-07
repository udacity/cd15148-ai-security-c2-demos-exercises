# Instructor Notes: Model Inversion Demo

## Timing

| Segment | Minutes | Notes |
|---------|---------|-------|
| Scenario setup | 1 | Emphasize that the attacker does not have training data or source code access. |
| Dataset and model outputs | 3 | Show validation predictions and probability vectors. Ask which fields create attack signal. |
| Attack configuration | 3 | Explain the reconstruction objective: find an input that makes the model highly confident in a target identity. |
| Visual comparison | 4 | Compare class prototypes with reconstructed approximations. Focus on feature leakage, not exact photo recovery. |
| Mitigation discussion | 3 | Connect output restriction, query limits, overfitting control, and privacy-preserving training. |

## Teaching Emphasis

The reconstructed images are representative feature approximations, not guaranteed exact recovered photographs. That distinction matters: the privacy concern is that repeated model queries can expose subject-level characteristics that should remain private. The `recovered_faces_from_model_inversion.png` artifact adds a face-prior inversion panel so learners can compare a nearest training image, the training prototype, and a face-like reconstruction driven by model confidence.

The `rich_probability` attack has smoother optimization signal because it receives full probability feedback. The `rounded_probability` setting intentionally weakens that signal and should usually reduce prototype similarity.

## Suggested Prompts

- What does a confidence score reveal that a label-only response does not?
- Why does repeated querying matter more than a single prediction?
- Which production controls would reduce this attack surface for a 60,000-request-per-day facility access system?
- How would overfitting change the expected privacy leakage?
