# Exercise: Build a Quantitative Robustness Assessment for a Traffic Sign Recognition System

## Scenario

A transportation technology company is preparing to expand an autonomous shuttle traffic sign recognition system to additional campus routes. Safety reviewers need measurable evidence about how the model behaves under degraded camera conditions and adversarial perturbations.

## Your Task

Complete the starter notebook and utility functions to produce a quantitative robustness assessment. Your assessment must compare clean validation performance against environmental distortions and adversarial attacks, then summarize the operational risk.

## Requirements

- Load the provided traffic sign recognition model and validation dataset.
- Measure clean accuracy and mean prediction confidence.
- Evaluate at least 3 environmental perturbation types: blur, noise, and compression.
- Run at least 3 adversarial perturbation strengths with ART.
- Calculate accuracy, confidence degradation, perturbation magnitude, and attack success rate for each condition.
- Generate visual comparisons of clean and degraded examples.
- Generate a comparison chart and a CSV scorecard.
- Write a short Markdown assessment report with operational risks and recommended mitigations.

## Files to Complete

- `notebooks/traffic_sign_robustness_assessment.ipynb`
- `src/traffic_sign_robustness_utils.py`

The TODO comments identify the expected implementation points. The notebook and script both use the same utility functions.

## Run

```bash
pip install -r requirements.txt
python scripts/run_assessment.py
```

For a faster debug run after completing non-ART functions:

```bash
python scripts/run_assessment.py --train-per-class 10 --val-per-class 5 --epochs 1 --max-eval 20 --skip-art
```

## Deliverables

- Completed notebook.
- `results/traffic_sign_robustness_scorecard.csv`
- `results/traffic_sign_metric_comparison.png`
- Clean-versus-degraded visualizations in `results/`
- `results/traffic_sign_assessment_report.md`
