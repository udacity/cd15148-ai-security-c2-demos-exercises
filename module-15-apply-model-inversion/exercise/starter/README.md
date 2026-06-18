# Starter: Medical Model Inversion Privacy Assessment

You are assessing a brain tumor MRI screening model before broader external API access. Your task is to measure whether detailed prediction outputs can leak sensitive characteristics through repeated inference queries.

## Tasks

1. Load the provided brain tumor MRI classifier and evaluation dataset.
2. Measure baseline confidence scores and output distributions.
3. Implement a model inversion attack using repeated queries and prediction probabilities.
4. Evaluate at least three output configurations:
   - full probability vector
   - rounded confidence scores
   - top-1 label only
5. Generate reconstructed feature approximations.
6. Compare reconstructions against representative validation samples.
7. Write a short privacy assessment report with risks and mitigations.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Complete the TODOs in `src/medical_inversion_utils.py`, then open and run `notebooks/medical_model_inversion_assessment.ipynb` end-to-end. The notebook is the canonical deliverable for this exercise.

## Expected Outputs

A successful run produces these files in `results/` with a `_<YYYYMMDD_HHMMSS>`
suffix derived from the notebook's run timestamp (so re-runs accumulate
side by side rather than overwriting each other):

- `results/confidence_outputs_<timestamp>.csv`
- `results/model_inversion_privacy_metrics_<timestamp>.csv`
- `results/privacy_assessment_summary_<timestamp>.json`
- `results/reconstructed_medical_features_<timestamp>.png`
- `results/privacy_leakage_by_output_config_<timestamp>.png`
- `results/privacy_assessment_report_<timestamp>.md`

Compare your outputs against the reference results in `../solution/results/`,
which are committed with a `_baseline` suffix.
