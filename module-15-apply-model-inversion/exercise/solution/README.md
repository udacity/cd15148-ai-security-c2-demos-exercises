# Solution: Medical Model Inversion Privacy Assessment

This solution completes the model inversion privacy assessment workflow for a brain tumor MRI screening classifier using the Ultralytics brain tumor dataset. It trains or loads a compact PyTorch model, evaluates prediction confidence, runs three output configurations, writes quantitative leakage metrics, creates reconstruction visuals, and produces a short privacy assessment report.

## Run

The canonical run path is the notebook `notebooks/medical_model_inversion_assessment.ipynb`:

```bash
pip install -r requirements.txt
```

Then open and run the notebook end-to-end.

`scripts/run_assessment.py` is provided as a non-interactive reproducibility entrypoint for instructors (same workflow, no notebook UI):

```bash
python scripts/run_assessment.py
```

For a faster smoke test:

```bash
python scripts/run_assessment.py --epochs 2 --train-per-class 30 --val-per-class 15 --attack-steps 20 --attack-restarts 1
```

## Outputs

Committed reference outputs in `results/` use the `_baseline` suffix:

- `results/confidence_outputs_baseline.csv`
- `results/model_inversion_privacy_metrics_baseline.csv`
- `results/privacy_assessment_summary_baseline.json`
- `results/reconstructed_medical_features_baseline.png`
- `results/privacy_leakage_by_output_config_baseline.png`
- `results/privacy_assessment_report_baseline.md`

Re-running the notebook or script produces fresh sibling files with a
`_<YYYYMMDD_HHMMSS>` suffix (e.g. `confidence_outputs_20260603_182755.csv`).
These runtime artifacts are gitignored so the committed baselines stay
untouched on re-runs.
