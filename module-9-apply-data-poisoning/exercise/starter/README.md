# Starter: Traffic Sign Label-Flipping Poisoning Assessment

In this starter workflow, you will complete a poisoning assessment against a compact GTSRB traffic sign classifier.

Your main work is in:

- `notebooks/traffic_sign_label_flip_poisoning_assessment.ipynb`
- `src/traffic_sign_poisoning_utils.py`
- `docs/assessment_report_template.md`

The starter code provides dataset preparation, model construction, training, evaluation, and plotting helpers. You will complete the label-flipping implementation and the analysis cells.

## Run Order

1. Install dependencies from `../requirements.txt`.
2. Open the notebook in `notebooks/`.
3. Run the setup and clean baseline sections.
4. Complete the TODOs for label flipping and targeted metric calculation.
5. Retrain the poisoned model.
6. Complete the assessment report.

The first run downloads GTSRB into `data/gtsrb/`.

## Optional Instructor Prep

To pre-generate the compact dataset and clean baseline checkpoint before class, run:

```bash
bash scripts/prepare_traffic_sign_assets.sh
```

Or run the Python script directly:

```bash
python scripts/prepare_traffic_sign_assets.py
```
