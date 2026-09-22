# Starter: Traffic Sign Robustness Assessment

Complete the notebook and TODOs in `src/traffic_sign_robustness_utils.py` to evaluate a traffic sign recognition model under environmental and adversarial stress.

The first run downloads GTSRB and creates a compact six-class traffic sign subset. A trained classroom checkpoint ships in `models/` and is loaded on every run, so no run trains by default. Set `RUN_FROM_CHECKPOINT = False` in the notebook to retrain it from scratch. Retraining overwrites the checkpoint committed to git.
