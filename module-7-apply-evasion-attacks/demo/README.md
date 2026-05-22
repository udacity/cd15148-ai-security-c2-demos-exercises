# Demo: Build an Adversarial Stop Sign Attack Pipeline

Estimated time: 12 minutes

## Overview

In this demo, learners build an adversarial evasion attack workflow against a computer vision classifier. The walkthrough compares a black-box attack, Simple Black-box Adversarial Attacks (SimBA), with a white-box Projected Gradient Descent (PGD) attack.

The demo focuses on the practical tradeoff between attack strength and visual stealth. Students tune epsilon values and compare prediction changes, confidence drops, and perturbation magnitude.

## Scenario

An autonomous warehouse robotics team deploys a vision model to classify navigation markers and safety signs inside a fulfillment center processing approximately 20,000 package movements per hour. During a security review, researchers investigate whether small perturbations to camera inputs could cause the system to misclassify important visual indicators, potentially leading to navigation failures or unsafe operating conditions.

## Demo Materials

| File | Purpose |
|------|---------|
| `notebooks/adversarial_stop_sign_attack_pipeline.ipynb` | Main 12-minute notebook walkthrough |
| `src/vision_attack_utils.py` | Helper functions for GTSRB model loading, plotting, perturbation metrics, and stop-sign image selection |
| `docs/instructor_notes.md` | Timing, talking points, and troubleshooting notes |
| `docs/results_template.md` | Table for recording clean, SimBA, and PGD results |
| `docs/references.md` | Library, dataset, and asset references |
| `requirements.txt` | Python package requirements |
| `downloads/` | Local download target for the GTSRB traffic sign model |
| `scripts/download_gtsrb_model.sh` | Shell script that downloads the Hugging Face model files |
| `results/` | Attack comparison image and result table appear here when the notebook runs |

## Dataset and Model Notes

- The notebook uses the GTSRB traffic sign dataset through `torchvision.datasets.GTSRB`.
- GTSRB includes a real `Stop` class, class ID `14`.
- The notebook uses the MIT-licensed Hugging Face model `kelvinandreas/vit-traffic-sign-GTSRB`, a ViT model fine-tuned for 43-class GTSRB traffic sign classification.
- Downloaded model files are stored locally under `downloads/gtsrb_model/`.

## Setup

From this demo folder, install dependencies:

```bash
pip install -r requirements.txt
```

Then download the model files:

```bash
bash scripts/download_gtsrb_model.sh
```

If you cannot run the shell script, manually download these files from `https://huggingface.co/kelvinandreas/vit-traffic-sign-GTSRB/tree/main` and place them in `downloads/gtsrb_model/`:

- `config.json`
- `preprocessor_config.json`
- `model.safetensors`

To execute the notebook from the command line:

```bash
python -m nbconvert --to notebook --execute notebooks/adversarial_stop_sign_attack_pipeline.ipynb --output executed_adversarial_stop_sign_attack_pipeline.ipynb --output-dir results
```

The requirements include `ipykernel` and `nbconvert` for notebook execution. If you want a full browser-based notebook UI in a local environment, install your preferred Jupyter frontend separately.

## Steps

1. Load the pretrained image classification model.
2. Evaluate baseline model accuracy and confidence scores on clean GTSRB images.
3. Select a real GTSRB stop sign image.
4. Configure and execute a SimBA evasion attack using the Adversarial Robustness Toolbox.
5. Implement a PGD white-box attack using model gradients on the same image.
6. Compare confidence drop, perturbation magnitude, and predictions for clean, SimBA, and PGD images.
7. Discuss operational implications for deployed vision systems.

## Key Takeaway

Adversarial evasion attacks demonstrate that even accurate models can fail under carefully crafted inputs. Robustness testing must evaluate behavior under adversarial conditions, not just clean validation performance.
