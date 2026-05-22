# Instructor Notes

## Timing

| Segment | Time | Talking Point |
|---------|------|---------------|
| Scenario and threat model | 1 min | Evasion attacks happen at inference time and do not require changing model weights. |
| Load model and clean images | 2 min | Clean accuracy is a baseline, not a robustness guarantee. |
| Select GTSRB stop sign image | 1 min | GTSRB includes a real `Stop` traffic sign class. |
| Run SimBA | 3 min | Black-box attacks use model queries rather than gradients. |
| Run PGD | 3 min | White-box attacks use gradients and are often stronger under the same perturbation budget. |
| Compare results | 2 min | Epsilon controls the tradeoff between stealth and attack success. |

## Suggested Narration

- "The attacker is not retraining the model or poisoning the dataset. They are manipulating the input at inference time."
- "A visually small perturbation can be operationally large if it crosses a model decision boundary."
- "SimBA is useful when the attacker can query the model but cannot inspect gradients."
- "PGD is useful for robustness testing when defenders have full model access."
- "Visual inspection alone is not enough. We need prediction changes, confidence deltas, and perturbation metrics."

## Content and Asset Notes

The notebook uses GTSRB through TorchVision and selects a sample from class `14`, `Stop`. The model files are downloaded from the MIT-licensed Hugging Face repository `kelvinandreas/vit-traffic-sign-GTSRB`.

## Troubleshooting

If `art` is not installed, install `adversarial-robustness-toolbox`.

If `downloads/gtsrb_model/model.safetensors` is missing, run `bash scripts/download_gtsrb_model.sh` from the demo root.

If GTSRB cannot download in the classroom environment, predownload the dataset with TorchVision or run the notebook once in a network-enabled environment and package the resulting `data/gtsrb/` folder for classroom use.
