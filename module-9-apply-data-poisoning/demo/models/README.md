# Model Checkpoints

| File | Source | Tracked in git? |
| --- | --- | --- |
| `baseline_resnet18.pt` | Shipped reference. Trained offline by `scripts/train_clean_baseline.py` with the demo's recipe (ResNet-18 from scratch, 2,000 images per class, 15 epochs, light augmentation). Loaded by the notebook by default. | Yes |
| `backdoor_resnet18.pt` | Trained live in the notebook on the backdoor-poisoned dataset. | No |
| `label_flipped_resnet18.pt` | Trained live in the notebook on the label-flipped dataset. | No |

To regenerate the shipped baseline checkpoint:

```bash
python scripts/train_clean_baseline.py
```