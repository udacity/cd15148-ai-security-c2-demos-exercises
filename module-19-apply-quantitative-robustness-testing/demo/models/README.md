# Demo Models

The demo writes compact ResNet-18 CIFAR-10 checkpoints here:

- `standard_resnet18_cifar10.pt`
- `noise_augmented_resnet18_cifar10.pt`

These checkpoints ship with the repo and are loaded on every run. They are regenerated only if you set `RUN_FROM_CHECKPOINT = False` in the notebook, which overwrites these committed files.
