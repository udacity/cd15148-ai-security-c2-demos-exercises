# References

- [University of Toronto: CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch documentation: TorchVision ResNet models](https://pytorch.org/vision/stable/models/resnet.html)

## Classroom Runtime Note

The notebook trains ResNet-18 from scratch on a 20,000-image CIFAR-10 subset (2,000 images per class) for 15 epochs with light augmentation, reaching roughly 83% clean validation accuracy. A pretrained clean baseline checkpoint is shipped in `models/baseline_resnet18.pt` so the notebook loads it by default rather than retraining; the backdoor-poisoned and label-flipped models are trained live so learners see the training step. The teaching objective is the side-by-side comparison of clean, backdoored, and label-flipped behavior — not a production-quality CIFAR-10 number.
