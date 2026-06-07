from pathlib import Path
import sys

import numpy as np
import torch

from counterfit.core.targets import CFTarget


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from robustness_eval_utils import (  # noqa: E402
    CIFAR10_CLASSES,
    ModelSpec,
    load_npz_dataset,
    prepare_cifar10_assets,
    predict,
    train_or_load_model,
)


class LogisticsCifar10Resnet18(CFTarget):
    """Counterfit target for the Module 19 CIFAR-10 ResNet-18 classifier."""

    data_type = "image"
    target_name = "module19_logistics_cifar10_resnet18"
    log_probs = False
    endpoint = "models/standard_resnet18_cifar10.pt"
    data_path = "data/generated/validation_1000.npz"
    input_shape = (3, 32, 32)
    output_classes = CIFAR10_CLASSES
    classifier = "closed-box"
    X = []

    def load(self):
        data_dir = ROOT / "data" / "generated"
        download_dir = ROOT / "data" / "cifar10"
        model_path = ROOT / self.endpoint

        prepare_cifar10_assets(data_dir, download_dir, train_per_class=150, val_per_class=100)
        train_images, train_labels = load_npz_dataset(data_dir / "train_subset.npz")
        val_images, val_labels = load_npz_dataset(ROOT / self.data_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = train_or_load_model(
            model_path,
            train_images,
            train_labels,
            spec=ModelSpec("standard_resnet18", "standard_resnet18_cifar10.pt"),
            device=self.device,
            epochs=2,
        )
        self.X = val_images.astype(np.float32)
        self.y = val_labels.astype(np.int64)

    def predict(self, x):
        outputs = predict(self.model, np.array(x, dtype=np.float32), device=self.device)
        return outputs["probabilities"].tolist()
