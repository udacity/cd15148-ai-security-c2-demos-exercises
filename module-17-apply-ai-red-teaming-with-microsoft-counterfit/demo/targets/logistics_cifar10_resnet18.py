from pathlib import Path
import sys

import numpy as np
import torch

from counterfit.core.targets import CFTarget


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from counterfit_demo_utils import (  # noqa: E402
    CIFAR10_CLASSES,
    build_resnet18_cifar10,
    load_npz_dataset,
    prepare_cifar10_assets,
    train_or_load_model,
)


class LogisticsCifar10Resnet18(CFTarget):
    """Counterfit target for the demo CIFAR-10 ResNet-18 image classifier."""

    data_type = "image"
    target_name = "logistics_cifar10_resnet18"
    log_probs = False
    endpoint = "models/cifar10_resnet18_demo.pt"
    data_path = "data/generated/validation_500.npz"
    input_shape = (3, 32, 32)
    output_classes = CIFAR10_CLASSES
    classifier = "closed-box"
    X = []

    def load(self):
        data_dir = ROOT / "data" / "generated"
        download_dir = ROOT / "data" / "cifar10"
        model_path = ROOT / self.endpoint

        prepare_cifar10_assets(data_dir, download_dir, train_per_class=150, val_per_class=50)
        train_images, train_labels = load_npz_dataset(data_dir / "train_subset.npz")
        val_images, val_labels = load_npz_dataset(ROOT / self.data_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = train_or_load_model(model_path, train_images, train_labels, device=self.device, epochs=2)
        self.X = val_images.astype(np.float32)
        self.y = val_labels.astype(np.int64)

    def predict(self, x):
        self.model.eval()
        x_tensor = torch.tensor(np.array(x), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        return probabilities.tolist()
