from pathlib import Path
import sys

import numpy as np
import torch

from counterfit.core.targets import CFTarget


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from drone_counterfit_utils import (  # noqa: E402
    DRONE_CLASSES,
    load_npz_dataset,
    prepare_drone_detection_assets,
    train_or_load_model,
)


class DroneDetectionTarget(CFTarget):
    """Counterfit target scaffold for the binary drone detection model."""

    data_type = "image"
    target_name = "drone_detection_resnet18"
    log_probs = False
    endpoint = "models/drone_detection_resnet18.pt"
    data_path = "data/generated/validation_subset.npz"
    input_shape = (3, 32, 32)
    output_classes = DRONE_CLASSES
    classifier = "closed-box"
    X = []

    def load(self):
        data_dir = ROOT / "data" / "generated"
        download_dir = ROOT / "data" / "cifar10"
        model_path = ROOT / self.endpoint

        prepare_drone_detection_assets(data_dir, download_dir)
        train_images, train_labels = load_npz_dataset(data_dir / "train_subset.npz")
        val_images, val_labels = load_npz_dataset(ROOT / self.data_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = train_or_load_model(model_path, train_images, train_labels, device=self.device)
        self.X = val_images.astype(np.float32)
        self.y = val_labels.astype(np.int64)

    def predict(self, x):
        # TODO: Convert the Counterfit input batch to a tensor, run the model,
        # apply softmax, and return probabilities as a Python list.
        raise NotImplementedError("Implement the Counterfit target predict method.")
