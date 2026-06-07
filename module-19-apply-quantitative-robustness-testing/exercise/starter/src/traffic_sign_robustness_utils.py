from __future__ import annotations

import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


GTSRB_CLASSES = [
    "Speed limit 20 km/h",
    "Speed limit 30 km/h",
    "Speed limit 50 km/h",
    "Speed limit 60 km/h",
    "Speed limit 70 km/h",
    "Speed limit 80 km/h",
    "End of speed limit 80 km/h",
    "Speed limit 100 km/h",
    "Speed limit 120 km/h",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice or snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]

SELECTED_ORIGINAL_CLASS_IDS = [1, 12, 13, 14, 17, 18]
SELECTED_CLASS_NAMES = [GTSRB_CLASSES[class_id] for class_id in SELECTED_ORIGINAL_CLASS_IDS]
ORIGINAL_TO_COMPACT = {original: compact for compact, original in enumerate(SELECTED_ORIGINAL_CLASS_IDS)}


class NumpyImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.images[index]), torch.tensor(self.labels[index])


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes: int = len(SELECTED_CLASS_NAMES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_traffic_sign_cnn() -> nn.Module:
    return TrafficSignCNN()


def prepare_gtsrb_subsets(
    output_dir: Path | str,
    download_root: Path | str,
    train_per_class: int = 80,
    val_per_class: int = 20,
    image_size: int = 64,
    seed: int = 19,
) -> Path:
    output_dir = Path(output_dir)
    download_root = Path(download_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "traffic_sign_train.npz"
    val_path = output_dir / "traffic_sign_val.npz"
    if train_path.exists() and val_path.exists():
        train_count = len(np.load(train_path)["labels"])
        val_count = len(np.load(val_path)["labels"])
        expected_train = train_per_class * len(SELECTED_CLASS_NAMES)
        expected_val = val_per_class * len(SELECTED_CLASS_NAMES)
        if train_count >= expected_train and val_count >= expected_val:
            return output_dir

    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    train_source = datasets.GTSRB(root=str(download_root), split="train", download=True, transform=transform)
    val_source = datasets.GTSRB(root=str(download_root), split="test", download=True, transform=transform)
    rng = np.random.default_rng(seed)

    def collect(source: Dataset, per_class: int) -> tuple[np.ndarray, np.ndarray]:
        by_class = {compact: [] for compact in range(len(SELECTED_CLASS_NAMES))}
        for image, original_label in source:
            original_label = int(original_label)
            if original_label in ORIGINAL_TO_COMPACT:
                compact_label = ORIGINAL_TO_COMPACT[original_label]
                if len(by_class[compact_label]) < per_class:
                    by_class[compact_label].append(image.numpy())
            if all(len(class_images) >= per_class for class_images in by_class.values()):
                break

        images = []
        labels = []
        for compact_label, class_images in by_class.items():
            if len(class_images) < per_class:
                raise ValueError(
                    f"Class {SELECTED_CLASS_NAMES[compact_label]} has {len(class_images)} samples; "
                    f"requested {per_class}."
                )
            for idx in range(per_class):
                images.append(class_images[idx])
                labels.append(compact_label)
        order = rng.permutation(len(labels))
        return np.stack(images)[order], np.array(labels, dtype=np.int64)[order]

    train_images, train_labels = collect(train_source, train_per_class)
    val_images, val_labels = collect(val_source, val_per_class)
    np.savez_compressed(train_path, images=train_images, labels=train_labels)
    np.savez_compressed(val_path, images=val_images, labels=val_labels)
    return output_dir


def load_subset(data_dir: Path | str, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(data_dir) / f"traffic_sign_{split}.npz"
    data = np.load(path)
    return data["images"].astype(np.float32), data["labels"].astype(np.int64)


def train_or_load_model(
    checkpoint_path: Path | str,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    device: str = "cpu",
    epochs: int = 4,
    batch_size: int = 64,
) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    model = build_traffic_sign_cnn().to(device)
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model

    loader = DataLoader(NumpyImageDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        print(f"epoch {epoch + 1}/{epochs} loss={running_loss / max(len(loader), 1):.4f}")

    model.eval()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return model


def predict(model: nn.Module, images: np.ndarray, device: str = "cpu", batch_size: int = 128) -> dict[str, np.ndarray]:
    model.eval()
    predictions = []
    confidences = []
    probabilities = []
    loader = DataLoader(NumpyImageDataset(images, np.zeros(len(images), dtype=np.int64)), batch_size=batch_size)
    with torch.no_grad():
        for batch, _ in loader:
            logits = model(batch.to(device))
            probs = torch.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            predictions.append(pred.cpu().numpy())
            confidences.append(confidence.cpu().numpy())
            probabilities.append(probs.cpu().numpy())

    return {
        "predictions": np.concatenate(predictions),
        "confidence": np.concatenate(confidences),
        "probabilities": np.concatenate(probabilities),
    }


def calculate_condition_metrics(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    clean_predictions: np.ndarray | None = None,
    clean_confidence: np.ndarray | None = None,
    device: str = "cpu",
) -> dict[str, float | np.ndarray]:
    # TODO: Predict labels and confidence scores for this condition.
    # TODO: Calculate accuracy with sklearn.metrics.accuracy_score.
    # TODO: Calculate mean confidence.
    # TODO: If clean_confidence is provided, calculate mean confidence degradation.
    # TODO: If clean_predictions is provided, calculate attack success rate on examples
    #       that were originally classified correctly.
    raise NotImplementedError("Complete calculate_condition_metrics.")


def gaussian_noise(images: np.ndarray, sigma: float = 0.08, seed: int = 19) -> np.ndarray:
    # TODO: Add Gaussian noise and clip pixel values to [0, 1].
    raise NotImplementedError("Complete gaussian_noise.")


def gaussian_blur(images: np.ndarray, radius: float = 1.0) -> np.ndarray:
    return np.stack([_pil_to_chw(_chw_to_pil(image).filter(ImageFilter.GaussianBlur(radius))) for image in images])


def jpeg_compression(images: np.ndarray, quality: int = 35) -> np.ndarray:
    compressed = []
    for image in images:
        buffer = io.BytesIO()
        _chw_to_pil(image).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed.append(_pil_to_chw(Image.open(buffer).convert("RGB")))
    return np.stack(compressed).astype(np.float32)


def _chw_to_pil(image: np.ndarray) -> Image.Image:
    array = np.transpose(np.clip(image, 0.0, 1.0), (1, 2, 0))
    return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")


def _pil_to_chw(image: Image.Image) -> np.ndarray:
    array = np.asarray(image).astype(np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def environmental_test_sets(images: np.ndarray) -> dict[str, np.ndarray]:
    # TODO: Return at least three environmental test sets:
    #       Gaussian noise, Gaussian blur, and JPEG compression.
    raise NotImplementedError("Complete environmental_test_sets.")


def build_art_classifier(model: nn.Module):
    from art.estimators.classification import PyTorchClassifier

    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        input_shape=(3, 64, 64),
        nb_classes=len(SELECTED_CLASS_NAMES),
        clip_values=(0.0, 1.0),
    )


def run_fgsm_attacks(
    model: nn.Module,
    images: np.ndarray,
    epsilons: list[float],
) -> dict[str, np.ndarray]:
    # TODO: Use ART FastGradientMethod to generate one adversarial dataset per epsilon.
    #       Return a dictionary keyed by names such as "art_fgsm_eps_0.02".
    raise NotImplementedError("Complete run_fgsm_attacks.")


def perturbation_linf(clean: np.ndarray, candidate: np.ndarray) -> float:
    # TODO: Calculate the average per-image L-infinity perturbation magnitude.
    raise NotImplementedError("Complete perturbation_linf.")


def row_from_metrics(
    condition: str,
    condition_type: str,
    metrics: dict[str, float | np.ndarray],
    perturbation: float,
) -> dict[str, float | str]:
    accuracy = float(metrics["accuracy"])
    confidence_drop = float(metrics["confidence_drop"])
    attack_success_rate = float(metrics["attack_success_rate"])
    return {
        "condition": condition,
        "condition_type": condition_type,
        "accuracy": round(accuracy, 4),
        "mean_confidence": round(float(metrics["mean_confidence"]), 4),
        "confidence_drop": round(confidence_drop, 4),
        "attack_success_rate": round(attack_success_rate, 4),
        "perturbation_linf": round(perturbation, 4),
        "risk_level": classify_risk(accuracy, confidence_drop, attack_success_rate),
    }


def classify_risk(accuracy: float, confidence_drop: float, attack_success_rate: float) -> str:
    if accuracy < 0.70 or attack_success_rate > 0.35:
        return "high"
    if accuracy < 0.82 or confidence_drop > 0.15 or attack_success_rate > 0.20:
        return "medium"
    return "low"


def write_scorecard(rows: list[dict[str, float | str]], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "condition_type",
        "accuracy",
        "mean_confidence",
        "confidence_drop",
        "attack_success_rate",
        "perturbation_linf",
        "risk_level",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def plot_metric_bars(rows: list[dict[str, float | str]], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["condition"]) for row in rows]
    accuracy = [float(row["accuracy"]) for row in rows]
    confidence_drop = [float(row["confidence_drop"]) for row in rows]
    attack_success = [float(row["attack_success_rate"]) for row in rows]

    x = np.arange(len(rows))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.9), 5))
    ax.bar(x - width, accuracy, width, label="Accuracy")
    ax.bar(x, confidence_drop, width, label="Confidence drop")
    ax.bar(x + width, attack_success, width, label="Attack success rate")
    ax.axhline(0.70, color="#6f3a00", linestyle="--", linewidth=1.2, label="Minimum accuracy threshold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_clean_vs_degraded(
    clean_images: np.ndarray,
    degraded_images: np.ndarray,
    output_path: Path | str,
    title: str,
    max_items: int = 6,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(max_items, len(clean_images))
    fig, axes = plt.subplots(2, count, figsize=(2.1 * count, 4.3))
    for idx in range(count):
        axes[0, idx].imshow(np.transpose(np.clip(clean_images[idx], 0, 1), (1, 2, 0)))
        axes[0, idx].set_title("Clean")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(np.transpose(np.clip(degraded_images[idx], 0, 1), (1, 2, 0)))
        axes[1, idx].set_title("Degraded")
        axes[1, idx].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_assessment_report(rows: list[dict[str, float | str]], output_path: Path | str) -> Path:
    # TODO: Write a short Markdown report that identifies the highest-risk condition,
    #       explains operational impact, and recommends mitigations.
    raise NotImplementedError("Complete write_assessment_report.")
