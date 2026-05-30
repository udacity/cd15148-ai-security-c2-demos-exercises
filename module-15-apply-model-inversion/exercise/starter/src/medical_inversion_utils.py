from __future__ import annotations

import csv
import json
import math
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = ["negative", "positive"]
BRAIN_TUMOR_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip"


@dataclass(frozen=True)
class MedicalDataset:
    train_images: np.ndarray
    train_labels: np.ndarray
    val_images: np.ndarray
    val_labels: np.ndarray
    class_names: list[str]
    dataset_name: str


@dataclass(frozen=True)
class OutputConfig:
    name: str
    description: str
    objective: str
    precision: int | None = None
    confidence_clip: float | None = None


class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(96 * 8 * 8, 160),
            nn.ReLU(),
            nn.Linear(160, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_medical_dataset(
    data_dir: Path,
    *,
    train_per_class: int = 300,
    val_per_class: int = 100,
    image_size: int = 128,
    seed: int = 42,
) -> MedicalDataset:
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / f"ultralytics_brain_tumor_{train_per_class}_{val_per_class}_{image_size}.npz"
    if dataset_path.exists():
        payload = np.load(dataset_path, allow_pickle=True)
        return MedicalDataset(
            payload["train_images"],
            payload["train_labels"],
            payload["val_images"],
            payload["val_labels"],
            list(payload["class_names"]),
            str(payload["dataset_name"]),
        )

    dataset_root = _download_or_find_brain_tumor_dataset(data_dir)
    train_images, train_labels = _load_brain_tumor_split(
        dataset_root,
        "train",
        image_size=image_size,
        per_class_limit=train_per_class,
        seed=seed,
    )
    val_images, val_labels = _load_brain_tumor_split(
        dataset_root,
        "val",
        image_size=image_size,
        per_class_limit=val_per_class,
        seed=seed + 1,
    )

    dataset = MedicalDataset(
        train_images,
        train_labels,
        val_images,
        val_labels,
        CLASS_NAMES,
        "Ultralytics brain tumor MRI detection subset",
    )
    np.savez_compressed(
        dataset_path,
        train_images=dataset.train_images,
        train_labels=dataset.train_labels,
        val_images=dataset.val_images,
        val_labels=dataset.val_labels,
        class_names=np.asarray(dataset.class_names),
        dataset_name=dataset.dataset_name,
    )
    return dataset


def _download_or_find_brain_tumor_dataset(data_dir: Path) -> Path:
    root = data_dir / "brain-tumor"
    if (root / "images" / "train").exists():
        return root
    zip_path = data_dir / "brain-tumor.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(BRAIN_TUMOR_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)
    if not (root / "images" / "train").exists():
        candidates = [p for p in data_dir.rglob("images") if (p / "train").exists()]
        if candidates:
            return candidates[0].parent
        raise FileNotFoundError("Could not locate Ultralytics brain-tumor images/train after extraction.")
    return root


def _load_brain_tumor_split(
    root: Path,
    split: str,
    *,
    image_size: int,
    per_class_limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from PIL import Image
    except Exception as exc:
        raise ImportError("Pillow is required to load the brain tumor image dataset.") from exc

    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    by_class: dict[int, list[Path]] = {0: [], 1: []}
    for image_path in image_paths:
        label = _brain_tumor_image_label(label_dir / f"{image_path.stem}.txt")
        by_class[label].append(image_path)

    rng = np.random.default_rng(seed)
    selected_paths, selected_labels = [], []
    for label, paths in by_class.items():
        if not paths:
            raise ValueError(f"No images found for class {label} in split {split}.")
        shuffled = np.asarray(paths, dtype=object)
        rng.shuffle(shuffled)
        for image_path in shuffled[: min(per_class_limit, len(shuffled))]:
            selected_paths.append(Path(image_path))
            selected_labels.append(label)

    images = []
    for image_path in selected_paths:
        image = Image.open(image_path).convert("L").resize((image_size, image_size))
        images.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.asarray(images, dtype=np.float32)[:, None, :, :], np.asarray(selected_labels, dtype=np.int64)


def _brain_tumor_image_label(label_path: Path) -> int:
    if not label_path.exists() or not label_path.read_text(encoding="utf-8").strip():
        return 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts and parts[0] == "1":
            return 1
    return 0


def train_or_load_model(
    model_path: Path,
    dataset: MedicalDataset,
    *,
    device: str,
    epochs: int = 16,
    batch_size: int = 64,
) -> BrainTumorCNN:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = BrainTumorCNN(num_classes=len(dataset.class_names)).to(device)
    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            return model
        except RuntimeError:
            model_path.unlink()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(dataset.train_images, dtype=torch.float32),
            torch.tensor(dataset.train_labels, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.002)
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), model_path)
    model.eval()
    return model


def evaluate_model_outputs(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    *,
    device: str,
    max_rows: int = 30,
) -> tuple[dict[str, float], list[dict[str, object]], np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(images, dtype=torch.float32, device=device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    rows = []
    for idx in range(min(max_rows, len(images))):
        rows.append(
            {
                "sample_id": idx,
                "true_class": class_names[int(labels[idx])],
                "predicted_class": class_names[int(preds[idx])],
                "confidence": round(float(confidences[idx]), 6),
                "probability_vector": json.dumps(
                    {class_names[j]: round(float(probs[idx, j]), 6) for j in range(len(class_names))}
                ),
            }
        )
    metrics = {
        "accuracy": float((preds == labels).mean()),
        "mean_confidence": float(confidences.mean()),
        "p95_confidence": float(np.quantile(confidences, 0.95)),
        "mean_entropy": float(np.mean(-np.sum(probs * np.log(probs + 1e-8), axis=1))),
    }
    return metrics, rows, probs


def output_configurations() -> list[OutputConfig]:
    """TODO: Define at least three output configurations to evaluate.

    Suggested configurations:
    - Full probability vector
    - Rounded confidence scores
    - Top-1 label only
    """
    raise NotImplementedError("TODO: return at least three OutputConfig objects.")


def run_inversion_attack(
    model: nn.Module,
    target_classes: Iterable[int],
    *,
    output_config: OutputConfig,
    device: str,
    image_size: int = 64,
    steps: int = 160,
    restarts: int = 2,
    learning_rate: float = 0.08,
) -> dict[int, dict[str, object]]:
    """TODO: Implement repeated-query model inversion.

    Hints:
    - Initialize a candidate image with random pixel values.
    - Optimize it for each target class using the signal exposed by output_config.
    - Add total variation / L2 regularization so outputs remain image-like.
    - Return one reconstructed image and metadata for each target class.
    """
    raise NotImplementedError("TODO: implement the repeated-query inversion attack.")


def _total_variation(image: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + torch.mean(
        torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
    )


def run_mri_prior_inversion_attack(
    model: nn.Module,
    target_classes: Iterable[int],
    reference_images: np.ndarray,
    *,
    device: str,
    image_size: int = 128,
    steps: int = 180,
    restarts: int = 2,
    learning_rate: float = 0.06,
    n_components: int = 80,
) -> dict[int, dict[str, object]]:
    """TODO: Optional extension.

    Constrain inversion to a PCA prior learned from reference MRI images so
    recovered images look more like plausible MRI scans.
    """
    raise NotImplementedError("TODO: implement MRI-prior inversion for recovered MRI examples.")


def representative_samples(images: np.ndarray, labels: np.ndarray, target_classes: Iterable[int]) -> dict[int, np.ndarray]:
    return {target: images[labels == target].mean(axis=0)[0] for target in target_classes}


def nearest_reference_samples(
    reconstructions: dict[int, dict[str, object]],
    images: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """TODO: Find the nearest validation sample for each reconstruction."""
    raise NotImplementedError("TODO: find nearest reference MRI samples.")


def inversion_metrics(
    config_name: str,
    reconstructions: dict[int, dict[str, object]],
    representatives: dict[int, np.ndarray],
    class_names: list[str],
) -> list[dict[str, object]]:
    """TODO: Quantify reconstruction quality and privacy leakage.

    Suggested metrics:
    - MSE against representative validation sample
    - Cosine similarity against representative validation sample
    - Target match flag
    - Privacy leakage score that is high only when the reconstruction matches
      the intended target and resembles the representative sample
    """
    raise NotImplementedError("TODO: calculate inversion privacy metrics.")


def write_csv(rows: list[dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(payload: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def plot_reconstruction_comparison(
    reconstruction_sets: dict[str, dict[int, dict[str, object]]],
    representatives: dict[int, np.ndarray],
    class_names: list[str],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = list(representatives.keys())
    columns = ["representative"] + list(reconstruction_sets.keys())
    fig, axes = plt.subplots(len(targets), len(columns), figsize=(3.0 * len(columns), 2.5 * len(targets)))
    if len(targets) == 1:
        axes = np.asarray([axes])
    for row_idx, target in enumerate(targets):
        axes[row_idx, 0].imshow(representatives[target], cmap="magma", vmin=0, vmax=1)
        axes[row_idx, 0].set_title("Representative sample", fontsize=9)
        axes[row_idx, 0].set_ylabel(class_names[target], rotation=0, labelpad=52, fontsize=9)
        axes[row_idx, 0].axis("off")
        for col_idx, config_name in enumerate(reconstruction_sets, start=1):
            result = reconstruction_sets[config_name][target]
            axes[row_idx, col_idx].imshow(result["image"], cmap="magma", vmin=0, vmax=1)
            axes[row_idx, col_idx].set_title(f"{config_name}\nconf={result['model_confidence']:.3f}", fontsize=8)
            axes[row_idx, col_idx].axis("off")
    fig.suptitle("Medical model inversion: representative samples vs reconstructed features", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_recovered_mri_examples(
    prior_reconstructions: dict[int, dict[str, object]],
    representatives: dict[int, np.ndarray],
    nearest_samples: dict[int, np.ndarray],
    class_names: list[str],
    output_path: Path,
) -> Path:
    """TODO: Plot nearest MRI, class prototype, and recovered MRI."""
    raise NotImplementedError("TODO: plot recovered MRI examples.")


def plot_leakage_scores(rows: list[dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["output_configuration"]), []).append(float(row["privacy_leakage_score"]))
    labels = list(grouped.keys())
    means = [float(np.mean(grouped[label])) for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, means, color=["#3f6c9d", "#7b9e54", "#9d4d4d"][: len(labels)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean leakage score")
    ax.set_title("Output detail changes inversion effectiveness")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(means):
        ax.text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_privacy_report(
    path: Path,
    baseline_metrics: dict[str, float],
    metric_rows: list[dict[str, object]],
    chart_path: Path,
    comparison_path: Path,
) -> Path:
    """TODO: Write a short privacy assessment report.

    Your report should summarize:
    - Baseline confidence behavior
    - Which output configuration leaked the most
    - Operational risks for a cloud-hosted brain tumor screening API
    - Mitigations such as output restriction, rate limiting, confidence clipping,
      overfitting checks, and differential privacy
    """
    raise NotImplementedError("TODO: write the privacy assessment report.")


def art_status() -> dict[str, object]:
    try:
        import art  # noqa: F401

        return {"available": True, "note": "ART is installed; this exercise uses a transparent PyTorch inversion loop."}
    except Exception as exc:
        return {"available": False, "note": f"ART is not installed or not importable in this environment: {exc}"}
