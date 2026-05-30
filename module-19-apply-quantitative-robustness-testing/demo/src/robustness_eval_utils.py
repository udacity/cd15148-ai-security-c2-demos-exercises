from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str
    augment_noise_std: float = 0.0
    label_smoothing: float = 0.0


class NumpyImageDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.tensor(self.images[index], dtype=torch.float32)
        if self.augment is not None:
            image = self.augment(image)
        return image, torch.tensor(self.labels[index], dtype=torch.long)


def build_resnet18_cifar10() -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, len(CIFAR10_CLASSES))
    return model


def prepare_cifar10_assets(
    output_dir: Path | str,
    download_root: Path | str,
    train_per_class: int = 150,
    val_per_class: int = 100,
    seed: int = 19,
) -> Path:
    output_dir = Path(output_dir)
    download_root = Path(download_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_subset.npz"
    val_path = output_dir / "validation_1000.npz"
    if train_path.exists() and val_path.exists():
        return output_dir

    transform = transforms.ToTensor()
    train_source = datasets.CIFAR10(root=str(download_root), train=True, download=True, transform=transform)
    val_source = datasets.CIFAR10(root=str(download_root), train=False, download=True, transform=transform)
    rng = np.random.default_rng(seed)

    def select_balanced(source: Dataset, per_class: int) -> tuple[np.ndarray, np.ndarray]:
        images: list[np.ndarray] = []
        labels: list[int] = []
        for class_id in range(len(CIFAR10_CLASSES)):
            indices = [idx for idx, (_, label) in enumerate(source) if label == class_id]
            selected = rng.choice(indices, size=per_class, replace=False)
            for idx in selected:
                image, label = source[int(idx)]
                images.append(image.numpy())
                labels.append(int(label))
        order = rng.permutation(len(labels))
        return np.stack(images)[order], np.array(labels, dtype=np.int64)[order]

    train_images, train_labels = select_balanced(train_source, train_per_class)
    val_images, val_labels = select_balanced(val_source, val_per_class)
    np.savez_compressed(train_path, images=train_images, labels=train_labels)
    np.savez_compressed(val_path, images=val_images, labels=val_labels)
    return output_dir


def load_npz_dataset(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["images"].astype(np.float32), data["labels"].astype(np.int64)


def _augmentation(noise_std: float) -> Callable[[torch.Tensor], torch.Tensor] | None:
    if noise_std <= 0:
        return None

    def augment(image: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=[2])
        noise = torch.randn_like(image) * noise_std
        return torch.clamp(image + noise, 0.0, 1.0)

    return augment


def train_or_load_model(
    model_path: Path | str,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    spec: ModelSpec,
    device: str = "cpu",
    epochs: int = 2,
    batch_size: int = 128,
) -> nn.Module:
    model_path = Path(model_path)
    model = build_resnet18_cifar10().to(device)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    dataset = NumpyImageDataset(train_images, train_labels, augment=_augmentation(spec.augment_noise_std))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=spec.label_smoothing)

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
        print(f"{spec.name} epoch {epoch + 1}/{epochs} loss={running_loss / max(len(loader), 1):.4f}")

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return model


def predict(model: nn.Module, images: np.ndarray, device: str = "cpu", batch_size: int = 256) -> dict[str, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
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


def evaluate_predictions(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    clean_predictions: np.ndarray | None = None,
    clean_confidence: np.ndarray | None = None,
    device: str = "cpu",
) -> dict[str, float | np.ndarray]:
    outputs = predict(model, images, device=device)
    predictions = outputs["predictions"]
    confidence = outputs["confidence"]
    accuracy = float(accuracy_score(labels, predictions))
    confidence_drop = 0.0
    attack_success_rate = 0.0

    if clean_confidence is not None:
        confidence_drop = float(np.mean(clean_confidence - confidence))
    if clean_predictions is not None:
        clean_correct = clean_predictions == labels
        if np.any(clean_correct):
            attack_success_rate = float(np.mean(predictions[clean_correct] != labels[clean_correct]))

    return {
        "accuracy": accuracy,
        "mean_confidence": float(np.mean(confidence)),
        "confidence_drop": confidence_drop,
        "attack_success_rate": attack_success_rate,
        "predictions": predictions,
        "confidence": confidence,
    }


def gaussian_noise(images: np.ndarray, sigma: float, seed: int = 19) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(images + rng.normal(0.0, sigma, size=images.shape), 0.0, 1.0).astype(np.float32)


def gaussian_blur(images: np.ndarray, radius: float) -> np.ndarray:
    blurred = []
    for image in images:
        pil_image = _numpy_chw_to_pil(image)
        blurred.append(_pil_to_numpy_chw(pil_image.filter(ImageFilter.GaussianBlur(radius=radius))))
    return np.stack(blurred).astype(np.float32)


def jpeg_compression(images: np.ndarray, quality: int) -> np.ndarray:
    compressed = []
    for image in images:
        pil_image = _numpy_chw_to_pil(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed.append(_pil_to_numpy_chw(Image.open(buffer).convert("RGB")))
    return np.stack(compressed).astype(np.float32)


def _numpy_chw_to_pil(image: np.ndarray) -> Image.Image:
    array = np.transpose(np.clip(image, 0.0, 1.0), (1, 2, 0))
    return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")


def _pil_to_numpy_chw(image: Image.Image) -> np.ndarray:
    array = np.asarray(image).astype(np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def run_environmental_suite(
    model_name: str,
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    device: str,
) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray]]:
    clean = evaluate_predictions(model, images, labels, device=device)
    clean_predictions = clean["predictions"]
    clean_confidence = clean["confidence"]
    rows = [
        _metric_row(model_name, "clean", "baseline", clean, perturbation_linf=0.0),
    ]

    distorted_sets = {
        "gaussian_noise_sigma_0.08": gaussian_noise(images, sigma=0.08),
        "gaussian_blur_radius_1.0": gaussian_blur(images, radius=1.0),
        "jpeg_quality_35": jpeg_compression(images, quality=35),
    }
    for condition, distorted_images in distorted_sets.items():
        metrics = evaluate_predictions(
            model,
            distorted_images,
            labels,
            clean_predictions=clean_predictions,
            clean_confidence=clean_confidence,
            device=device,
        )
        perturbation = perturbation_metrics(images, distorted_images)
        rows.append(_metric_row(model_name, condition, "environmental", metrics, perturbation["linf_mean"]))

    return rows, {"predictions": clean_predictions, "confidence": clean_confidence}


def build_art_classifier(model: nn.Module):
    try:
        from art.estimators.classification import PyTorchClassifier
    except ImportError as error:
        raise RuntimeError(
            "Adversarial Robustness Toolbox is required for ART attacks. "
            "Install adversarial-robustness-toolbox in the course environment."
        ) from error

    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        input_shape=(3, 32, 32),
        nb_classes=len(CIFAR10_CLASSES),
        clip_values=(0.0, 1.0),
    )


def run_art_attack_suite(
    model_name: str,
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    clean_state: dict[str, np.ndarray],
    device: str,
    epsilons: Iterable[float] = (0.02, 0.04, 0.08),
) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray]]:
    try:
        from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    except ImportError as error:
        raise RuntimeError(
            "Adversarial Robustness Toolbox is required for ART attacks. "
            "Install adversarial-robustness-toolbox in the course environment."
        ) from error

    classifier = build_art_classifier(model)
    rows: list[dict[str, float | str]] = []
    example_images: dict[str, np.ndarray] = {}

    for epsilon in epsilons:
        attack = FastGradientMethod(estimator=classifier, eps=epsilon)
        adversarial = attack.generate(x=images)
        metrics = evaluate_predictions(
            model,
            adversarial,
            labels,
            clean_predictions=clean_state["predictions"],
            clean_confidence=clean_state["confidence"],
            device=device,
        )
        perturbation = perturbation_metrics(images, adversarial)
        condition = f"art_fgsm_eps_{epsilon:.2f}"
        rows.append(_metric_row(model_name, condition, "adversarial", metrics, perturbation["linf_mean"]))
        example_images[condition] = adversarial

    pgd = ProjectedGradientDescent(estimator=classifier, eps=0.04, eps_step=0.01, max_iter=5, verbose=False)
    adversarial = pgd.generate(x=images)
    metrics = evaluate_predictions(
        model,
        adversarial,
        labels,
        clean_predictions=clean_state["predictions"],
        clean_confidence=clean_state["confidence"],
        device=device,
    )
    perturbation = perturbation_metrics(images, adversarial)
    rows.append(_metric_row(model_name, "art_pgd_eps_0.04", "adversarial", metrics, perturbation["linf_mean"]))
    example_images["art_pgd_eps_0.04"] = adversarial
    return rows, example_images


def run_counterfit_example_if_available(target, attack_plan: dict) -> list[dict[str, float | str]]:
    try:
        from counterfit import Counterfit
    except ImportError:
        return [
            {
                "model": "counterfit_target",
                "condition": "counterfit_optional",
                "condition_type": "tooling",
                "accuracy": "",
                "mean_confidence": "",
                "confidence_drop": "",
                "attack_success_rate": "",
                "perturbation_linf": "",
                "operational_status": "Counterfit is not installed in this environment",
            }
        ]

    rows = []
    for attack_config in attack_plan.get("attacks", []):
        attack_name = attack_config["name"]
        cfattack = Counterfit.build_attack(target, attack_name)
        if not cfattack:
            continue
        cfattack.options.update(attack_config.get("parameters", {}))
        run_ok = Counterfit.run_attack(cfattack)
        rows.append(
            {
                "model": "counterfit_target",
                "condition": attack_name.split(".")[-1],
                "condition_type": "counterfit",
                "accuracy": "",
                "mean_confidence": "",
                "confidence_drop": "",
                "attack_success_rate": _counterfit_success_rate(cfattack) if run_ok else "",
                "perturbation_linf": "",
                "operational_status": "executed" if run_ok else "failed",
            }
        )
    return rows


def _counterfit_success_rate(cfattack) -> float:
    success_values = cfattack.success
    if isinstance(success_values, (list, tuple, np.ndarray)):
        return round(float(np.mean(success_values)), 4)
    return round(float(bool(success_values)), 4)


def perturbation_metrics(clean: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = np.abs(candidate.astype(np.float32) - clean.astype(np.float32))
    return {
        "l1_mean": float(delta.mean()),
        "l2_mean": float(np.sqrt(np.mean(np.square(delta), axis=(1, 2, 3))).mean()),
        "linf_mean": float(delta.reshape(delta.shape[0], -1).max(axis=1).mean()),
    }


def _metric_row(
    model_name: str,
    condition: str,
    condition_type: str,
    metrics: dict[str, float | np.ndarray],
    perturbation_linf: float,
) -> dict[str, float | str]:
    accuracy = float(metrics["accuracy"])
    attack_success_rate = float(metrics["attack_success_rate"])
    confidence_drop = float(metrics["confidence_drop"])
    return {
        "model": model_name,
        "condition": condition,
        "condition_type": condition_type,
        "accuracy": round(accuracy, 4),
        "mean_confidence": round(float(metrics["mean_confidence"]), 4),
        "confidence_drop": round(confidence_drop, 4),
        "attack_success_rate": round(attack_success_rate, 4),
        "perturbation_linf": round(float(perturbation_linf), 4),
        "operational_status": classify_operational_status(accuracy, confidence_drop, attack_success_rate),
    }


def classify_operational_status(accuracy: float, confidence_drop: float, attack_success_rate: float) -> str:
    if accuracy < 0.55 or attack_success_rate > 0.35:
        return "fails expansion gate"
    if accuracy < 0.70 or confidence_drop > 0.15 or attack_success_rate > 0.20:
        return "requires mitigation"
    return "meets demo threshold"


def robustness_scorecard(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    scored_rows = []
    for row in rows:
        if row["accuracy"] == "":
            scored_rows.append({**row, "robustness_score": ""})
            continue
        accuracy = float(row["accuracy"])
        confidence_drop = max(float(row["confidence_drop"]), 0.0)
        attack_success_rate = float(row["attack_success_rate"])
        score = 100.0 * (0.60 * accuracy + 0.25 * (1.0 - attack_success_rate) + 0.15 * (1.0 - confidence_drop))
        scored_rows.append({**row, "robustness_score": round(score, 1)})
    return scored_rows


def write_csv(rows: list[dict[str, float | str]], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "condition",
        "condition_type",
        "accuracy",
        "mean_confidence",
        "confidence_drop",
        "attack_success_rate",
        "perturbation_linf",
        "robustness_score",
        "operational_status",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def load_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_scorecard(rows: list[dict[str, float | str]], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_rows = [row for row in rows if row["robustness_score"] != ""]
    labels = [f"{row['model']}\n{row['condition']}" for row in plot_rows]
    scores = [float(row["robustness_score"]) for row in plot_rows]

    fig_width = max(10, len(scores) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bars = ax.bar(range(len(scores)), scores, color="#2f6f6d")
    ax.axhline(75, color="#7a3e00", linestyle="--", linewidth=1.5, label="Demo release gate")
    ax.set_ylabel("Robustness score, higher is better")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.bar_label(bars, fmt="%.1f", padding=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_example_grid(
    clean_images: np.ndarray,
    perturbed_images: np.ndarray,
    output_path: Path | str,
    title: str,
    max_items: int = 6,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(max_items, len(clean_images))
    fig, axes = plt.subplots(2, count, figsize=(count * 2, 4))
    for idx in range(count):
        axes[0, idx].imshow(np.transpose(clean_images[idx], (1, 2, 0)))
        axes[0, idx].set_title("Clean")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(np.transpose(np.clip(perturbed_images[idx], 0, 1), (1, 2, 0)))
        axes[1, idx].set_title("Perturbed")
        axes[1, idx].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
