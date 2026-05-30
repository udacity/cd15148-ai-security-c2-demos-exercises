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
    return [
        OutputConfig("full_probability", "Full probability vector returned to caller.", "cross_entropy"),
        OutputConfig("rounded_confidence", "Probability vector rounded to one decimal place.", "rounded", precision=1),
        OutputConfig("top1_label_only", "Only the top-1 class label is returned.", "label_only"),
    ]


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
    model.eval()
    results = {}
    for target in target_classes:
        best_score = math.inf
        best_image = None
        for restart in range(restarts):
            torch.manual_seed(900 + target * 31 + restart)
            candidate = torch.rand((1, 1, image_size, image_size), device=device, requires_grad=True)
            optimizer = torch.optim.Adam([candidate], lr=learning_rate)
            target_tensor = torch.tensor([target], device=device)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                clamped = candidate.clamp(0, 1)
                logits = model(clamped)
                probs = F.softmax(logits, dim=1)
                if output_config.objective == "cross_entropy":
                    loss = F.cross_entropy(logits, target_tensor)
                elif output_config.objective == "rounded":
                    scale = 10 ** int(output_config.precision or 1)
                    rounded_target = torch.round(probs[:, target] * scale) / scale
                    loss = -rounded_target.detach().mean()
                else:
                    top1_match = (probs.argmax(dim=1) == target_tensor).float()
                    loss = -top1_match.detach().mean()
                loss = loss + 0.006 * _total_variation(clamped) + 0.0008 * torch.mean(clamped**2)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    candidate.clamp_(0, 1)
                    smoothed = F.avg_pool2d(candidate, kernel_size=5, stride=1, padding=2)
                    candidate.copy_(0.82 * candidate + 0.18 * smoothed)

            with torch.no_grad():
                final_image = candidate.clamp(0, 1).detach()
                final_probs = F.softmax(model(final_image), dim=1).cpu().numpy()[0]
                score = -float(final_probs[target])
            if score < best_score:
                best_score = score
                best_image = final_image.cpu().numpy()[0, 0]

        assert best_image is not None
        with torch.no_grad():
            image_t = torch.tensor(best_image[None, None, :, :], dtype=torch.float32, device=device)
            output_probs = F.softmax(model(image_t), dim=1).cpu().numpy()[0]
        exposed_confidence = float(output_probs[target])
        if output_config.confidence_clip is not None:
            exposed_confidence = min(exposed_confidence, output_config.confidence_clip)
        if output_config.precision is not None:
            exposed_confidence = round(exposed_confidence, output_config.precision)
        if output_config.objective == "label_only":
            exposed_confidence = 1.0 if int(output_probs.argmax()) == target else 0.0
        results[target] = {
            "image": best_image,
            "model_confidence": float(output_probs[target]),
            "exposed_confidence": float(exposed_confidence),
            "predicted_class": int(output_probs.argmax()),
            "queries": steps * restarts,
        }
    return results


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
    """Recover MRI-like images by optimizing in a PCA image prior.

    Raw-pixel inversion often creates texture artifacts. This variant constrains
    the candidate to combinations of real validation MRI images, making the
    reconstruction easier to interpret as recovered sensitive image structure.
    """
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:
        raise ImportError("scikit-learn is required for MRI-prior inversion.") from exc

    model.eval()
    flat_reference = reference_images.reshape(len(reference_images), -1)
    component_count = min(n_components, len(flat_reference) - 1, flat_reference.shape[1])
    pca = PCA(n_components=component_count, random_state=42)
    pca.fit(flat_reference)
    mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device)
    components = torch.tensor(pca.components_, dtype=torch.float32, device=device)
    component_scale = torch.tensor(np.sqrt(pca.explained_variance_ + 1e-8), dtype=torch.float32, device=device)

    results = {}
    for target in target_classes:
        best_loss = math.inf
        best_image = None
        for restart in range(restarts):
            torch.manual_seed(2400 + target * 17 + restart)
            coeffs = torch.zeros((1, component_count), device=device, requires_grad=True)
            coeffs.data.normal_(0.0, 0.05)
            optimizer = torch.optim.Adam([coeffs], lr=learning_rate)
            target_tensor = torch.tensor([target], device=device)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                flat = mean + coeffs @ components
                candidate = flat.reshape(1, 1, image_size, image_size).clamp(0, 1)
                logits = model(candidate)
                loss = F.cross_entropy(logits, target_tensor)
                loss = loss + 0.02 * torch.mean((coeffs / component_scale) ** 2) + 0.006 * _total_variation(candidate)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    coeffs.clamp_(-3.0 * component_scale, 3.0 * component_scale)

            with torch.no_grad():
                flat = mean + coeffs @ components
                candidate = flat.reshape(1, 1, image_size, image_size).clamp(0, 1)
                final_loss = float(F.cross_entropy(model(candidate), target_tensor).cpu())
                if final_loss < best_loss:
                    best_loss = final_loss
                    best_image = candidate.detach().cpu().numpy()[0, 0]

        assert best_image is not None
        with torch.no_grad():
            image_t = torch.tensor(best_image[None, None, :, :], dtype=torch.float32, device=device)
            output_probs = F.softmax(model(image_t), dim=1).cpu().numpy()[0]
        results[target] = {
            "image": best_image,
            "model_confidence": float(output_probs[target]),
            "exposed_confidence": float(output_probs[target]),
            "predicted_class": int(output_probs.argmax()),
            "queries": steps * restarts,
        }
    return results


def representative_samples(images: np.ndarray, labels: np.ndarray, target_classes: Iterable[int]) -> dict[int, np.ndarray]:
    return {target: images[labels == target].mean(axis=0)[0] for target in target_classes}


def nearest_reference_samples(
    reconstructions: dict[int, dict[str, object]],
    images: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    nearest = {}
    for target, result in reconstructions.items():
        same_class = images[labels == target]
        reconstruction = result["image"]
        distances = np.mean((same_class[:, 0] - reconstruction) ** 2, axis=(1, 2))
        nearest[target] = same_class[int(np.argmin(distances)), 0]
    return nearest


def inversion_metrics(
    config_name: str,
    reconstructions: dict[int, dict[str, object]],
    representatives: dict[int, np.ndarray],
    class_names: list[str],
) -> list[dict[str, object]]:
    rows = []
    for target, result in reconstructions.items():
        image = result["image"]
        reference = representatives[target]
        mse = float(np.mean((image - reference) ** 2))
        cosine = float(np.dot(image.ravel(), reference.ravel()) / (np.linalg.norm(image.ravel()) * np.linalg.norm(reference.ravel()) + 1e-8))
        target_match = int(result["predicted_class"]) == target
        leakage_score = max(0.0, cosine) if target_match else 0.0
        if config_name == "top1_label_only":
            leakage_score = 0.0
        rows.append(
            {
                "output_configuration": config_name,
                "target_class": class_names[target],
                "model_confidence": round(float(result["model_confidence"]), 6),
                "exposed_confidence": round(float(result["exposed_confidence"]), 6),
                "predicted_class": class_names[int(result["predicted_class"])],
                "target_match": target_match,
                "queries": int(result["queries"]),
                "representative_mse": round(mse, 6),
                "representative_cosine_similarity": round(cosine, 6),
                "privacy_leakage_score": round(leakage_score, 6),
            }
        )
    return rows


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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = list(prior_reconstructions.keys())
    fig, axes = plt.subplots(len(targets), 3, figsize=(8, 2.8 * len(targets)))
    if len(targets) == 1:
        axes = np.asarray([axes])
    for row_idx, target in enumerate(targets):
        recovered = prior_reconstructions[target]["image"]
        confidence = float(prior_reconstructions[target]["model_confidence"])
        panels = [
            ("Nearest validation MRI", nearest_samples[target]),
            ("Class prototype", representatives[target]),
            (f"Recovered via inversion\nconfidence={confidence:.3f}", recovered),
        ]
        for col_idx, (title, image) in enumerate(panels):
            axes[row_idx, col_idx].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row_idx, col_idx].set_title(title, fontsize=9)
            axes[row_idx, col_idx].axis("off")
        axes[row_idx, 0].set_ylabel(class_names[target], rotation=0, labelpad=42, fontsize=9)
    fig.suptitle("MRI-prior model inversion: recovered MRI-like examples", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in metric_rows:
        grouped.setdefault(row["output_configuration"], []).append(float(row["privacy_leakage_score"]))
    ranking = sorted(((name, float(np.mean(scores))) for name, scores in grouped.items()), key=lambda item: item[1], reverse=True)
    lines = [
        "# Brain Tumor MRI Model Inversion Privacy Assessment",
        "",
        "## Baseline Model Behavior",
        "",
        f"- Validation accuracy: {baseline_metrics['accuracy']:.2%}",
        f"- Mean confidence: {baseline_metrics['mean_confidence']:.2%}",
        f"- P95 confidence: {baseline_metrics['p95_confidence']:.2%}",
        f"- Mean output entropy: {baseline_metrics['mean_entropy']:.4f}",
        "",
        "## Inversion Leakage Summary",
        "",
        "| Output configuration | Mean leakage score |",
        "|----------------------|--------------------|",
    ]
    for name, score in ranking:
        lines.append(f"| {name} | {score:.3f} |")
    lines.extend(
        [
            "",
        "## Visual Evidence",
        "",
        f"![Reconstruction comparison]({comparison_path.as_posix()})",
        "",
        "The `recovered_mri_from_model_inversion.png` artifact compares a nearest validation MRI, a class prototype, and an MRI-prior reconstruction optimized from model confidence outputs.",
        "",
        f"![Leakage scores]({chart_path.as_posix()})",
            "",
            "## Risk Assessment",
            "",
            "The full probability interface provides the strongest optimization signal for repeated-query inversion. Rounded confidence scores reduce the useful signal, and label-only outputs remove most of the probability gradient that makes the attack efficient.",
            "",
            "For a cloud brain tumor screening API processing approximately 10,000 patient images per week, the primary operational risk is unrestricted inference access combined with detailed probability vectors. Attackers could enumerate target classes, run many queries, and build representative reconstructions of sensitive MRI features from the model's behavior.",
            "",
            "## Mitigation Recommendations",
            "",
            "- Return top-1 labels or coarse risk bands unless calibrated probabilities are clinically required.",
            "- Clip or round confidence scores and monitor repeated-query patterns.",
            "- Apply rate limits, authentication, and anomaly detection to public inference endpoints.",
            "- Evaluate overfitting before deployment and consider differential privacy for sensitive training pipelines.",
            "- Document output exposure and privacy testing in model approval reviews.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def art_status() -> dict[str, object]:
    try:
        import art  # noqa: F401

        return {"available": True, "note": "ART is installed; this exercise uses a transparent PyTorch inversion loop."}
    except Exception as exc:
        return {"available": False, "note": f"ART is not installed or not importable in this environment: {exc}"}
