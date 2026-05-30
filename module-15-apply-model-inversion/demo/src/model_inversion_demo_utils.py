from __future__ import annotations

import csv
import json
import math
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


@dataclass(frozen=True)
class DatasetBundle:
    train_images: np.ndarray
    train_labels: np.ndarray
    val_images: np.ndarray
    val_labels: np.ndarray
    class_names: list[str]
    dataset_name: str
    source: str


@dataclass(frozen=True)
class AttackConfig:
    steps: int = 180
    restarts: int = 2
    learning_rate: float = 0.08
    tv_weight: float = 0.0007
    l2_weight: float = 0.0002
    confidence_mode: str = "rich_probability"


class FaceRecognitionCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, 96),
            nn.ReLU(),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def seed_everything(seed: int = 7) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_att_face_dataset(
    data_dir: Path,
    *,
    local_att_dir: Path | None = None,
    train_per_identity: int = 7,
    val_per_identity: int = 3,
    validation_queries: int = 500,
    image_size: int = 64,
    seed: int = 7,
    download_if_missing: bool = True,
) -> DatasetBundle:
    """Load the AT&T/ORL faces dataset from disk or scikit-learn.

    Preferred local layout is the original AT&T archive structure:
    `data/att_faces/s1/1.pgm` through `data/att_faces/s40/10.pgm`.
    If that folder is not present, this loads scikit-learn's Olivetti faces
    package, which is the AT&T Laboratories Cambridge face dataset.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    local_att_dir = local_att_dir or data_dir.parent / "att_faces"
    dataset_path = data_dir / (
        f"att_olivetti_faces_train{train_per_identity}_val{val_per_identity}_"
        f"queries{validation_queries}_size{image_size}.npz"
    )
    if dataset_path.exists():
        payload = np.load(dataset_path, allow_pickle=True)
        return DatasetBundle(
            train_images=payload["train_images"],
            train_labels=payload["train_labels"],
            val_images=payload["val_images"],
            val_labels=payload["val_labels"],
            class_names=list(payload["class_names"]),
            dataset_name=str(payload["dataset_name"]),
            source=str(payload["source"]),
        )

    images, labels, source = _load_local_att_faces(local_att_dir, image_size)
    dataset_name = "AT&T/ORL face dataset"
    if images is None:
        images, labels, source = _load_sklearn_att_faces(data_dir / "sklearn_cache", download_if_missing)
        dataset_name = "AT&T/Olivetti faces from scikit-learn"
        if image_size != images.shape[-1]:
            images = _resize_images(images, image_size)

    class_names = [f"subject_{idx + 1:02d}" for idx in sorted(np.unique(labels))]
    train_images, train_labels, base_val_images, base_val_labels = _stratified_subject_split(
        images,
        labels,
        train_per_identity=train_per_identity,
        val_per_identity=val_per_identity,
        seed=seed,
    )
    val_images, val_labels = _expand_validation_queries(base_val_images, base_val_labels, validation_queries)

    np.savez_compressed(
        dataset_path,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        class_names=np.asarray(class_names),
        dataset_name=dataset_name,
        source=source,
    )
    return DatasetBundle(train_images, train_labels, val_images, val_labels, class_names, dataset_name, source)


def _load_local_att_faces(root: Path, image_size: int) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if not root.exists():
        return None, None, f"Local AT&T folder not found: {root}"
    try:
        from PIL import Image
    except Exception as exc:
        raise ImportError("Pillow is required to load local AT&T .pgm files.") from exc

    images, labels = [], []
    for subject_dir in sorted(root.glob("s*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 9999):
        if not subject_dir.is_dir() or not subject_dir.name[1:].isdigit():
            continue
        label = int(subject_dir.name[1:]) - 1
        for image_path in sorted(subject_dir.glob("*.pgm"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            image = Image.open(image_path).convert("L").resize((image_size, image_size))
            images.append(np.asarray(image, dtype=np.float32) / 255.0)
            labels.append(label)
    if not images:
        return None, None, f"No .pgm files found under local AT&T folder: {root}"
    return np.asarray(images, dtype=np.float32)[:, None, :, :], np.asarray(labels, dtype=np.int64), str(root)


def _load_sklearn_att_faces(cache_dir: Path, download_if_missing: bool) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from sklearn.datasets import fetch_olivetti_faces
    except Exception as exc:
        raise ImportError("scikit-learn is required to load the AT&T/Olivetti faces dataset.") from exc

    faces = fetch_olivetti_faces(
        data_home=str(cache_dir),
        shuffle=False,
        download_if_missing=download_if_missing,
    )
    return faces.images.astype(np.float32)[:, None, :, :], faces.target.astype(np.int64), str(cache_dir)


def _resize_images(images: np.ndarray, image_size: int) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:
        raise ImportError("Pillow is required when resizing AT&T face images.") from exc

    resized = []
    for image in images[:, 0]:
        pil_image = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255), mode="L")
        resized.append(np.asarray(pil_image.resize((image_size, image_size)), dtype=np.float32) / 255.0)
    return np.asarray(resized, dtype=np.float32)[:, None, :, :]


def _stratified_subject_split(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    train_per_identity: int,
    val_per_identity: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_images, train_labels, val_images, val_labels = [], [], [], []
    for label in sorted(np.unique(labels)):
        subject_indices = np.where(labels == label)[0]
        rng.shuffle(subject_indices)
        required = train_per_identity + val_per_identity
        if len(subject_indices) < required:
            raise ValueError(
                f"Subject {label} has {len(subject_indices)} images, but {required} are required "
                "for the requested train/validation split."
            )
        train_idx = subject_indices[:train_per_identity]
        val_idx = subject_indices[train_per_identity:required]
        train_images.extend(images[train_idx])
        train_labels.extend(labels[train_idx])
        val_images.extend(images[val_idx])
        val_labels.extend(labels[val_idx])
    return (
        np.asarray(train_images, dtype=np.float32),
        np.asarray(train_labels, dtype=np.int64),
        np.asarray(val_images, dtype=np.float32),
        np.asarray(val_labels, dtype=np.int64),
    )


def _expand_validation_queries(images: np.ndarray, labels: np.ndarray, validation_queries: int) -> tuple[np.ndarray, np.ndarray]:
    if validation_queries <= len(images):
        return images[:validation_queries], labels[:validation_queries]
    repeats = int(math.ceil(validation_queries / len(images)))
    return np.tile(images, (repeats, 1, 1, 1))[:validation_queries], np.tile(labels, repeats)[:validation_queries]


def train_or_load_model(
    model_path: Path,
    dataset: DatasetBundle,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    overfit: bool,
) -> FaceRecognitionCNN:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = FaceRecognitionCNN(len(dataset.class_names), dropout=0.0 if overfit else 0.20).to(device)
    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            return model
        except RuntimeError:
            model_path.unlink()

    train_x = torch.tensor(dataset.train_images, dtype=torch.float32)
    train_y = torch.tensor(dataset.train_labels, dtype=torch.long)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0025 if overfit else 0.0015,
        weight_decay=0.0 if overfit else 0.01,
    )
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.0 if overfit else 0.04)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), model_path)
    model.eval()
    return model


def evaluate_predictions(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    *,
    device: str,
    max_rows: int = 30,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(images, dtype=torch.float32, device=device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    rows = []
    for idx in range(min(max_rows, len(images))):
        top = probs[idx].argsort()[-3:][::-1]
        rows.append(
            {
                "sample_id": idx,
                "true_identity": class_names[int(labels[idx])],
                "predicted_identity": class_names[int(preds[idx])],
                "confidence": round(float(confidence[idx]), 6),
                "top3_probability_vector": json.dumps(
                    {class_names[int(j)]: round(float(probs[idx, j]), 6) for j in top}
                ),
            }
        )
    metrics = {
        "accuracy": float((preds == labels).mean()),
        "mean_confidence": float(confidence.mean()),
        "p95_confidence": float(np.quantile(confidence, 0.95)),
    }
    return metrics, rows


def invert_model_outputs(
    model: nn.Module,
    target_classes: Iterable[int],
    *,
    config: AttackConfig,
    device: str,
    image_size: int = 32,
) -> dict[int, dict[str, object]]:
    model.eval()
    results = {}
    for target in target_classes:
        best_loss = math.inf
        best_image = None
        for restart in range(config.restarts):
            torch.manual_seed(1000 + target * 17 + restart)
            candidate = torch.rand((1, 1, image_size, image_size), device=device, requires_grad=True)
            optimizer = torch.optim.Adam([candidate], lr=config.learning_rate)
            target_tensor = torch.tensor([target], device=device)
            for _ in range(config.steps):
                optimizer.zero_grad(set_to_none=True)
                clamped = candidate.clamp(0, 1)
                logits = model(clamped)
                probs = F.softmax(logits, dim=1)
                if config.confidence_mode == "rounded_probability":
                    rounded = torch.round(probs[:, target] * 10) / 10
                    confidence_term = -rounded.detach()
                else:
                    confidence_term = F.cross_entropy(logits, target_tensor)
                loss = confidence_term.mean()
                loss = loss + config.tv_weight * _total_variation(clamped) + config.l2_weight * torch.mean(clamped**2)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    candidate.clamp_(0, 1)
            final_loss = float(loss.detach().cpu())
            if final_loss < best_loss:
                best_loss = final_loss
                best_image = candidate.detach().cpu().numpy()[0, 0]

        assert best_image is not None
        with torch.no_grad():
            image_t = torch.tensor(best_image[None, None, :, :], dtype=torch.float32, device=device)
            output_probs = F.softmax(model(image_t), dim=1).cpu().numpy()[0]
        results[target] = {
            "image": best_image,
            "confidence": float(output_probs[target]),
            "predicted_class": int(output_probs.argmax()),
            "queries": config.steps * config.restarts,
        }
    return results


def invert_model_outputs_with_face_prior(
    model: nn.Module,
    target_classes: Iterable[int],
    reference_images: np.ndarray,
    *,
    config: AttackConfig,
    device: str,
    n_components: int = 80,
) -> dict[int, dict[str, object]]:
    """Invert model outputs in an eigenface-style latent space.

    Optimizing raw pixels often discovers model-specific textures. Constraining
    the search to a PCA face basis makes the result easier to interpret as a
    reconstructed face while still using model confidence as the attack signal.
    """
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:
        raise ImportError("scikit-learn is required for face-prior inversion.") from exc

    model.eval()
    image_size = int(reference_images.shape[-1])
    flat_reference = reference_images.reshape(len(reference_images), -1)
    component_count = min(n_components, len(flat_reference) - 1, flat_reference.shape[1])
    pca = PCA(n_components=component_count, random_state=7)
    pca.fit(flat_reference)

    mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device)
    components = torch.tensor(pca.components_, dtype=torch.float32, device=device)
    component_scale = torch.tensor(np.sqrt(pca.explained_variance_ + 1e-8), dtype=torch.float32, device=device)
    results = {}
    for target in target_classes:
        best_loss = math.inf
        best_image = None
        for restart in range(config.restarts):
            torch.manual_seed(3000 + target * 19 + restart)
            coeffs = torch.zeros((1, component_count), device=device, requires_grad=True)
            coeffs.data.normal_(0.0, 0.05)
            optimizer = torch.optim.Adam([coeffs], lr=config.learning_rate)
            target_tensor = torch.tensor([target], device=device)
            for _ in range(config.steps):
                optimizer.zero_grad(set_to_none=True)
                flat = mean + coeffs @ components
                candidate = flat.reshape(1, 1, image_size, image_size).clamp(0, 1)
                logits = model(candidate)
                loss = F.cross_entropy(logits, target_tensor)
                loss = loss + 0.015 * torch.mean((coeffs / component_scale) ** 2) + (config.tv_weight * 3.0) * _total_variation(candidate)
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
            "confidence": float(output_probs[target]),
            "predicted_class": int(output_probs.argmax()),
            "queries": config.steps * config.restarts,
        }
    return results


def _total_variation(image: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + torch.mean(
        torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
    )


def class_prototypes(images: np.ndarray, labels: np.ndarray, target_classes: Iterable[int]) -> dict[int, np.ndarray]:
    return {target: images[labels == target].mean(axis=0)[0] for target in target_classes}


def nearest_training_examples(
    reconstructions: dict[int, dict[str, object]],
    train_images: np.ndarray,
    train_labels: np.ndarray,
) -> dict[int, np.ndarray]:
    nearest = {}
    for target, result in reconstructions.items():
        same_class = train_images[train_labels == target]
        reconstruction = result["image"]
        distances = np.mean((same_class[:, 0] - reconstruction) ** 2, axis=(1, 2))
        nearest[target] = same_class[int(np.argmin(distances)), 0]
    return nearest


def reconstruction_metrics(
    attack_name: str,
    reconstructions: dict[int, dict[str, object]],
    prototypes: dict[int, np.ndarray],
    class_names: list[str],
) -> list[dict[str, object]]:
    rows = []
    for target, result in reconstructions.items():
        image = result["image"]
        prototype = prototypes[target]
        mse = float(np.mean((image - prototype) ** 2))
        cosine = float(np.dot(image.ravel(), prototype.ravel()) / (np.linalg.norm(image.ravel()) * np.linalg.norm(prototype.ravel()) + 1e-8))
        target_match = int(result["predicted_class"]) == target
        rows.append(
            {
                "attack_setting": attack_name,
                "target_identity": class_names[target],
                "target_confidence": round(float(result["confidence"]), 6),
                "predicted_identity": class_names[int(result["predicted_class"])],
                "target_match": target_match,
                "queries": int(result["queries"]),
                "prototype_mse": round(mse, 6),
                "prototype_cosine_similarity": round(cosine, 6),
                "privacy_leakage_score": round(max(0.0, min(1.0, cosine)) if target_match else 0.0, 6),
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


def plot_reconstruction_grid(
    rich_reconstructions: dict[int, dict[str, object]],
    rounded_reconstructions: dict[int, dict[str, object]],
    prototypes: dict[int, np.ndarray],
    class_names: list[str],
    output_path: Path,
) -> Path:
    targets = list(prototypes.keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(targets), 3, figsize=(7.5, 2.2 * len(targets)))
    if len(targets) == 1:
        axes = np.asarray([axes])
    for row_idx, target in enumerate(targets):
        panels = [
            ("Representative train prototype", prototypes[target]),
            ("Rich probability inversion", rich_reconstructions[target]["image"]),
            ("Rounded probability inversion", rounded_reconstructions[target]["image"]),
        ]
        for col_idx, (title, image) in enumerate(panels):
            axes[row_idx, col_idx].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row_idx, col_idx].set_title(title, fontsize=9)
            axes[row_idx, col_idx].axis("off")
        axes[row_idx, 0].set_ylabel(class_names[target], rotation=0, labelpad=38, fontsize=9)
    fig.suptitle("Model inversion: reconstructed facial feature approximations", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_face_recovery_examples(
    prior_reconstructions: dict[int, dict[str, object]],
    prototypes: dict[int, np.ndarray],
    nearest_examples: dict[int, np.ndarray],
    class_names: list[str],
    output_path: Path,
) -> Path:
    targets = list(prior_reconstructions.keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(targets), 3, figsize=(7.5, 2.2 * len(targets)))
    if len(targets) == 1:
        axes = np.asarray([axes])
    for row_idx, target in enumerate(targets):
        recovered = prior_reconstructions[target]["image"]
        confidence = float(prior_reconstructions[target]["confidence"])
        panels = [
            ("Nearest training image", nearest_examples[target]),
            ("Training class prototype", prototypes[target]),
            (f"Recovered via inversion\nconfidence={confidence:.3f}", recovered),
        ]
        for col_idx, (title, image) in enumerate(panels):
            axes[row_idx, col_idx].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row_idx, col_idx].set_title(title, fontsize=9)
            axes[row_idx, col_idx].axis("off")
        axes[row_idx, 0].set_ylabel(class_names[target], rotation=0, labelpad=38, fontsize=9)
    fig.suptitle("Face-prior model inversion: recovered face-like examples", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_leakage_comparison(rows: list[dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["attack_setting"]), []).append(float(row["privacy_leakage_score"]))
    labels = list(grouped.keys())
    means = [np.mean(grouped[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, means, color=["#4062bb", "#59a14f"][: len(labels)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean prototype similarity")
    ax.set_title("Output detail changes inversion effectiveness")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(means):
        ax.text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def art_status() -> dict[str, object]:
    try:
        import art  # noqa: F401

        return {"available": True, "note": "ART is installed; this demo uses a transparent PyTorch inversion loop for teachability."}
    except Exception as exc:
        return {"available": False, "note": f"ART is not installed or not importable in this environment: {exc}"}
