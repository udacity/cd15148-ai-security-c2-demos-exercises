from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms


DRONE_CLASSES = ["no_drone", "drone"]
CIFAR10_SOURCE_CLASSES = {
    0: "airplane",
    1: "automobile",
    3: "cat",
    8: "ship",
    9: "truck",
}
DRONE_PROXY_CLASS = 0
NO_DRONE_PROXY_CLASSES = [1, 3, 8, 9]


class NumpyImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.images[index]), torch.tensor(self.labels[index])


def build_resnet18_drone_detector():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, len(DRONE_CLASSES))
    return model


def prepare_drone_detection_assets(
    output_dir,
    download_root,
    train_per_class=300,
    val_per_class=100,
    seed=29,
):
    output_dir = Path(output_dir)
    download_root = Path(download_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_source = datasets.CIFAR10(root=str(download_root), train=True, download=True, transform=transform)
    val_source = datasets.CIFAR10(root=str(download_root), train=False, download=True, transform=transform)
    rng = np.random.default_rng(seed)

    train_images, train_labels = _select_binary_subset(train_source, train_per_class, rng, shuffle=True)
    val_images, val_labels = _select_binary_subset(val_source, val_per_class, rng, shuffle=False)

    np.savez_compressed(output_dir / "train_subset.npz", images=train_images, labels=train_labels)
    np.savez_compressed(output_dir / "validation_subset.npz", images=val_images, labels=val_labels)
    return output_dir


def _select_binary_subset(source, per_class, rng, shuffle):
    drone_indices = [idx for idx, (_, label) in enumerate(source) if label == DRONE_PROXY_CLASS]
    no_drone_indices = [idx for idx, (_, label) in enumerate(source) if label in NO_DRONE_PROXY_CLASSES]

    selected_drone = rng.choice(drone_indices, size=per_class, replace=False)
    selected_no_drone = rng.choice(no_drone_indices, size=per_class, replace=False)

    images = []
    labels = []
    for idx in selected_drone:
        image, _ = source[int(idx)]
        images.append(image.numpy())
        labels.append(1)
    for idx in selected_no_drone:
        image, _ = source[int(idx)]
        images.append(image.numpy())
        labels.append(0)

    images = np.stack(images).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    if shuffle:
        order = rng.permutation(len(labels))
        images = images[order]
        labels = labels[order]
    return images, labels


def load_npz_dataset(path):
    data = np.load(path)
    return data["images"].astype(np.float32), data["labels"].astype(np.int64)


def train_or_load_model(model_path, train_images, train_labels, device="cpu", epochs=3):
    model_path = Path(model_path)
    model = build_resnet18_drone_detector().to(device)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    dataset = NumpyImageDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return model


def predict(model, images, device="cpu", batch_size=128):
    model.eval()
    predictions = []
    confidences = []
    probabilities = []
    dataset = NumpyImageDataset(images, np.zeros(len(images), dtype=np.int64))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            predictions.append(pred.cpu().numpy())
            confidences.append(confidence.cpu().numpy())
            probabilities.append(probs.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(confidences), np.concatenate(probabilities)


def evaluate_detection_metrics(model, images, labels, device="cpu"):
    predictions, confidences, probabilities = predict(model, images, device=device)
    drone_mask = labels == 1
    no_drone_mask = labels == 0
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "drone_recall": float(np.mean(predictions[drone_mask] == 1)),
        "false_negative_rate": float(np.mean(predictions[drone_mask] == 0)),
        "no_drone_specificity": float(np.mean(predictions[no_drone_mask] == 0)),
        "mean_confidence": float(np.mean(confidences)),
        "predictions": predictions,
        "confidence": confidences,
        "probabilities": probabilities,
    }


def require_counterfit():
    try:
        import counterfit
        from counterfit import Counterfit
    except ImportError as error:
        raise RuntimeError(
            "Microsoft Counterfit is required for this exercise. Install the official "
            "Azure Counterfit package from https://github.com/Azure/counterfit in the "
            "course environment, or use a supported Python 3.8 Linux/WSL environment."
        ) from error
    return counterfit, Counterfit


def run_counterfit_attack_plan(target, attack_plan):
    _, Counterfit = require_counterfit()
    rows = []
    sample_index = attack_plan.get("sample_index", 0)

    for attack_config in attack_plan["attacks"]:
        attack_name = attack_config["name"]
        cfattack = Counterfit.build_attack(target, attack_name)
        if not cfattack:
            raise RuntimeError(f"Counterfit could not build attack: {attack_name}")

        options = {"sample_index": sample_index}
        options.update(attack_config.get("parameters", {}))
        cfattack.options.update(options)

        run_ok = Counterfit.run_attack(cfattack)
        if not run_ok:
            raise RuntimeError(f"Counterfit failed while running attack: {attack_name}")

        rows.append(summarize_counterfit_attack(cfattack, target, sample_index))

    return rows


def summarize_counterfit_attack(cfattack, target, sample_index):
    sample_indexes = normalize_sample_index(sample_index)
    true_labels = [DRONE_CLASSES[int(target.y[idx])] for idx in sample_indexes]
    initial_labels = list(cfattack.initial_labels or [])
    final_labels = list(cfattack.final_labels or [])
    success_values = cfattack.success

    if isinstance(success_values, (list, tuple, np.ndarray)):
        success_rate = float(np.mean(success_values))
        success_count = int(np.sum(success_values))
        total = len(success_values)
    else:
        success_rate = float(bool(success_values))
        success_count = int(bool(success_values))
        total = 1

    adversarial_accuracy = _label_accuracy(final_labels, true_labels)
    confidence_drop = _mean_confidence_drop(
        getattr(cfattack, "initial_outputs", None),
        getattr(cfattack, "final_outputs", None),
    )
    false_negative_rate = _drone_false_negative_rate(final_labels, true_labels)

    return {
        "attack": cfattack.name.split(".")[-1],
        "attack_id": cfattack.attack_id,
        "sample_count": total,
        "success_count": success_count,
        "attack_success_rate": round(success_rate, 4),
        "adversarial_accuracy": round(adversarial_accuracy, 4),
        "mean_confidence_drop": round(confidence_drop, 4),
        "drone_false_negative_rate": round(false_negative_rate, 4),
        "true_labels": "|".join(map(str, true_labels)),
        "initial_labels": "|".join(map(str, initial_labels)),
        "final_labels": "|".join(map(str, final_labels)),
        "elapsed_time_sec": round(float(cfattack.elapsed_time or 0.0), 4),
        "queries": getattr(cfattack.logger, "num_queries", ""),
        "operational_risk": classify_operational_risk(false_negative_rate, success_rate),
    }


def normalize_sample_index(sample_index):
    if isinstance(sample_index, range):
        return list(sample_index)
    if isinstance(sample_index, (list, tuple, np.ndarray)):
        return [int(index) for index in sample_index]
    return [int(sample_index)]


def _label_accuracy(predicted_labels, true_labels):
    if not predicted_labels:
        return 0.0
    total = min(len(predicted_labels), len(true_labels))
    return float(np.mean([str(predicted_labels[i]) == str(true_labels[i]) for i in range(total)]))


def _drone_false_negative_rate(final_labels, true_labels):
    if not final_labels:
        return 0.0
    drone_positions = [idx for idx, label in enumerate(true_labels) if label == "drone"]
    misses = [str(final_labels[idx]) != "drone" for idx in drone_positions if idx < len(final_labels)]
    return float(np.mean(misses)) if misses else 0.0


def _mean_confidence_drop(initial_outputs, final_outputs):
    try:
        initial = np.max(np.array(initial_outputs, dtype=np.float32), axis=1)
        final = np.max(np.array(final_outputs, dtype=np.float32), axis=1)
        return float(np.mean(initial - final))
    except Exception:
        return 0.0


def classify_operational_risk(false_negative_rate, success_rate):
    if false_negative_rate >= 0.50:
        return "high"
    if false_negative_rate >= 0.20 or success_rate >= 0.40:
        return "medium"
    return "low"


def write_results_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "attack",
        "attack_id",
        "sample_count",
        "success_count",
        "attack_success_rate",
        "adversarial_accuracy",
        "mean_confidence_drop",
        "drone_false_negative_rate",
        "true_labels",
        "initial_labels",
        "final_labels",
        "elapsed_time_sec",
        "queries",
        "operational_risk",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in header))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
