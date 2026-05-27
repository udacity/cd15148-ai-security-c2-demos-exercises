from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
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


class NumpyImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.images[index]), torch.tensor(self.labels[index])


def build_resnet18_cifar10():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, len(CIFAR10_CLASSES))
    return model


def prepare_cifar10_assets(output_dir, download_root, train_per_class=150, val_per_class=50, seed=17):
    output_dir = Path(output_dir)
    download_root = Path(download_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_source = datasets.CIFAR10(root=str(download_root), train=True, download=True, transform=transform)
    val_source = datasets.CIFAR10(root=str(download_root), train=False, download=True, transform=transform)
    rng = np.random.default_rng(seed)

    def select_balanced(source, per_class):
        images = []
        labels = []
        for class_id in range(len(CIFAR10_CLASSES)):
            indices = [idx for idx, (_, label) in enumerate(source) if label == class_id]
            selected = rng.choice(indices, size=per_class, replace=False)
            for idx in selected:
                image, label = source[int(idx)]
                images.append(image.numpy())
                labels.append(label)
        order = rng.permutation(len(labels))
        return np.stack(images)[order], np.array(labels, dtype=np.int64)[order]

    train_images, train_labels = select_balanced(train_source, train_per_class)
    val_images, val_labels = select_balanced(val_source, val_per_class)

    np.savez_compressed(output_dir / "train_subset.npz", images=train_images, labels=train_labels)
    np.savez_compressed(output_dir / "validation_500.npz", images=val_images, labels=val_labels)
    return output_dir


def load_npz_dataset(path):
    data = np.load(path)
    return data["images"].astype(np.float32), data["labels"].astype(np.int64)


def train_or_load_model(model_path, train_images, train_labels, device="cpu", epochs=2):
    model_path = Path(model_path)
    model = build_resnet18_cifar10().to(device)
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


def evaluate_accuracy(model, images, labels, device="cpu"):
    predictions, confidences, _ = predict(model, images, device=device)
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "mean_confidence": float(np.mean(confidences)),
        "predictions": predictions,
        "confidence": confidences,
    }


def require_counterfit():
    try:
        import counterfit
        from counterfit import Counterfit
    except ImportError as error:
        raise RuntimeError(
            "Microsoft Counterfit is required for this demo. Install it from "
            "https://github.com/Azure/counterfit in a supported Python 3.8 Linux/WSL environment."
        ) from error
    return counterfit, Counterfit


def run_counterfit_attack_plan(target, attack_plan, sample_index=0):
    _, Counterfit = require_counterfit()
    rows = []

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

        rows.append(summarize_counterfit_attack(cfattack))

    return rows


def summarize_counterfit_attack(cfattack):
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

    return {
        "attack": cfattack.name.split(".")[-1],
        "attack_id": cfattack.attack_id,
        "sample_count": total,
        "success_count": success_count,
        "attack_success_rate": round(success_rate, 4),
        "initial_labels": "|".join(map(str, initial_labels)),
        "final_labels": "|".join(map(str, final_labels)),
        "elapsed_time_sec": round(float(cfattack.elapsed_time or 0.0), 4),
        "queries": getattr(cfattack.logger, "num_queries", ""),
        "operational_risk": classify_operational_risk(success_rate),
    }


def classify_operational_risk(success_rate):
    if success_rate >= 0.40:
        return "high"
    if success_rate >= 0.15:
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
