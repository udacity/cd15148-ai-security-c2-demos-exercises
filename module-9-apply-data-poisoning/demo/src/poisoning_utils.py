from pathlib import Path

import matplotlib.pyplot as plt
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

IMAGE_SHAPE = (3, 32, 32)


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


def prepare_cifar10_subsets(output_dir, download_root, train_per_class=500, val_per_class=100, seed=9):
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
        return np.stack(images)[order], np.array(labels)[order]

    train_images, train_labels = select_balanced(train_source, train_per_class)
    val_images, val_labels = select_balanced(val_source, val_per_class)

    np.savez_compressed(output_dir / "train_clean.npz", images=train_images, labels=train_labels)
    np.savez_compressed(output_dir / "val_clean.npz", images=val_images, labels=val_labels)
    return output_dir


def load_subset(data_dir, name):
    path = Path(data_dir) / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    data = np.load(path)
    return data["images"].astype(np.float32), data["labels"].astype(np.int64)


def add_square_trigger(images, size=4, value=1.0):
    poisoned = images.copy()
    poisoned[:, :, -size:, -size:] = value
    return poisoned


def create_backdoor_poisoned_training_set(
    images,
    labels,
    source_class=3,
    target_class=9,
    poison_fraction=0.08,
    trigger_size=4,
    seed=11,
):
    rng = np.random.default_rng(seed)
    poisoned_images = images.copy()
    poisoned_labels = labels.copy()

    source_indices = np.where(labels == source_class)[0]
    poison_count = max(1, int(len(images) * poison_fraction))
    poison_count = min(poison_count, len(source_indices))
    selected = rng.choice(source_indices, size=poison_count, replace=False)

    poisoned_images[selected] = add_square_trigger(poisoned_images[selected], size=trigger_size)
    poisoned_labels[selected] = target_class
    return poisoned_images, poisoned_labels, selected


def train_model(model, train_images, train_labels, device="cpu", epochs=2, batch_size=128, lr=0.001):
    dataset = NumpyImageDataset(train_images, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
    return model


def predict(model, images, device="cpu", batch_size=256):
    model.eval()
    predictions = []
    confidences = []
    probs_out = []
    dataset = NumpyImageDataset(images, np.zeros(len(images)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            predictions.append(pred.cpu().numpy())
            confidences.append(confidence.cpu().numpy())
            probs_out.append(probs.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(confidences), np.concatenate(probs_out)


def evaluate_clean(model, images, labels, device="cpu"):
    pred, conf, _ = predict(model, images, device=device)
    return {
        "accuracy": float(np.mean(pred == labels)),
        "mean_confidence": float(np.mean(conf)),
        "predictions": pred,
        "confidence": conf,
    }


def targeted_backdoor_success(model, images, labels, source_class, target_class, trigger_size=4, device="cpu"):
    source_mask = labels == source_class
    source_images = images[source_mask]
    triggered = add_square_trigger(source_images, size=trigger_size)
    pred, conf, _ = predict(model, triggered, device=device)
    return {
        "attack_success_rate": float(np.mean(pred == target_class)),
        "mean_triggered_confidence": float(np.mean(conf)),
        "triggered_images": triggered,
        "predictions": pred,
        "confidence": conf,
    }


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(path, device="cpu"):
    model = build_resnet18_cifar10().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def plot_poison_examples(clean_images, poisoned_images, output_path, count=6):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(count, len(clean_images), len(poisoned_images))
    fig, axes = plt.subplots(2, count, figsize=(2.1 * count, 4.2))

    for idx in range(count):
        clean = np.transpose(np.clip(clean_images[idx], 0, 1), (1, 2, 0))
        poisoned = np.transpose(np.clip(poisoned_images[idx], 0, 1), (1, 2, 0))
        axes[0, idx].imshow(clean)
        axes[0, idx].set_title("Clean")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(poisoned)
        axes[1, idx].set_title("Triggered")
        axes[1, idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
    return output_path


def write_results_table(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", "clean_accuracy", "targeted_attack_success_rate", "mean_clean_confidence", "notes"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in header))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
