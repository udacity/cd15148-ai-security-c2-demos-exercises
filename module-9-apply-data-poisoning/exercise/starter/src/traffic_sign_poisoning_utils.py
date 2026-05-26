from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
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
COMPACT_TO_ORIGINAL = {compact: original for original, compact in ORIGINAL_TO_COMPACT.items()}

SOURCE_CLASS_NAME = "Stop"
TARGET_CLASS_NAME = "Yield"
SOURCE_CLASS_ID = SELECTED_CLASS_NAMES.index(SOURCE_CLASS_NAME)
TARGET_CLASS_ID = SELECTED_CLASS_NAMES.index(TARGET_CLASS_NAME)


class NumpyImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.images[index]), torch.tensor(self.labels[index])


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=len(SELECTED_CLASS_NAMES)):
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

    def forward(self, x):
        return self.classifier(self.features(x))


def build_traffic_sign_cnn():
    return TrafficSignCNN()


def prepare_gtsrb_subsets(output_dir, download_root, train_per_class=80, val_per_class=30, image_size=64, seed=19):
    output_dir = Path(output_dir)
    download_root = Path(download_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    train_source = datasets.GTSRB(root=str(download_root), split="train", download=True, transform=transform)
    val_source = datasets.GTSRB(root=str(download_root), split="test", download=True, transform=transform)
    rng = np.random.default_rng(seed)

    def collect(source, per_class):
        by_class = {compact: [] for compact in range(len(SELECTED_CLASS_NAMES))}
        for image, original_label in source:
            original_label = int(original_label)
            if original_label in ORIGINAL_TO_COMPACT:
                by_class[ORIGINAL_TO_COMPACT[original_label]].append(image.numpy())

        images = []
        labels = []
        for compact_label, class_images in by_class.items():
            if len(class_images) < per_class:
                raise ValueError(
                    f"Class {SELECTED_CLASS_NAMES[compact_label]} only has {len(class_images)} samples; "
                    f"requested {per_class}."
                )
            selected = rng.choice(len(class_images), size=per_class, replace=False)
            for idx in selected:
                images.append(class_images[int(idx)])
                labels.append(compact_label)

        order = rng.permutation(len(labels))
        return np.stack(images)[order], np.array(labels, dtype=np.int64)[order]

    train_images, train_labels = collect(train_source, train_per_class)
    val_images, val_labels = collect(val_source, val_per_class)

    np.savez_compressed(output_dir / "traffic_sign_train_clean.npz", images=train_images, labels=train_labels)
    np.savez_compressed(output_dir / "traffic_sign_val_clean.npz", images=val_images, labels=val_labels)
    return output_dir


def load_subset(data_dir, name):
    path = Path(data_dir) / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    data = np.load(path)
    return data["images"].astype(np.float32), data["labels"].astype(np.int64)


def create_label_flip_poisoned_training_set(
    images,
    labels,
    source_class=SOURCE_CLASS_ID,
    target_class=TARGET_CLASS_ID,
    poison_fraction=0.75,
    seed=23,
):
    """Return a copy of the dataset with a fraction of source labels flipped to target labels."""
    # TODO: Implement the label-flipping workflow.
    # 1. Copy images and labels so the clean dataset remains unchanged.
    # 2. Find indices where labels == source_class.
    # 3. Select int(len(source_indices) * poison_fraction) source-class samples.
    # 4. Change only the selected labels to target_class.
    # 5. Return poisoned_images, poisoned_labels, selected_indices.
    raise NotImplementedError("Complete create_label_flip_poisoned_training_set in the starter utility file.")


def train_model(model, train_images, train_labels, device="cpu", epochs=4, batch_size=64, lr=0.001):
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


def predict(model, images, device="cpu", batch_size=128):
    model.eval()
    predictions = []
    confidences = []
    probs_out = []
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


def class_accuracy_table(predictions, labels):
    rows = []
    for class_id, class_name in enumerate(SELECTED_CLASS_NAMES):
        mask = labels == class_id
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "sample_count": int(np.sum(mask)),
                "accuracy": float(np.mean(predictions[mask] == labels[mask])),
            }
        )
    return rows


def targeted_misclassification_rate(predictions, labels, source_class=SOURCE_CLASS_ID, target_class=TARGET_CLASS_ID):
    """Measure how often clean source-class validation examples are predicted as the target class."""
    # TODO: Implement the targeted misclassification rate.
    # 1. Filter validation examples where labels == source_class.
    # 2. Compute the fraction of those examples where predictions == target_class.
    raise NotImplementedError("Complete targeted_misclassification_rate in the starter utility file.")


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(path, device="cpu"):
    model = build_traffic_sign_cnn().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def plot_label_flip_examples(images, clean_labels, poisoned_labels, selected_indices, output_path, count=8):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shown = selected_indices[: min(count, len(selected_indices))]

    fig, axes = plt.subplots(1, len(shown), figsize=(2.2 * len(shown), 2.8))
    if len(shown) == 1:
        axes = [axes]

    for axis, idx in zip(axes, shown):
        image = np.transpose(np.clip(images[int(idx)], 0, 1), (1, 2, 0))
        axis.imshow(image)
        axis.set_title(
            f"{SELECTED_CLASS_NAMES[int(clean_labels[idx])]}\nlabel -> {SELECTED_CLASS_NAMES[int(poisoned_labels[idx])]}"
        )
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
    return output_path


def write_results_table(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "model",
        "clean_accuracy",
        "source_class_accuracy",
        "targeted_misclassification_rate",
        "mean_confidence",
        "notes",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in header))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
