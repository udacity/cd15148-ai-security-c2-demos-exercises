from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms


CLASS_NAMES = ["background", "airplane"]
CIFAR10_AIRPLANE_CLASS = 0
IMAGE_SHAPE = (3, 32, 32)


class AerialObjectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class NumpyImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.images[index]), torch.tensor(self.labels[index])


def _balanced_binary_arrays(cifar_dataset, positive_count, negative_count, seed):
    rng = np.random.default_rng(seed)
    images = []
    labels = []

    positive_indices = [idx for idx, (_, label) in enumerate(cifar_dataset) if label == CIFAR10_AIRPLANE_CLASS]
    negative_indices = [idx for idx, (_, label) in enumerate(cifar_dataset) if label != CIFAR10_AIRPLANE_CLASS]

    positive_indices = rng.choice(positive_indices, size=positive_count, replace=False)
    negative_indices = rng.choice(negative_indices, size=negative_count, replace=False)
    selected = [(idx, 1) for idx in positive_indices] + [(idx, 0) for idx in negative_indices]
    rng.shuffle(selected)

    for idx, binary_label in selected:
        image, _ = cifar_dataset[int(idx)]
        images.append(image.numpy())
        labels.append(binary_label)

    return np.stack(images), np.array(labels)


def prepare_cifar10_airplane_dataset(output_dir, download_root, train_count_per_class=2500, val_count_per_class=500, seed=7):
    output_dir = Path(output_dir)
    download_root = Path(download_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_source = datasets.CIFAR10(root=str(download_root), train=True, download=True, transform=transform)
    val_source = datasets.CIFAR10(root=str(download_root), train=False, download=True, transform=transform)

    train_images, train_labels = _balanced_binary_arrays(
        train_source,
        positive_count=train_count_per_class,
        negative_count=train_count_per_class,
        seed=seed,
    )
    val_images, val_labels = _balanced_binary_arrays(
        val_source,
        positive_count=val_count_per_class,
        negative_count=val_count_per_class,
        seed=seed + 1,
    )

    np.savez_compressed(output_dir / "train.npz", images=train_images, labels=train_labels)
    np.savez_compressed(output_dir / "val.npz", images=val_images, labels=val_labels)
    return output_dir


def load_aerial_dataset(data_dir, split="val"):
    path = Path(data_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset split: {path}. Run scripts/prepare_airplane_assets.py first.")
    data = np.load(path)
    return NumpyImageDataset(data["images"], data["labels"])


def train_or_load_model(model_path, data_dir, device="cpu", epochs=5, force_train=False):
    model_path = Path(model_path)
    model = AerialObjectCNN().to(device)

    if model_path.exists() and not force_train:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    train_dataset = load_aerial_dataset(data_dir, split="train")
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
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

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    model.eval()
    return model


def predict(model, x_np, device="cpu"):
    model.eval()
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)
    return pred.cpu().numpy(), confidence.cpu().numpy(), probs.cpu().numpy()


def evaluate_numpy(model, images, labels, device="cpu"):
    pred, conf, _ = predict(model, images, device=device)
    accuracy = float(np.mean(pred == labels))
    aerial_mask = labels == 1
    false_negative_rate = float(np.mean(pred[aerial_mask] == 0)) if np.any(aerial_mask) else 0.0
    return {
        "accuracy": accuracy,
        "mean_confidence": float(np.mean(conf)),
        "false_negative_rate": false_negative_rate,
        "predictions": pred,
        "confidence": conf,
    }


def perturbation_metrics(clean, adversarial):
    delta = adversarial - clean
    return {
        "linf": float(np.max(np.abs(delta))),
        "l2_mean": float(np.mean(np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=1))),
        "mean_abs": float(np.mean(np.abs(delta))),
    }


def build_art_classifier(model):
    from art.estimators.classification import PyTorchClassifier

    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=IMAGE_SHAPE,
        nb_classes=len(CLASS_NAMES),
        clip_values=(0.0, 1.0),
    )


def make_attack(attack_name, classifier, epsilon):
    attack_name = attack_name.lower()
    if attack_name == "fgsm":
        from art.attacks.evasion import FastGradientMethod

        return FastGradientMethod(estimator=classifier, eps=epsilon)
    if attack_name == "pgd":
        from art.attacks.evasion import ProjectedGradientDescent

        return ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=epsilon / 4, max_iter=20)
    if attack_name == "bim":
        from art.attacks.evasion import BasicIterativeMethod

        return BasicIterativeMethod(estimator=classifier, eps=epsilon, eps_step=epsilon / 4, max_iter=20)
    if attack_name == "deepfool":
        from art.attacks.evasion import DeepFool

        return DeepFool(classifier=classifier, max_iter=20, epsilon=epsilon)
    if attack_name == "simba":
        from art.attacks.evasion import SimBA

        return SimBA(classifier=classifier, attack="px", max_iter=100, epsilon=epsilon, batch_size=16)
    raise ValueError(f"Unsupported attack: {attack_name}")


def attack_success_rate(clean_predictions, adv_predictions, labels):
    clean_correct = clean_predictions == labels
    if not np.any(clean_correct):
        return 0.0
    return float(np.mean(adv_predictions[clean_correct] != labels[clean_correct]))


def plot_comparison_grid(clean_images, adv_images, clean_preds, adv_preds, labels, output_path, max_items=6):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(max_items, len(clean_images))
    fig, axes = plt.subplots(2, count, figsize=(2.2 * count, 4.4))

    for idx in range(count):
        clean = np.transpose(np.clip(clean_images[idx], 0, 1), (1, 2, 0))
        adv = np.transpose(np.clip(adv_images[idx], 0, 1), (1, 2, 0))

        axes[0, idx].imshow(clean)
        axes[0, idx].set_title(f"Clean\n{CLASS_NAMES[int(clean_preds[idx])]}\ntrue {CLASS_NAMES[int(labels[idx])]}")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(adv)
        axes[1, idx].set_title(f"Adversarial\n{CLASS_NAMES[int(adv_preds[idx])]}")
        axes[1, idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
    return output_path


def save_metrics_table(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "attack",
        "epsilon",
        "clean_accuracy",
        "adversarial_accuracy",
        "attack_success_rate",
        "mean_clean_confidence",
        "mean_adversarial_confidence",
        "mean_linf",
        "mean_l2",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in header))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
