from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


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

STOP_CLASS_ID = 14


class RawImageViTClassifier(nn.Module):
    """Wrap a Hugging Face ViT so attacks can operate on raw [0, 1] image tensors."""

    def __init__(self, hf_model, image_mean, image_std):
        super().__init__()
        self.hf_model = hf_model
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.image_mean) / self.image_std
        return self.hf_model(pixel_values=x).logits


def load_gtsrb_vit(model_dir, device="cpu"):
    model_dir = Path(model_dir)
    required_files = ["config.json", "model.safetensors", "preprocessor_config.json"]
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing GTSRB model files: "
            + ", ".join(missing)
            + f". Run: bash scripts/download_gtsrb_model.sh"
        )

    processor = ViTImageProcessor.from_pretrained(model_dir)
    hf_model = ViTForImageClassification.from_pretrained(model_dir)
    wrapped_model = RawImageViTClassifier(
        hf_model,
        image_mean=processor.image_mean,
        image_std=processor.image_std,
    )
    wrapped_model.to(device)
    wrapped_model.eval()
    return wrapped_model, processor


def pil_to_raw_array(image, size=224):
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    image = image.convert("RGB").resize((size, size))
    array = np.asarray(image).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def predict_numpy(model, images):
    """Run forward inference on a numpy batch.

    Builds the input tensor on the model's current device (queried via
    next(model.parameters()).device) rather than accepting a caller-passed
    device. ART's PyTorchClassifier has no MPS code path and silently moves
    wrapped models to CPU on non-CUDA setups; trusting the model's actual
    device makes this helper self-healing across ART attack calls.
    """
    model.eval()
    model_device = next(model.parameters()).device
    tensor = torch.tensor(images, dtype=torch.float32, device=model_device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
    confidence, label_idx = torch.max(probs, dim=1)
    return label_idx.cpu().numpy(), confidence.cpu().numpy(), probs.cpu().numpy()


def get_first_class_image(dataset, class_id):
    for image, label in dataset:
        if int(label) == int(class_id):
            return image.convert("RGB"), int(label)
    raise ValueError(f"Could not find class id {class_id} in dataset.")


def perturbation_metrics(clean, adversarial):
    delta = adversarial - clean
    return {
        "linf": float(np.max(np.abs(delta))),
        "l2": float(np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=1)[0]),
        "mean_abs": float(np.mean(np.abs(delta))),
    }


def plot_attack_comparison(clean, simba_adv, pgd_adv, labels, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = [clean[0], simba_adv[0], pgd_adv[0]]
    titles = ["Clean stop sign", "SimBA adversarial", "PGD adversarial"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for axis, image, title, label in zip(axes, images, titles, labels):
        axis.imshow(np.transpose(np.clip(image, 0, 1), (1, 2, 0)))
        axis.set_title(f"{title}\n{label}")
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
    return output_path
