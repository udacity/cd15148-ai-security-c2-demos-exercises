"""Train the shipped clean baseline ResNet-18 checkpoint with the demo's recipe.

The demo notebook loads `models/baseline_resnet18.pt` by default. Run this once
to regenerate it (or after changing the training recipe). Students do not need
to run this; it is for instructor / maintainer use.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

DEMO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(DEMO_ROOT / "src"))

from poisoning_utils import (  # noqa: E402
    build_resnet18_cifar10,
    evaluate_clean,
    load_subset,
    prepare_cifar10_subsets,
    save_checkpoint,
    train_model,
)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    torch.manual_seed(7)
    np.random.seed(7)

    device = pick_device()
    print(f"device: {device}", flush=True)

    generated_data_dir = DEMO_ROOT / "data" / "generated"
    download_dir = DEMO_ROOT / "data" / "cifar10"
    prepare_cifar10_subsets(generated_data_dir, download_dir, train_per_class=2000, val_per_class=200)

    train_images, train_labels = load_subset(generated_data_dir, "train_clean")
    val_images, val_labels = load_subset(generated_data_dir, "val_clean")
    print(f"train: {train_images.shape} | val: {val_images.shape}", flush=True)

    model = build_resnet18_cifar10()
    start = time.time()
    model = train_model(model, train_images, train_labels, device=device, epochs=15, augment=True)
    elapsed = time.time() - start

    checkpoint_path = DEMO_ROOT / "models" / "baseline_resnet18.pt"
    save_checkpoint(model, checkpoint_path)

    eval_result = evaluate_clean(model, val_images, val_labels)
    print(f"trained in {elapsed:.1f}s", flush=True)
    print(f"clean validation accuracy: {eval_result['accuracy']:.4f}", flush=True)
    print(f"mean confidence:           {eval_result['mean_confidence']:.4f}", flush=True)
    print(f"saved checkpoint to: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
