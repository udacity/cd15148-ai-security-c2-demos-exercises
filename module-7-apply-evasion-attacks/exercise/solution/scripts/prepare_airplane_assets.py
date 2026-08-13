from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from aerial_eval_utils import prepare_cifar10_airplane_dataset, train_or_load_model


def main():
    data_dir = ROOT / "data/generated"
    # Shared workspace asset cache (C2_ASSET_CACHE); falls back to this module's data/ folder.
    asset_cache = os.environ.get("C2_ASSET_CACHE")
    download_root = Path(asset_cache) / "torchvision" if asset_cache else ROOT / "data/cifar10"
    model_path = ROOT / "models/aerial_object_cnn.pt"
    prepare_cifar10_airplane_dataset(data_dir, download_root)
    train_or_load_model(model_path, data_dir, device="cpu", epochs=5, force_train=True)
    print(f"Prepared CIFAR-10 airplane dataset at: {data_dir}")
    print(f"Generated model checkpoint at: {model_path}")


if __name__ == "__main__":
    main()
