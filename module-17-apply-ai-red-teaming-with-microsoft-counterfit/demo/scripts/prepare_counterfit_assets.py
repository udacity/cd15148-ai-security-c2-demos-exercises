from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from counterfit_demo_utils import load_npz_dataset, prepare_cifar10_assets, train_or_load_model  # noqa: E402


def main():
    data_dir = ROOT / "data" / "generated"
    download_dir = ROOT / "data" / "cifar10"
    model_path = ROOT / "models" / "cifar10_resnet18_demo.pt"

    prepare_cifar10_assets(data_dir, download_dir)
    train_images, train_labels = load_npz_dataset(data_dir / "train_subset.npz")
    train_or_load_model(model_path, train_images, train_labels)
    print(f"Prepared data under {data_dir}")
    print(f"Prepared model checkpoint at {model_path}")


if __name__ == "__main__":
    main()
