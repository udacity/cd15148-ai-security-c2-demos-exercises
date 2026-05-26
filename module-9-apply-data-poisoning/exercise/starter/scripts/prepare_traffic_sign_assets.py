from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from traffic_sign_poisoning_utils import (  # noqa: E402
    build_traffic_sign_cnn,
    evaluate_clean,
    prepare_gtsrb_subsets,
    save_checkpoint,
    train_model,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = ROOT / "data" / "generated"
    download_dir = ROOT / "data" / "gtsrb"
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    prepare_gtsrb_subsets(data_dir, download_dir, train_per_class=80, val_per_class=30)
    train = np.load(data_dir / "traffic_sign_train_clean.npz")
    val = np.load(data_dir / "traffic_sign_val_clean.npz")

    model = build_traffic_sign_cnn()
    model = train_model(model, train["images"], train["labels"], device=device, epochs=4)
    save_checkpoint(model, model_dir / "clean_traffic_sign_cnn.pt")

    metrics = evaluate_clean(model, val["images"], val["labels"], device=device)
    print(f"Prepared clean baseline checkpoint. Validation accuracy: {metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
