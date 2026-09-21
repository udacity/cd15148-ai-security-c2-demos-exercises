from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from traffic_sign_poisoning_utils import prepare_gtsrb_subsets  # noqa: E402


def main():
    data_dir = ROOT / "data" / "generated"
    # Shared workspace asset cache (C2_ASSET_CACHE); falls back to this module's data/ folder.
    asset_cache = os.environ.get("C2_ASSET_CACHE")
    download_dir = Path(asset_cache) / "torchvision" if asset_cache else ROOT / "data" / "gtsrb"

    prepare_gtsrb_subsets(data_dir, download_dir, train_per_class=200, val_per_class=100)
    print(f"Prepared compact traffic sign subsets at: {data_dir}")


if __name__ == "__main__":
    main()
