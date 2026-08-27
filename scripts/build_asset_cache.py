#!/usr/bin/env python3
"""Populate the shared external-asset cache used by the GPU-image modules.

Four modules (M7, M9, M15, M19) download datasets and model weights from the
public internet on first run. Between them they fetch only six distinct upstream
artifacts, but they fetch them into thirteen separate module directories -- about
2.4 GB of downloads and 2.8 GB on disk for 864 MB of unique content.

This script downloads each artifact exactly once into a single cache directory
laid out so that every module's loader finds it there. Point the workspace
sidecar at that directory and export ``C2_ASSET_CACHE`` in the image; the module
code reads that variable and falls back to its own ``data/`` folder when it is
unset, so a plain ``git clone`` still works unchanged.

Cache layout::

    $C2_ASSET_CACHE/
    |-- torchvision/                    torchvision `root=` for both datasets
    |   |-- cifar-10-batches-py/            CIFAR-10 (M7 exercise, M9 demo, M19 demo)
    |   `-- gtsrb/                          GTSRB    (M7 demo, M9 exercise, M19 exercise)
    |-- hf/gtsrb_vit/                   ViT traffic-sign classifier (M7 demo)
    |-- sklearn/                        fetch_olivetti_faces `data_home=` (M15 demo)
    `-- ultralytics/brain-tumor/        brain tumour MRI subset (M15 exercise)

Usage::

    python scripts/build_asset_cache.py --cache-dir /path/to/sidecar
    python scripts/build_asset_cache.py --cache-dir /path/to/sidecar --verify-only

Needs the GPU image's environment (torch, torchvision, scikit-learn) for the
build; ``--verify-only`` is filesystem checks and runs anywhere.

Every step is idempotent (STYLE-007): a second run re-verifies and downloads
nothing. Archives are deleted after extraction because no loader in the repo
checks for them -- ``CIFAR10._check_integrity`` md5-checks the extracted batch
files and ``GTSRB._check_exists`` checks the extracted directory. Pass
``--keep-archives`` to retain them anyway (+435 MB).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# --- Cache layout -----------------------------------------------------------

TORCHVISION_DIR = "torchvision"
GTSRB_VIT_DIR = "hf/gtsrb_vit"
SKLEARN_DIR = "sklearn"
BRAIN_TUMOR_DIR = "ultralytics"

# --- Upstream sources (kept in sync with the module code that consumes them) --

GTSRB_VIT_REPO = "kelvinandreas/vit-traffic-sign-GTSRB"
GTSRB_VIT_BASE_URL = f"https://huggingface.co/{GTSRB_VIT_REPO}/resolve/main"
GTSRB_VIT_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")

BRAIN_TUMOR_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip"

# Archives torchvision and the M15 loader leave behind after extraction. Nothing
# in the repo reads them, so they are pruned unless --keep-archives is passed.
PRUNABLE_ARCHIVES = (
    f"{TORCHVISION_DIR}/cifar-10-python.tar.gz",
    f"{TORCHVISION_DIR}/gtsrb/GTSRB-Training_fixed.zip",
    f"{TORCHVISION_DIR}/gtsrb/GTSRB_Final_Test_Images.zip",
    f"{TORCHVISION_DIR}/gtsrb/GTSRB_Final_Test_GT.zip",
    f"{BRAIN_TUMOR_DIR}/brain-tumor.zip",
)

# Paths each module's loader actually probes, checked by --verify-only, with a
# floor on the size measured from a real build. Existence alone is not enough:
# a half-synced sidecar mount presents the right paths at the wrong sizes, and a
# false PASS here means a classroom download. Keep this list aligned with the
# loaders.
EXPECTED = (
    # CIFAR10._check_integrity() md5-checks these six pickles.
    (f"{TORCHVISION_DIR}/cifar-10-batches-py/data_batch_1", "file", "CIFAR-10 train batch 1", 25000000),
    (f"{TORCHVISION_DIR}/cifar-10-batches-py/data_batch_5", "file", "CIFAR-10 train batch 5", 25000000),
    (f"{TORCHVISION_DIR}/cifar-10-batches-py/test_batch", "file", "CIFAR-10 test batch", 25000000),
    (f"{TORCHVISION_DIR}/cifar-10-batches-py/batches.meta", "file", "CIFAR-10 label names", 100),
    # GTSRB._check_exists() checks these directories; __init__ reads the CSV directly.
    (f"{TORCHVISION_DIR}/gtsrb/GTSRB/Training", "dir", "GTSRB train images", 200000000),
    (f"{TORCHVISION_DIR}/gtsrb/GTSRB/Final_Test/Images", "dir", "GTSRB test images", 100000000),
    (f"{TORCHVISION_DIR}/gtsrb/GT-final_test.csv", "file", "GTSRB test ground truth", 300000),
    # load_gtsrb_vit() requires all three before calling from_pretrained().
    (f"{GTSRB_VIT_DIR}/config.json", "file", "ViT config", 1000),
    (f"{GTSRB_VIT_DIR}/preprocessor_config.json", "file", "ViT preprocessor config", 200),
    (f"{GTSRB_VIT_DIR}/model.safetensors", "file", "ViT weights", 300000000),
    # sklearn's _pkl_filepath() inserts the _py3 suffix; the raw .mat is deleted
    # by fetch_olivetti_faces itself.
    (f"{SKLEARN_DIR}/olivetti_py3.pkz", "file", "Olivetti faces", 1000000),
    # _download_or_find_brain_tumor_dataset() short-circuits on images/train.
    (f"{BRAIN_TUMOR_DIR}/brain-tumor/images/train", "dir", "Brain tumour train images", 3000000),
    (f"{BRAIN_TUMOR_DIR}/brain-tumor/images/val", "dir", "Brain tumour val images", 700000),
    (f"{BRAIN_TUMOR_DIR}/brain-tumor/labels/train", "dir", "Brain tumour train labels", 25000),
    (f"{BRAIN_TUMOR_DIR}/brain-tumor/labels/val", "dir", "Brain tumour val labels", 6000),
)


def log(message: str) -> None:
    print(message, flush=True)


def human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def download(url: str, destination: Path, *, retries: int = 3) -> None:
    """Download `url` to `destination` via a temp file, so a partial transfer
    never looks like a completed one to the next run."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, temp_path)
            temp_path.replace(destination)
            return
        except Exception as error:  # network flake, HTTP error, truncated read
            last_error = error
            temp_path.unlink(missing_ok=True)
            log(f"    attempt {attempt}/{retries} failed: {error}")
    raise RuntimeError(f"Could not download {url}") from last_error


# --- Per-artifact cache steps ----------------------------------------------


def cache_torchvision_datasets(cache_dir: Path) -> None:
    """CIFAR-10 and GTSRB share one torchvision `root`: CIFAR-10 extracts to
    `root/cifar-10-batches-py`, GTSRB to `root/gtsrb`."""
    from torchvision import datasets

    root = cache_dir / TORCHVISION_DIR
    root.mkdir(parents=True, exist_ok=True)

    log("[1/5] CIFAR-10 -> torchvision/cifar-10-batches-py")
    datasets.CIFAR10(root=str(root), train=True, download=True)
    datasets.CIFAR10(root=str(root), train=False, download=True)

    log("[2/5] GTSRB -> torchvision/gtsrb (scanning ~39k images, takes a minute)")
    datasets.GTSRB(root=str(root), split="train", download=True)
    datasets.GTSRB(root=str(root), split="test", download=True)


def cache_gtsrb_vit(cache_dir: Path) -> None:
    """Mirror of module-7's download_gtsrb_model.sh, which curls these three
    files rather than going through huggingface_hub."""
    log(f"[3/5] {GTSRB_VIT_REPO} -> {GTSRB_VIT_DIR}")
    model_dir = cache_dir / GTSRB_VIT_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in GTSRB_VIT_FILES:
        target = model_dir / filename
        if target.exists():
            log(f"    {filename} already cached ({human(target.stat().st_size)})")
            continue
        log(f"    downloading {filename}")
        download(f"{GTSRB_VIT_BASE_URL}/{filename}", target)


def cache_sklearn_faces(cache_dir: Path) -> None:
    from sklearn.datasets import fetch_olivetti_faces

    log(f"[4/5] Olivetti faces -> {SKLEARN_DIR}/olivetti_py3.pkz")
    data_home = cache_dir / SKLEARN_DIR
    data_home.mkdir(parents=True, exist_ok=True)
    fetch_olivetti_faces(data_home=str(data_home), shuffle=False, download_if_missing=True)


def cache_brain_tumor(cache_dir: Path) -> None:
    log(f"[5/5] brain-tumor.zip -> {BRAIN_TUMOR_DIR}/brain-tumor/")
    target_dir = cache_dir / BRAIN_TUMOR_DIR
    # The archive has no top-level folder of its own -- it holds images/, labels/,
    # brain-tumor.yaml and LICENSE.txt at the root -- so extract into a named
    # subdirectory rather than scattering those four entries across the cache.
    dataset_dir = target_dir / "brain-tumor"
    target_dir.mkdir(parents=True, exist_ok=True)
    if (dataset_dir / "images" / "train").exists():
        log("    already extracted")
        return
    zip_path = target_dir / "brain-tumor.zip"
    if not zip_path.exists():
        download(BRAIN_TUMOR_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(dataset_dir)


def prune_archives(cache_dir: Path) -> None:
    reclaimed = 0
    for relative in PRUNABLE_ARCHIVES:
        archive = cache_dir / relative
        if archive.exists():
            reclaimed += archive.stat().st_size
            archive.unlink()
            log(f"    removed {relative}")
    if reclaimed:
        log(f"    reclaimed {human(reclaimed)}")


def check_no_sidecars(cache_dir: Path) -> bool:
    """Fail on macOS metadata sidecars anywhere in the cache.

    A tree copied or tarred through macOS onto another filesystem gains an
    AppleDouble companion (``._<name>``) beside every real file. The companion
    carries the same extension as the file it shadows, so every loader that
    discovers inputs by globbing an extension picks it up as data: torchvision's
    GTSRB `split="train"` calls make_dataset(extensions=(".ppm",)) and hands PIL a
    ``._00000_00000.ppm``, and the M15 brain-tumor loader globs ``*.jpg`` and then
    reads the matching ``._*.txt`` label as UTF-8. Loaders that read a manifest or
    a named file instead -- CIFAR's pickles, GTSRB's test CSV, Olivetti's .pkz --
    never see them, so contamination takes out some modules and not others.

    The EXPECTED checks above cannot catch this: sidecars *inflate* the directory
    totals, so junk makes the size floors easier to clear, not harder. A cache in
    this state passes every check above and still fails 3 of the 8 notebooks.
    """
    patterns = ("._*", ".DS_Store", "__MACOSX")
    found = [p for pattern in patterns for p in cache_dir.rglob(pattern)]
    if not found:
        return True
    log(f"\n  FAIL  {len(found)} macOS metadata file(s) in the cache, e.g. {found[0].relative_to(cache_dir)}")
    log("        Glob-based loaders will read these as data. Delete them where the")
    log("        volume is writable (the classroom mount is read-only):")
    log(f"          find {cache_dir} \\( -name '._*' -o -name '.DS_Store' \\) -delete")
    log("        Better: rebuild the cache on a Linux host so the tree never")
    log("        transits macOS. If it must be copied from a Mac, use")
    log("        `rsync -a --exclude='._*'` or `COPYFILE_DISABLE=1 tar`.")
    return False


def verify(cache_dir: Path) -> bool:
    log(f"\nVerifying cache at {cache_dir}")
    ok = True
    for relative, kind, label, minimum in EXPECTED:
        path = cache_dir / relative
        present = path.is_dir() if kind == "dir" else path.is_file()
        actual = tree_size(path) if present else 0
        good = present and actual >= minimum
        size = f"{human(actual):>10}" if present else " " * 10
        if not present:
            note = "  missing"
        elif not good:
            note = f"  only {human(actual)}, expected >= {human(minimum)}"
        else:
            note = ""
        log(f"  {'PASS' if good else 'FAIL'}  {size}  {label:<28} {relative}{note}")
        ok = ok and good
    ok = check_no_sidecars(cache_dir) and ok
    log(f"\n  total: {human(tree_size(cache_dir))}")
    log(f"  status: {'cache is complete' if ok else 'CACHE IS INCOMPLETE -- rerun without --verify-only'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download every external dataset and model weight the GPU-image modules need.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=os.environ.get("C2_ASSET_CACHE"),
        help="Cache directory to populate. Defaults to $C2_ASSET_CACHE.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check an existing cache without downloading anything.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the source archives after extraction (+435 MB; nothing in the repo reads them).",
    )
    args = parser.parse_args()

    if args.cache_dir is None:
        parser.error("pass --cache-dir or set C2_ASSET_CACHE")

    cache_dir = args.cache_dir.expanduser().resolve()

    if args.verify_only:
        return 0 if verify(cache_dir) else 1

    log(f"Populating asset cache at {cache_dir}")
    free_bytes = shutil.disk_usage(cache_dir.parent if not cache_dir.exists() else cache_dir).free
    log(f"Free space: {human(free_bytes)} (needs ~1.3 GB during build, ~865 MB after)\n")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_torchvision_datasets(cache_dir)
    cache_gtsrb_vit(cache_dir)
    cache_sklearn_faces(cache_dir)
    cache_brain_tumor(cache_dir)

    if args.keep_archives:
        log("\nKeeping source archives (--keep-archives)")
    else:
        log("\nPruning source archives")
        prune_archives(cache_dir)

    if not verify(cache_dir):
        return 1
    log(f"\nExport this in the workspace image:\n  C2_ASSET_CACHE={cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
