from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
os.environ.setdefault("ART_DATA_PATH", str(ROOT / "art_data"))

from robustness_eval_utils import (  # noqa: E402
    ModelSpec,
    load_npz_dataset,
    plot_example_grid,
    plot_scorecard,
    prepare_cifar10_assets,
    robustness_scorecard,
    run_art_attack_suite,
    run_environmental_suite,
    train_or_load_model,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Module 19 robustness evaluation demo.")
    parser.add_argument("--train-per-class", type=int, default=150)
    parser.add_argument("--val-per-class", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-eval", type=int, default=1000)
    parser.add_argument("--skip-art", action="store_true", help="Run only clean and environmental tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = ROOT / "data" / "generated"
    download_dir = ROOT / "data" / "cifar10"
    model_dir = ROOT / "models"
    results_dir = ROOT / "results"

    prepare_cifar10_assets(
        data_dir,
        download_dir,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
    )
    train_images, train_labels = load_npz_dataset(data_dir / "train_subset.npz")
    val_images, val_labels = load_npz_dataset(data_dir / "validation_1000.npz")
    val_images = val_images[: args.max_eval]
    val_labels = val_labels[: args.max_eval]

    model_specs = [
        ModelSpec("standard_resnet18", "standard_resnet18_cifar10.pt"),
        ModelSpec("noise_augmented_resnet18", "noise_augmented_resnet18_cifar10.pt", augment_noise_std=0.06, label_smoothing=0.05),
    ]

    all_rows = []
    first_art_examples = None
    for spec in model_specs:
        model = train_or_load_model(
            model_dir / spec.checkpoint,
            train_images,
            train_labels,
            spec=spec,
            device=device,
            epochs=args.epochs,
        )
        environmental_rows, clean_state = run_environmental_suite(spec.name, model, val_images, val_labels, device=device)
        all_rows.extend(environmental_rows)

        if not args.skip_art:
            art_rows, art_examples = run_art_attack_suite(spec.name, model, val_images, val_labels, clean_state, device=device)
            all_rows.extend(art_rows)
            if first_art_examples is None and art_examples:
                first_condition = sorted(art_examples.keys())[0]
                first_art_examples = (first_condition, art_examples[first_condition])

    scorecard = robustness_scorecard(all_rows)
    csv_path = write_csv(scorecard, results_dir / "robustness_scorecard.csv")
    chart_path = plot_scorecard(scorecard, results_dir / "robustness_scorecard.png")
    if first_art_examples is not None:
        condition, adversarial_images = first_art_examples
        plot_example_grid(
            val_images,
            adversarial_images,
            results_dir / "sample_adversarial_examples.png",
            title=f"Sample adversarial examples: {condition}",
        )

    print(f"Wrote comparative scorecard: {csv_path}")
    print(f"Wrote scorecard chart: {chart_path}")
    print("Top rows:")
    for row in scorecard[:8]:
        print(row)


if __name__ == "__main__":
    main()
