from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
os.environ["ART_DATA_PATH"] = str(ROOT / "art_data")
os.environ["USERPROFILE"] = str(ROOT)
os.environ["HOME"] = str(ROOT)
os.environ["MPLCONFIGDIR"] = str(ROOT / ".matplotlib")

from traffic_sign_robustness_utils import (  # noqa: E402
    calculate_condition_metrics,
    environmental_test_sets,
    load_subset,
    perturbation_linf,
    plot_clean_vs_degraded,
    plot_metric_bars,
    prepare_gtsrb_subsets,
    row_from_metrics,
    run_fgsm_attacks,
    train_or_load_model,
    write_assessment_report,
    write_scorecard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the traffic sign robustness assessment solution.")
    parser.add_argument("--train-per-class", type=int, default=80)
    parser.add_argument("--val-per-class", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-eval", type=int, default=120)
    parser.add_argument("--skip-art", action="store_true", help="Skip adversarial tests for a quick smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = ROOT / "data" / "generated"
    download_dir = ROOT / "data" / "gtsrb"
    model_path = ROOT / "models" / "traffic_sign_cnn.pt"
    results_dir = ROOT / "results"

    prepare_gtsrb_subsets(data_dir, download_dir, args.train_per_class, args.val_per_class)
    train_images, train_labels = load_subset(data_dir, "train")
    val_images, val_labels = load_subset(data_dir, "val")
    val_images = val_images[: args.max_eval]
    val_labels = val_labels[: args.max_eval]

    model = train_or_load_model(model_path, train_images, train_labels, device=device, epochs=args.epochs)
    clean_metrics = calculate_condition_metrics(model, val_images, val_labels, device=device)
    clean_predictions = clean_metrics["predictions"]
    clean_confidence = clean_metrics["confidence"]
    rows = [row_from_metrics("clean", "baseline", clean_metrics, perturbation=0.0)]

    first_degraded_set = None
    for condition, degraded_images in environmental_test_sets(val_images).items():
        metrics = calculate_condition_metrics(
            model,
            degraded_images,
            val_labels,
            clean_predictions=clean_predictions,
            clean_confidence=clean_confidence,
            device=device,
        )
        rows.append(row_from_metrics(condition, "environmental", metrics, perturbation_linf(val_images, degraded_images)))
        if first_degraded_set is None:
            first_degraded_set = (condition, degraded_images)

    first_adversarial_set = None
    if not args.skip_art:
        for condition, adversarial_images in run_fgsm_attacks(model, val_images, epsilons=[0.01, 0.03, 0.06]).items():
            metrics = calculate_condition_metrics(
                model,
                adversarial_images,
                val_labels,
                clean_predictions=clean_predictions,
                clean_confidence=clean_confidence,
                device=device,
            )
            rows.append(row_from_metrics(condition, "adversarial", metrics, perturbation_linf(val_images, adversarial_images)))
            if first_adversarial_set is None:
                first_adversarial_set = (condition, adversarial_images)

    write_scorecard(rows, results_dir / "traffic_sign_robustness_scorecard.csv")
    plot_metric_bars(rows, results_dir / "traffic_sign_metric_comparison.png")
    write_assessment_report(rows, results_dir / "traffic_sign_assessment_report.md")

    if first_degraded_set is not None:
        condition, degraded_images = first_degraded_set
        plot_clean_vs_degraded(
            val_images,
            degraded_images,
            results_dir / "sample_environmental_degradation.png",
            title=f"Clean vs {condition}",
        )
    if first_adversarial_set is not None:
        condition, adversarial_images = first_adversarial_set
        plot_clean_vs_degraded(
            val_images,
            adversarial_images,
            results_dir / "sample_adversarial_degradation.png",
            title=f"Clean vs {condition}",
        )

    print("Assessment complete. Results written to:", results_dir)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
