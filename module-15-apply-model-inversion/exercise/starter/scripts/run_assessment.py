from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
os.environ.setdefault("ART_DATA_PATH", str(ROOT / "art_data"))

from medical_inversion_utils import (  # noqa: E402
    art_status,
    evaluate_model_outputs,
    inversion_metrics,
    output_configurations,
    plot_leakage_scores,
    plot_reconstruction_comparison,
    prepare_medical_dataset,
    representative_samples,
    run_inversion_attack,
    seed_everything,
    train_or_load_model,
    write_csv,
    write_json,
    write_privacy_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Module 15 brain tumor model inversion privacy assessment.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-per-class", type=int, default=120)
    parser.add_argument("--val-per-class", type=int, default=60)
    parser.add_argument("--attack-steps", type=int, default=160)
    parser.add_argument("--attack-restarts", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = ROOT / "data" / "generated"
    model_dir = ROOT / "models"
    results_dir = ROOT / "results"

    dataset = prepare_medical_dataset(
        data_dir,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
    )
    model = train_or_load_model(
        model_dir / "brain_tumor_screening_cnn.pt",
        dataset,
        device=device,
        epochs=args.epochs,
    )
    baseline_metrics, confidence_rows, _ = evaluate_model_outputs(
        model,
        dataset.val_images,
        dataset.val_labels,
        dataset.class_names,
        device=device,
    )
    confidence_path = write_csv(confidence_rows, results_dir / "baseline_confidence_outputs.csv")

    target_classes = list(range(len(dataset.class_names)))
    representatives = representative_samples(dataset.val_images, dataset.val_labels, target_classes)
    reconstruction_sets = {}
    metric_rows = []
    for config in output_configurations():
        reconstructions = run_inversion_attack(
            model,
            target_classes,
            output_config=config,
            device=device,
            image_size=int(dataset.train_images.shape[-1]),
            steps=args.attack_steps,
            restarts=args.attack_restarts,
        )
        reconstruction_sets[config.name] = reconstructions
        metric_rows.extend(inversion_metrics(config.name, reconstructions, representatives, dataset.class_names))

    metrics_path = write_csv(metric_rows, results_dir / "model_inversion_privacy_metrics.csv")
    comparison_path = plot_reconstruction_comparison(
        reconstruction_sets,
        representatives,
        dataset.class_names,
        results_dir / "reconstructed_medical_features.png",
    )
    chart_path = plot_leakage_scores(metric_rows, results_dir / "privacy_leakage_by_output_config.png")
    summary_path = write_json(
        {
            "scenario": "Brain tumor MRI screening model inversion privacy assessment",
            "weekly_patient_images": 10000,
            "dataset": dataset.dataset_name,
            "baseline_model": baseline_metrics,
            "art_status": art_status(),
            "attack_metrics": metric_rows,
        },
        results_dir / "privacy_assessment_summary.json",
    )
    report_path = write_privacy_report(
        results_dir / "privacy_assessment_report.md",
        baseline_metrics,
        metric_rows,
        chart_path,
        comparison_path,
    )

    print(f"Dataset: {dataset.dataset_name}")
    print(f"Device: {device}")
    print(f"ART status: {art_status()['note']}")
    print("\nBaseline model outputs")
    print(f"{'Metric':<24} {'Value':>10}")
    print("-" * 36)
    print(f"{'Validation accuracy':<24} {baseline_metrics['accuracy']:>10.4f}")
    print(f"{'Mean confidence':<24} {baseline_metrics['mean_confidence']:>10.4f}")
    print(f"{'P95 confidence':<24} {baseline_metrics['p95_confidence']:>10.4f}")
    print(f"{'Mean entropy':<24} {baseline_metrics['mean_entropy']:>10.4f}")

    print("\nModel inversion results")
    print(f"{'Output config':<22} {'Target':<24} {'Conf':>8} {'Leakage':>9} {'Queries':>8}")
    print("-" * 78)
    for row in metric_rows:
        print(
            f"{row['output_configuration']:<22} "
            f"{row['target_class']:<24} "
            f"{float(row['model_confidence']):>8.4f} "
            f"{float(row['privacy_leakage_score']):>9.4f} "
            f"{int(row['queries']):>8}"
        )

    print("\nOutputs")
    print(f"Confidence outputs: {confidence_path}")
    print(f"Privacy metrics: {metrics_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Visual comparison: {comparison_path}")
    print(f"Leakage chart: {chart_path}")
    print(f"Assessment report: {report_path}")


if __name__ == "__main__":
    main()
