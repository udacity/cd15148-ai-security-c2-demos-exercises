from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
os.environ.setdefault("ART_DATA_PATH", str(ROOT / "art_data"))

from model_inversion_demo_utils import (  # noqa: E402
    AttackConfig,
    art_status,
    class_prototypes,
    evaluate_predictions,
    invert_model_outputs,
    invert_model_outputs_with_face_prior,
    load_att_face_dataset,
    nearest_training_examples,
    plot_face_recovery_examples,
    plot_leakage_comparison,
    plot_reconstruction_grid,
    reconstruction_metrics,
    seed_everything,
    train_or_load_model,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Module 15 facial model inversion demo.")
    parser.add_argument("--local-att-dir", type=Path, default=ROOT / "data" / "att_faces")
    parser.add_argument("--train-per-identity", type=int, default=7)
    parser.add_argument("--val-per-identity", type=int, default=3)
    parser.add_argument("--validation-queries", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--attack-steps", type=int, default=180)
    parser.add_argument("--attack-restarts", type=int, default=2)
    parser.add_argument("--prior-components", type=int, default=80)
    parser.add_argument("--targets", type=int, default=6, help="Number of identities to reconstruct.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = ROOT / "data" / "generated"
    model_dir = ROOT / "models"
    results_dir = ROOT / "results"

    dataset = load_att_face_dataset(
        data_dir,
        local_att_dir=args.local_att_dir,
        train_per_identity=args.train_per_identity,
        val_per_identity=args.val_per_identity,
        validation_queries=args.validation_queries,
        image_size=args.image_size,
    )
    print(f"Dataset: {dataset.dataset_name}")
    print(f"Source: {dataset.source}")
    print(f"Images: {len(dataset.train_images)} train images, {len(dataset.val_images)} validation queries")
    print(f"Device: {device}")
    print(f"ART status: {art_status()['note']}")

    model = train_or_load_model(
        model_dir / "facility_face_cnn_overfit.pt",
        dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        overfit=True,
    )
    baseline_metrics, probability_rows = evaluate_predictions(
        model,
        dataset.val_images,
        dataset.val_labels,
        dataset.class_names,
        device=device,
    )
    confidence_path = write_csv(probability_rows, results_dir / "sample_confidence_outputs.csv")

    target_classes = list(range(min(args.targets, len(dataset.class_names))))
    prototypes = class_prototypes(dataset.train_images, dataset.train_labels, target_classes)

    rich_config = AttackConfig(
        steps=args.attack_steps,
        restarts=args.attack_restarts,
        confidence_mode="rich_probability",
    )
    rounded_config = AttackConfig(
        steps=args.attack_steps,
        restarts=args.attack_restarts,
        confidence_mode="rounded_probability",
    )
    image_size = int(dataset.train_images.shape[-1])
    rich_reconstructions = invert_model_outputs(
        model,
        target_classes,
        config=rich_config,
        device=device,
        image_size=image_size,
    )
    rounded_reconstructions = invert_model_outputs(
        model,
        target_classes,
        config=rounded_config,
        device=device,
        image_size=image_size,
    )
    prior_reconstructions = invert_model_outputs_with_face_prior(
        model,
        target_classes,
        dataset.val_images,
        config=rich_config,
        device=device,
        n_components=args.prior_components,
    )
    nearest_examples = nearest_training_examples(
        prior_reconstructions,
        dataset.train_images,
        dataset.train_labels,
    )

    metric_rows = []
    metric_rows.extend(reconstruction_metrics("rich_probability", rich_reconstructions, prototypes, dataset.class_names))
    metric_rows.extend(reconstruction_metrics("rounded_probability", rounded_reconstructions, prototypes, dataset.class_names))
    metric_rows.extend(reconstruction_metrics("face_prior_probability", prior_reconstructions, prototypes, dataset.class_names))
    metrics_path = write_csv(metric_rows, results_dir / "model_inversion_metrics.csv")
    json_path = write_json(
        {
            "scenario": "Controlled facility facial recognition privacy assessment",
            "daily_authentication_requests": 60000,
            "dataset": dataset.dataset_name,
            "source": dataset.source,
            "training_images": int(len(dataset.train_images)),
            "validation_queries": int(len(dataset.val_images)),
            "baseline_model": baseline_metrics,
            "art_status": art_status(),
            "attack_metrics": metric_rows,
            "mitigation_notes": [
                "Return labels instead of full probability vectors when business requirements allow.",
                "Round or bucket confidence scores to reduce optimization signal.",
                "Rate-limit repeated inference queries and monitor identity enumeration patterns.",
                "Evaluate overfitting and consider privacy-preserving training where appropriate.",
            ],
        },
        results_dir / "model_inversion_summary.json",
    )
    grid_path = plot_reconstruction_grid(
        rich_reconstructions,
        rounded_reconstructions,
        prototypes,
        dataset.class_names,
        results_dir / "reconstructed_feature_approximations.png",
    )
    face_recovery_path = plot_face_recovery_examples(
        prior_reconstructions,
        prototypes,
        nearest_examples,
        dataset.class_names,
        results_dir / "recovered_faces_from_model_inversion.png",
    )
    chart_path = plot_leakage_comparison(metric_rows, results_dir / "confidence_leakage_comparison.png")

    print("\nBaseline model outputs")
    print(f"{'Metric':<24} {'Value':>10}")
    print("-" * 36)
    print(f"{'Validation accuracy':<24} {baseline_metrics['accuracy']:>10.4f}")
    print(f"{'Mean confidence':<24} {baseline_metrics['mean_confidence']:>10.4f}")
    print(f"{'P95 confidence':<24} {baseline_metrics['p95_confidence']:>10.4f}")

    print("\nModel inversion results")
    print(f"{'Setting':<22} {'Identity':<14} {'Conf':>8} {'Cosine':>8} {'Queries':>8}")
    print("-" * 68)
    for row in metric_rows:
        print(
            f"{row['attack_setting']:<22} "
            f"{row['target_identity']:<14} "
            f"{float(row['target_confidence']):>8.4f} "
            f"{float(row['prototype_cosine_similarity']):>8.4f} "
            f"{int(row['queries']):>8}"
        )

    print("\nOutputs")
    print(f"Confidence vectors: {confidence_path}")
    print(f"Inversion metrics: {metrics_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Reconstruction grid: {grid_path}")
    print(f"Recovered face examples: {face_recovery_path}")
    print(f"Leakage chart: {chart_path}")


if __name__ == "__main__":
    main()
