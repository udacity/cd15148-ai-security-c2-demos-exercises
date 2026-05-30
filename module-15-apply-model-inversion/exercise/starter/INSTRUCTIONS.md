# Exercise: Build a Model Inversion Privacy Assessment Workflow for a Medical Diagnosis Model

## Scenario

A healthcare provider deploys an AI system that classifies brain MRI images for preliminary tumor screening workflows. The model processes approximately 10,000 patient images per week through a cloud-hosted inference API. Security engineers and privacy officers want to determine whether attackers could reconstruct sensitive patient characteristics or infer properties of the underlying training dataset through repeated inference queries before the platform is approved for broader external access.

## Your Tasks

- Load the provided model and evaluation dataset.
- Measure baseline prediction confidence and output distributions.
- Implement repeated-query model inversion.
- Evaluate at least three output configurations for privacy leakage impact.
- Generate visual comparisons between representative samples and reconstructions.
- Produce a short privacy assessment report with operational risks and mitigations.

## Acceptance Criteria

- The workflow performs repeated inference queries against the provided model.
- Reconstructed outputs are generated successfully.
- At least three output configurations are evaluated.
- Visual comparisons are included.
- The final report includes operational privacy risks and mitigation recommendations.

## Expert Tips

- Limiting prediction outputs to top-1 labels can significantly reduce inversion effectiveness.
- Overfitting increases both apparent model confidence and privacy leakage risk.
- Differential privacy, confidence clipping, and output restriction are useful mitigations for sensitive deployments.
