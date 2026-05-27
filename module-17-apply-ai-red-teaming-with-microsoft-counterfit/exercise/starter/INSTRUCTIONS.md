# Exercise: Build a Counterfit Assessment Workflow for a Drone Detection System

Estimated time: 45 minutes

## Overview

In this exercise, you will implement an AI red teaming workflow using Microsoft Counterfit against a drone detection image classifier. The provided target wraps a PyTorch image classifier behind Counterfit's `CFTarget` interface. You will configure attacks, execute them, compare results, and interpret operational risk.

The classroom dataset uses a compact CIFAR-derived aerial-object proxy: `airplane` images represent the `drone` class and selected non-aerial classes represent `no_drone`. The security workflow and evaluation metrics are the focus.

## Scenario

A critical infrastructure facility deploys an AI-powered drone detection platform that monitors restricted airspace using multiple camera feeds and automated alerting pipelines. The system processes approximately 5,000 image frames per minute with strict low-latency requirements.

Security engineers want to determine whether adversarial attacks could reduce detection reliability or create dangerous false negatives before the platform is deployed across additional facilities.

## Tasks

1. Load the provided drone detection model and validation dataset.
2. Configure the model as a valid Counterfit target.
3. Verify baseline prediction behavior.
4. Execute at least 3 Counterfit attacks.
5. Measure attack success rate, adversarial accuracy, confidence degradation, and drone false-negative behavior.
6. Compare attack effectiveness.
7. Identify the highest operational-risk attack.
8. Complete the short AI red team assessment report.

## Deliverables

- Completed notebook implementing the Counterfit assessment workflow.
- Successful execution of at least 3 Counterfit attack modules.
- Quantitative comparison table of attack effectiveness metrics.
- Visual or tabular outputs showing clean versus adversarial prediction changes.
- Short assessment report describing operational risks and mitigations.

## Acceptance Criteria

- The notebook configures and executes Counterfit attacks.
- At least 3 adversarial attacks complete successfully.
- Attack success rate and adversarial accuracy are calculated correctly.
- Clean versus adversarial prediction changes are included.
- Final report includes operational impact and mitigation guidance.

## Expert Tips

- Use operational metrics, especially false negatives for actual drone images.
- Compare multiple attack families because different algorithms expose different weaknesses.
- Record attack configuration values carefully so assessment results are reproducible.
