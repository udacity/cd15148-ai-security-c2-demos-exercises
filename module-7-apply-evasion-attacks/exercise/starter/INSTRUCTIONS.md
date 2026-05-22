# Exercise: Build an Adversarial Evaluation Workflow for an Aerial Object Classifier

Estimated time: 40 minutes

## Overview

In this exercise, you will implement an adversarial evaluation workflow against an aerial object image classifier used in a security monitoring environment. You will use provided starter code, a binary classifier trained from the CIFAR-10 `airplane` class, and the Adversarial Robustness Toolbox to evaluate how evasion attacks affect model reliability.

This exercise extends the demo by adding quantitative metrics across multiple perturbation strengths.

## Scenario

A security operations team deploys an aerial object classifier at a critical infrastructure facility to identify unauthorized aircraft-like objects near restricted airspace. For this course exercise, CIFAR-10 `airplane` images stand in for the restricted aerial object class. The system processes live camera feeds with an average inference latency of 90 ms and triggers automated alerts when aerial objects are detected.

Before operational deployment, the team wants to determine whether adversarial image perturbations could reduce detection reliability or cause dangerous false negatives.

## Your Tasks

1. Prepare the CIFAR-10 airplane binary dataset and pretrained model checkpoint.
2. Load the aerial object classifier and validation dataset using the supplied starter code.
3. Measure baseline accuracy and confidence scores on clean validation images.
4. Select and implement one evasion attack from the notebook options.
5. Evaluate at least three epsilon configurations.
6. Generate adversarial examples and compare predictions against clean inputs.
7. Compute adversarial accuracy, attack success rate, confidence degradation, and perturbation magnitude.
8. Create visual comparisons showing clean and adversarial predictions.
9. Write a short robustness assessment in `docs/robustness_assessment.md`.

## Setup

Install dependencies from this folder:

```bash
pip install -r requirements.txt
```

Prepare the starter dataset and model checkpoint:

```bash
bash scripts/prepare_airplane_assets.sh
```

On Windows without Bash, run:

```powershell
python scripts/prepare_airplane_assets.py
```

## Deliverables

Submit:

- A completed notebook implementing the adversarial evaluation workflow
- Generated adversarial examples across at least three epsilon values
- A comparison table showing clean accuracy versus adversarial accuracy
- Visual outputs comparing clean and adversarial predictions
- A short robustness assessment summarizing attack effectiveness and operational risk

## Acceptance Criteria

- The selected attack successfully generates adversarial examples.
- At least three epsilon configurations are evaluated.
- The notebook computes attack success rate and adversarial accuracy correctly.
- Visual comparisons display both clean and adversarial predictions.
- The final report includes quantitative analysis and operational risk discussion.

## Expert Tips

- Start with smaller epsilon values and gradually increase them to observe how attack visibility changes alongside effectiveness.
- Monitor model confidence scores in addition to accuracy. Confidence degradation often appears before full misclassification.
- Evaluate operational failure modes, not just aggregate benchmark metrics.
