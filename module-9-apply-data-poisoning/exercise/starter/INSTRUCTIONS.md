# Exercise: Build a Poisoned Training Dataset Assessment Workflow

Estimated time: 45 minutes

## Overview

In this exercise, you will implement a poisoning assessment workflow against a traffic sign image classifier. The workflow uses the same traffic-sign security scenario as the evasion demo, but shifts the focus from inference-time adversarial examples to training-time data pipeline compromise.

You will inject a simple label-flipping poison into a clean training subset, retrain the model, and compare the resulting model behavior against a clean baseline. The goal is to measure whether a small number of manipulated labels can create operationally meaningful class-specific failures while leaving aggregate validation accuracy easy to misread.

## Scenario

A transportation technology company develops a traffic sign recognition model used in autonomous shuttle systems operating across a corporate campus. The model processes camera feeds continuously with low-latency inference requirements to support navigation and safety systems.

Security engineers want to evaluate whether manipulated training data could introduce hidden behaviors that cause specific traffic signs to be misclassified before the model is approved for expanded deployment.

## Tasks

1. Load the clean traffic sign training and validation subsets.
2. Train or load the clean baseline traffic sign classifier.
3. Measure baseline validation accuracy and class-specific behavior.
4. Implement a label-flipping poisoning function.
5. Retrain the model on the poisoned dataset.
6. Measure clean accuracy, source-class accuracy, and targeted misclassification rate.
7. Generate visual examples showing flipped-label samples.
8. Complete the short operational risk assessment.

## Deliverables

Submit:

- A completed notebook or Python workflow implementing the poisoning assessment.
- A retrained model generated using poisoned training data.
- A quantitative comparison of clean versus poisoned model behavior.
- Visual examples of flipped-label training samples.
- A short report summarizing attack effectiveness, stealth, operational impact, and mitigations.

## Acceptance Criteria

- The poisoning workflow successfully modifies training labels.
- The retrained model demonstrates measurable poisoned behavior.
- The notebook computes clean accuracy and targeted misclassification rate correctly.
- Visual outputs clearly identify flipped-label samples.
- The final report includes operational risk analysis and mitigation discussion.

## Expert Tips

- Start with smaller poisoning percentages and increase gradually.
- Compare class-specific accuracy changes rather than relying only on aggregate validation accuracy.
- Label flipping does not require image triggers, so it can be harder to detect through visual inspection alone.
- Dataset provenance, label quality review, and class-conditional validation are important defenses.
