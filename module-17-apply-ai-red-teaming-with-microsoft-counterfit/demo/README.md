# Demo: Build an Automated AI Red Team Assessment with Counterfit

Estimated time: 15 minutes

## Overview

This demo shows how to structure an automated AI red team assessment for a computer vision model using Microsoft Counterfit-style target and attack configuration. Learners configure a model target, execute multiple adversarial attacks, compare attack outcomes, and interpret security results in operational context.

The notebook uses Microsoft Counterfit directly through its Python API: a target class wraps the CIFAR-10 PyTorch model, `Counterfit.build_attack(...)` creates attack objects, and `Counterfit.run_attack(...)` executes each attack.

## Scenario

A logistics company deploys a computer vision model to classify package labels and hazard indicators in a distribution center processing approximately 250,000 packages per day. The AI security team wants to rapidly assess whether adversarial attacks could manipulate predictions before the model is connected to automated routing and handling workflows.

## Contents

- `notebooks/counterfit_red_team_assessment_demo.ipynb`: main walkthrough.
- `src/counterfit_demo_utils.py`: model, dataset, Counterfit orchestration, and reporting helpers.
- `targets/logistics_cifar10_resnet18.py`: Counterfit target class for the CIFAR-10 ResNet-18 model.
- `configs/counterfit_target_config.json`: target configuration template.
- `configs/counterfit_attack_plan.json`: attack plan template.
- `results/sample_counterfit_attack_results.csv`: sample assessment output.

## Run

Microsoft Counterfit currently documents Linux or Windows with WSL as the supported local environment. Use Python 3.8 in the student environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Open:

```text
notebooks/counterfit_red_team_assessment_demo.ipynb
```

The first run downloads CIFAR-10 and trains a compact ResNet-18 checkpoint if one is not already present.

## Counterfit Note

Do not install the unrelated PyPI package named `CounterFit`; that package is an IoT simulator. This demo depends on Microsoft Counterfit from `https://github.com/Azure/counterfit`.
