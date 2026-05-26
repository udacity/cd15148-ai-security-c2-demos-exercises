# Poisoned Training Dataset Assessment Exercise

This exercise contains a starter and solution implementation for a traffic sign label-flipping poisoning assessment.

The workflow extends the Module 9 data poisoning demo and reuses the traffic-sign setting from the Module 7 stop sign evasion demo. Instead of creating triggered backdoor examples, students implement a simpler label-flipping attack: a portion of `Stop` signs in the training data are relabeled as `Yield`, then the model is retrained and evaluated for class-specific failure.

## Structure

- `starter/`: student-facing scaffold with TODO sections.
- `solution/`: completed reference implementation.

## Environment

Use Python 3.11.

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

## Dataset

The notebooks download GTSRB through `torchvision.datasets.GTSRB` when run. The exercise uses a compact subset of six operationally relevant sign classes so the training loop remains practical in a classroom environment.

Generated datasets, checkpoints, plots, and result tables are ignored by git.
