# Counterfit Drone Detection Assessment Exercise

This exercise extends the Module 17 Counterfit demo into a student workflow for assessing a drone detection model.

The exercise uses Microsoft Counterfit directly. Students configure a target class, run three adversarial attacks, and compare operational impact metrics.

## Structure

- `starter/`: student scaffold with TODOs.
- `solution/`: completed reference implementation.

## Environment

Microsoft Counterfit is expected to be preinstalled in the course environment. Microsoft Counterfit currently documents Linux or Windows with WSL as the supported local environment for local installs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Do not install the unrelated PyPI package named `CounterFit`; that package is an IoT simulator. This exercise depends on Microsoft Counterfit from `https://github.com/Azure/counterfit`. If you are testing outside the course environment, install the official Counterfit package separately in a supported Python 3.8 Linux/WSL environment.
