# Solution: Counterfit Drone Detection Assessment

This solution completes the Counterfit assessment workflow for a binary drone detection classifier.

The reference workflow:

- Loads a compact aerial-object dataset.
- Wraps the PyTorch model as a Counterfit target.
- Executes three Counterfit attack modules.
- Calculates attack success rate, adversarial accuracy, confidence degradation, and drone false-negative rate.
- Writes `results/solution_counterfit_drone_results.csv`.

The notebook requires Microsoft Counterfit in a supported Linux/WSL Python 3.8 environment.
