# References and Runtime Notes

## Libraries

- PyTorch is used for the face classifier and transparent inversion loop.
- NumPy and Matplotlib are used for data processing and visualizations.
- Scikit-learn provides `fetch_olivetti_faces`, a packaged copy of the AT&T Laboratories Cambridge face dataset.
- Adversarial Robustness Toolbox is listed as an optional dependency. The runnable demo uses a compact PyTorch attack loop so learners can inspect every step.

## Dataset Note

The demo uses the AT&T Laboratories Cambridge face dataset, commonly referenced as the ORL or Olivetti faces dataset. If instructors have the original AT&T archive, place it under `data/att_faces/` with folders such as `s1`, `s2`, and so on. Otherwise the demo downloads the scikit-learn packaged copy on first run.

The dataset contains 40 subjects with 10 grayscale face images per subject. The default demo uses 7 images per subject for training and 3 held-out images per subject for validation, then cycles those held-out labeled images into 500 validation queries to match the lesson scenario's repeated-inference framing.

## Safety Note

This material is intended for defensive AI privacy assessment. Keep the discussion focused on reducing output leakage, controlling inference exposure, and validating privacy risks before deployment.
