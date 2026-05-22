# CIFAR-10 Airplane Dataset

Run the setup script to download CIFAR-10 and generate a balanced binary dataset:

```bash
bash scripts/prepare_airplane_assets.sh
```

Generated files are written under:

```text
data/generated/
```

Binary labels:

- `0`: background, sampled from non-airplane CIFAR-10 classes
- `1`: airplane, sampled from the CIFAR-10 `airplane` class
