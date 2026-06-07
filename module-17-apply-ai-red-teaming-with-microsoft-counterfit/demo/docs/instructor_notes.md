# Instructor Notes

This demo uses Microsoft Counterfit directly. The target class lives in `targets/logistics_cifar10_resnet18.py`, and the notebook runs attacks with `Counterfit.build_attack(...)` and `Counterfit.run_attack(...)`.

## Suggested Flow

1. Show the target configuration.
2. Show the attack plan.
3. Load the model and 500-image validation subset.
4. Run the three Counterfit attacks.
5. Compare attack success rate, confidence drop, and perturbation size.
6. Discuss why automated tooling still needs human interpretation.

## Talking Points

- Attack automation improves repeatability and coverage.
- Parameter choices change results significantly.
- A high attack success rate is not automatically a complete risk finding; it needs operational context.
- Automated red teaming tools help generate evidence, but humans still decide scope, impact, and mitigation priority.
