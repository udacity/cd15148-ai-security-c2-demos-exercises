# Instructor Notes

## Timing

| Segment | Time | Talking Point |
|---------|------|---------------|
| Scenario and threat model | 1 min | Poisoning attacks compromise training, not just inference. |
| Load CIFAR-10 subset and baseline model | 2 min | The CIFAR-10 classes stand in for manufacturing component categories. |
| Baseline evaluation | 2 min | Clean validation accuracy is necessary but incomplete. |
| Inject poisoned samples | 3 min | A small label-flipped triggered subset can create hidden behavior. |
| Retrain poisoned model | 3 min | The poisoned model can preserve normal validation behavior. |
| Compare clean and triggered metrics | 2 min | Targeted attack success rate exposes behavior that clean validation misses. |

## Scenario Mapping

CIFAR-10 is used as a lightweight classroom dataset. In the story, classes represent visual inspection categories on an assembly line. The trigger pattern represents a hidden data manipulation inserted into training samples by an attacker with limited pipeline access.

## Key Discussion Point

From clean validation accuracy alone, it usually cannot be determined that a model contains a targeted backdoor. Evaluators need trigger-aware tests, data lineage checks, training data integrity controls, and class-conditional behavior analysis.
