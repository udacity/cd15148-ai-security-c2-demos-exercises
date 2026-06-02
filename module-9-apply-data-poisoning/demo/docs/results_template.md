# Results Template

| Model | Clean Accuracy | Source-Class Accuracy | Targeted Attack Success Rate (Triggered Input) | Mean Confidence | Notes |
|-------|----------------|-----------------------|------------------------------------------------|-----------------|-------|
| Clean baseline | TODO | TODO | TODO | TODO | Trained on clean data (shipped checkpoint) |
| Backdoor-poisoned | TODO | TODO | TODO | TODO | ~8% of training set triggered and relabeled |
| Label-flipped | TODO | TODO | TODO | TODO | 75% of source-class labels flipped to target |

## Discussion Prompts

1. Did aggregate clean validation accuracy reveal either attack?
2. Did per-class accuracy reveal one attack but not the other? Which, and why?
3. The backdoor attack's targeted attack success rate is measured on *triggered* inputs. What does that imply about validation pipelines that only test clean inputs?
4. The label-flip attack does not need a trigger at inference. What does that imply about the attacker's required capabilities?
5. What combination of defender-side checks (aggregate accuracy, per-class accuracy, trigger-aware tests, dataset provenance, hashing) would catch both attacks?