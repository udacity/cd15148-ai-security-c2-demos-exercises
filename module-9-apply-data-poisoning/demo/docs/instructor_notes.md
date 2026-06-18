# Instructor Notes

## Timing

These times assume the shipped clean baseline checkpoint is loaded rather than retrained. Live training of the backdoor and label-flipped models dominates the runtime; numbers below are rough orientation for an MPS / CUDA T4 class environment. **For recorded video walkthroughs, the two live training cells can be cut from the recording** — they produce no narrative information beyond a progress indicator. Resume narration when the trained model is evaluated.

| Segment | On-video time | Talking Point |
|---------|---------------|---------------|
| Scenario and threat model | 1 min | Poisoning attacks compromise training, not just inference. |
| Load CIFAR-10 subset and clean baseline checkpoint | 1 min | Recipe: ResNet-18 from scratch, 2,000 images per class, 15 epochs, light augmentation. The checkpoint is shipped so retraining is not in the recording. |
| Inject the backdoor trigger and visualize | 2 min | A small triggered + relabeled subset can create hidden behavior. |
| Retrain on the backdoor-poisoned dataset | **cut from video** | Same recipe as baseline. ~5 min on a T4 GPU; longer on MPS / CPU. Pause the recording, let the cell finish, then resume. |
| Measure backdoor attack success | 2 min | Targeted attack success rate on triggered inputs is the only way to see this attack. |
| Inject the label flip and visualize | 2 min | No pixel change — only labels. No trigger needed at inference. |
| Retrain on the label-flipped dataset | **cut from video** | ~5 min on T4. Same treatment — pause and resume. |
| Per-class accuracy and three-way comparison | 3 min | Per-class accuracy catches label flipping but not backdoors. Aggregate accuracy catches neither. |
| Interpretation and mitigations | 3 min | Defense layers: aggregate metrics, per-class metrics, trigger-aware tests, dataset provenance. |

**On-video total: ~14 min.** Cut training cells run independently on the instructor's machine.

## Scenario Mapping

CIFAR-10 is used as a lightweight classroom dataset. In the story, classes represent visual inspection categories on an assembly line. The trigger pattern represents a hidden data manipulation inserted by an attacker with limited pipeline access. The label-flipping variant represents a compromised annotator or labeling vendor — same outcome (a poisoned model) achieved without ever touching pixels.

## Key Discussion Points

- Aggregate validation accuracy alone catches neither attack. Both poisoned models match the clean baseline within a few points.
- Per-class accuracy catches the label-flip attack immediately (source class drops to ~0%) but does not catch the backdoor (source-class accuracy stays normal on clean inputs).
- The backdoor only reveals itself if defenders actively test with triggered inputs, which standard validation pipelines do not do.
- Both attacks succeed with a small fraction of corrupted training data (~8% for the backdoor here, ~75% of one class's labels for the flip), demonstrating that protecting training data integrity is a structural defense, not a polish step.
