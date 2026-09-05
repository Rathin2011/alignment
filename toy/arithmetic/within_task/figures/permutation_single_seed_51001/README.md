# Permutation composition figures

These figures summarize the matched six-method permutation-composition run
with initialization seed `51001` and primitive-library seed `31001`.

- `heldout_accuracy`: accuracy on 768 composites from unseen `(A, B)` pairs.
- `causal_patching`: crossed-target accuracy after intervening on `Z`.
- `causal_control_vs_ood`: relationship between patching and held-out accuracy.
- `learning_curves`: held-out accuracy over fine-tuning updates.
- `atomic_retention`: accuracy on the atomic operations after fine-tuning.

PNG files are provided for Google Docs; SVG files are the vector originals.
This is a single-seed exploratory result and should not be described as a
multi-seed estimate.
