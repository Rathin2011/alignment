# Causal alignment experiments

This repository contains matched toy experiments testing whether causal
fine-tuning improves compositional and out-of-distribution generalization.

## Layout

- `toy/arithmetic/within_task/`: permutation composition and its six-method comparison.
- `toy/safety/`: safety-policy generalization to held-out surface styles.
- `jobs/`: SCC launchers. Submit from the repository root with `qsub jobs/<script>`.

Both comparisons start from an atomically pretrained four-layer Transformer
and test regular SFT, explicit two-pass `Z`, `Z` supervision, IIT, a full
graph-cut/canonical scaffold, and staged scaffold removal. Each run records
held-out accuracy, causal-intervention accuracy, atomic retention, and learning
curves over three seeds.

Run the checks from the repository root:

```bash
python -m unittest discover -s tests
```
