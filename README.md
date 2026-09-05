# Causal alignment experiments

This repository contains matched toy experiments testing whether causal
fine-tuning improves compositional and out-of-distribution generalization.

## Layout

- `toy/arithmetic/within_task/`: permutation composition and its six-method comparison.
- `toy/safety/`: safety-policy generalization to held-out surface styles.
- `causal/prepare_learned_multipass.py`: constructs paired `X -> Z` and
  `Z -> decision` supervision for one Qwen adapter.
- `causal/evaluate_learned_multipass.py`: evaluates predicted, oracle, and
  deliberately mismatched textual `Z` in a non-bypassable second call.
- `causal/evaluate_oracle_enforced_path.py`: positive control with oracle `Z`,
  a deterministic policy, and a fixed refusal response.
- `jobs/`: SCC launchers. Submit from the repository root with `qsub jobs/<script>`.

Both comparisons start from an atomically pretrained four-layer Transformer
and test regular SFT, explicit two-pass `Z`, `Z` supervision, IIT, a full
graph-cut/canonical scaffold, and staged scaffold removal. Each run records
held-out accuracy, causal-intervention accuracy, atomic retention, and learning
curves. The published safety result uses three seeds; the published permutation
result is a single-seed pilot.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete data and execution
path. Compact Qwen summaries are in `causal/results/published/`; large model
weights, source datasets, and raw harmful generations are intentionally not
stored in this repository.

Run the checks from the repository root:

```bash
python -m unittest discover -s tests
```

## Qwen causal-path experiments

The learned multi-pass experiment calls the same fine-tuned Qwen adapter twice:

```text
request X -> structured textual Z
structured textual Z only -> REFUSE or COMPLY
```

Because the second call cannot see the request, its decision cannot bypass
`Z`.  Build its training data from JSONL records containing `prompt`, `kind`,
and a tagged safety-state `response`:

```bash
python -m causal.prepare_learned_multipass \
  --input causal/data/qwen_multipass_analyzer_seed2025.jsonl \
  --output causal/data/qwen_learned_multipass_shared_seed2025.jsonl
qsub jobs/run_qwen_learned_multipass.sh
```

The oracle-enforced experiment supplies the correct harmful/benign `Z` and
uses a fixed refusal for harmful requests.  It is a positive control, not a
learned solution:

```bash
qsub jobs/run_qwen_oracle_enforced_path.sh
```
