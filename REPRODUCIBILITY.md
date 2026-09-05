# Reproducing the experiments

Use Python 3.10 and install `requirements.txt`. `gnuplot` is needed only for
rendering the figures.

## Toy experiments

From the repository root:

```bash
python toy/arithmetic/within_task/matched_methods.py \
  --output toy/arithmetic/within_task/results/reproduction
python toy/safety/matched_methods.py \
  --output toy/safety/results/reproduction
```

The published permutation artifact is a single-seed pilot. The default command
now runs three seeds. The published toy-safety summary contains three seeds.

## External Qwen prerequisites

The repository does not redistribute model weights or third-party datasets.
Place the following resources at the paths used by the SCC launchers:

- Qwen-2.5-7B: `causal/models/Qwen2.5-7B`
- Granite Guardian 3.1 2B: `artifacts/models/.../granite-guardian-3.1-2b/...`
- COCA data: clone `https://github.com/tmlr-group/COCA.git` to `causal/COCA`
- XSTest: clone `https://github.com/paul-rottger/xstest.git` to
  `artifacts/datasets/xstest`

## Rebuild the exact Qwen training records

The following commands reproduce the recorded data hashes:

```bash
python -m causal.prepare_causal_training \
  --data-dir causal/COCA/data \
  --output causal/data/coca_causal_reasoning_pilot_seed2025.jsonl \
  --harmful-count 500 --benign-count 3000 --seed 2025

python -m causal.prepare_multipass_analyzer \
  --input causal/data/coca_causal_reasoning_pilot_seed2025.jsonl \
  --output causal/data/qwen_multipass_analyzer_seed2025.jsonl \
  --seed 2025

python -m causal.prepare_learned_multipass \
  --input causal/data/qwen_multipass_analyzer_seed2025.jsonl \
  --output causal/data/qwen_learned_multipass_shared_seed2025.jsonl \
  --seed 2025
```

Expected SHA-256 hashes:

```text
7b3e73a3375f35f5e8177887a3cc67c8e198ea7ed2aafe99796e871fbe1ac784  coca_causal_reasoning_pilot_seed2025.jsonl
f9894e5d583cb90070d659dfded564e4d06f76963d0e6fc919d57077cdec0a3b  qwen_learned_multipass_shared_seed2025.jsonl
```

## Qwen execution order

```bash
qsub jobs/run_qwen_no_sft.sh
qsub jobs/run_qwen_learned_multipass.sh
qsub jobs/run_qwen_oracle_enforced_path.sh
```

The learned multi-pass job always evaluates `X -> Z -> decision`. It additionally
evaluates cached response routing when both the base-Qwen and safety-trained
response files named in the launcher are available.

Compact result summaries are committed under `causal/results/published`.
Raw generations are omitted because they are large and contain harmful content.
