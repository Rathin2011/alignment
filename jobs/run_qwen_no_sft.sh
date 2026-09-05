#!/bin/bash -l
#$ -N causal_base
#$ -P buinlp
#$ -cwd
#$ -j y
#$ -o jobs/causal_base.$JOB_ID.log
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_type=A6000
#$ -l gpu_memory=40G
#$ -l mem_per_core=12G
#$ -pe omp 4

set -euo pipefail
source .venv-olmo/bin/activate
export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

judge=$(find artifacts/models/models--ibm-granite--granite-guardian-3.1-2b/snapshots \
  -mindepth 1 -maxdepth 1 -type d | head -1)

python -m causal.evaluate_no_sft \
  --model causal/models/Qwen2.5-7B \
  --judge "$judge" \
  --coca-data causal/COCA/data \
  --xstest artifacts/datasets/xstest/xstest_prompts.csv \
  --output causal/results/qwen2.5-7b_no_sft
