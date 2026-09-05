#!/bin/bash -l
#$ -N oracle_path
#$ -P buinlp
#$ -cwd
#$ -j y
#$ -o jobs/oracle_path.$JOB_ID.log
#$ -l h_rt=02:00:00
#$ -l gpus=1
#$ -l gpu_type=A6000
#$ -l gpu_memory=40G
#$ -l mem_per_core=8G
#$ -pe omp 2

set -euo pipefail
source .venv-olmo/bin/activate
export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python -m causal.evaluate_oracle_enforced_path \
  --input causal/results/qwen2.5-7b_no_sft/judged.jsonl \
  --judge artifacts/models/models--ibm-granite--granite-guardian-3.1-2b/snapshots/81145486e85c6c82c01e759c0356d9d6da4d21a5 \
  --output causal/results/qwen2.5-7b_oracle_enforced_path
