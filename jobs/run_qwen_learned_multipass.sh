#!/bin/bash -l
#$ -N learned_multipass
#$ -P buinlp
#$ -cwd
#$ -j y
#$ -o jobs/learned_multipass.$JOB_ID.log
#$ -l h_rt=06:00:00
#$ -l gpus=1
#$ -l gpu_type=A6000
#$ -l gpu_memory=40G
#$ -l mem_per_core=8G
#$ -pe omp 4

set -euo pipefail
source .venv-olmo/bin/activate
export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

model="causal/models/Qwen2.5-7B"
data="causal/data/qwen_learned_multipass_shared_seed2025.jsonl"
output="causal/results/qwen2.5-7b_learned_multipass_shared"

python -m src.srg_experiment.train_refusal \
  --model "$model" \
  --data "$data" \
  --output "$output" \
  --target-mode final_response \
  --epochs 2 \
  --global-batch-size 64 \
  --learning-rate 1e-4 \
  --seed 2025

python -m causal.evaluate_learned_multipass \
  --base-model "$model" \
  --adapter "$output/final_adapter" \
  --coca-data causal/COCA/data \
  --xstest artifacts/datasets/xstest/xstest_prompts.csv \
  --base-results causal/results/qwen2.5-7b_no_sft/judged.jsonl \
  --safe-results causal/results/qwen2.5-7b_principle_reasoning_pilot_eval/judged.jsonl \
  --output "${output}_eval"
