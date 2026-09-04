#!/bin/bash -l
#$ -N perm_methods
#$ -P buinlp
#$ -cwd
#$ -j y
#$ -o jobs/perm_methods.$JOB_ID.log
#$ -l h_rt=03:00:00
#$ -l mem_per_core=4G
#$ -pe omp 1

set -euo pipefail
export PYTHONPATH=toy/arithmetic/within_task
RUN_NAME="matched_methods_${JOB_ID}"

python toy/arithmetic/within_task/matched_methods.py \
  --output "toy/arithmetic/within_task/results/${RUN_NAME}"
python toy/arithmetic/within_task/plot_matched_methods.py \
  "toy/arithmetic/within_task/results/${RUN_NAME}/summary.json" \
  --output "toy/arithmetic/within_task/figures/${RUN_NAME}"
