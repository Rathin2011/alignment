#!/bin/bash -l
#$ -N safety_methods
#$ -P buinlp
#$ -cwd
#$ -j y
#$ -o jobs/safety_methods.$JOB_ID.log
#$ -l h_rt=03:00:00
#$ -l mem_per_core=4G
#$ -pe omp 1

set -euo pipefail
export PYTHONPATH=.
RUN_NAME="matched_methods_${JOB_ID}"

python toy/safety/matched_methods.py \
  --output "toy/safety/results/${RUN_NAME}"
python toy/arithmetic/within_task/plot_matched_methods.py \
  "toy/safety/results/${RUN_NAME}/summary.json" \
  --output "toy/safety/figures/${RUN_NAME}" \
  --task-label "toy safety OOD"
