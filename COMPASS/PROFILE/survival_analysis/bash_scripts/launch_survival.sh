#!/bin/bash
# Launch wrapper for array_survival_run.sh.
#
# Reads the model manifest, computes the SLURM array size, and submits the job.
# Environment variables accepted here are forwarded to the array script.
#
# Usage:
#   bash COMPASS/PROFILE/survival_analysis/bash_scripts/launch_survival.sh
#   ROWS_PER_TASK=2 bash COMPASS/PROFILE/survival_analysis/bash_scripts/launch_survival.sh
#   MANIFEST=/path/to/tasks.tsv bash COMPASS/PROFILE/survival_analysis/bash_scripts/launch_survival.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/CAIA}"
SURVIVAL_DIR="$PROJECT_ROOT/COMPASS/PROFILE/survival_analysis"
MANIFEST="${MANIFEST:-$SURVIVAL_DIR/bash_scripts/slurm_manifests/survival_tasks.tsv}"
ROWS_PER_TASK="${ROWS_PER_TASK:-1}"
INPUTS_DIR="${INPUTS_DIR:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/prediction_inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Error: manifest not found: $MANIFEST"
  exit 1
fi

N_ROWS=$(wc -l < "$MANIFEST")
if [[ "$N_ROWS" -eq 0 ]]; then
  echo "Error: manifest is empty: $MANIFEST"
  exit 1
fi

N_TASKS=$(( (N_ROWS + ROWS_PER_TASK - 1) / ROWS_PER_TASK ))
MAX_TASK=$(( N_TASKS - 1 ))

echo "Manifest:      $MANIFEST"
echo "Rows:          $N_ROWS"
echo "Rows per task: $ROWS_PER_TASK"
echo "Array tasks:   $N_TASKS  (--array=0-${MAX_TASK})"
echo "Inputs dir:    $INPUTS_DIR"
echo "Output dir:    $OUTPUT_DIR"

mkdir -p \
  "$SURVIVAL_DIR/bash_scripts/array_survival_run/output" \
  "$SURVIVAL_DIR/bash_scripts/array_survival_run/error"

cd "$PROJECT_ROOT"

sbatch \
  --array="0-${MAX_TASK}" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MANIFEST="$MANIFEST",ROWS_PER_TASK="$ROWS_PER_TASK",INPUTS_DIR="$INPUTS_DIR",OUTPUT_DIR="$OUTPUT_DIR" \
  "$SURVIVAL_DIR/bash_scripts/array_survival_run.sh"
