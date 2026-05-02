#!/bin/bash

#SBATCH --job-name=profile_surv
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-0%1
#SBATCH --output=COMPASS/PROFILE/survival_analysis/bash_scripts/array_survival_run/output/%A_%a.out
#SBATCH --error=COMPASS/PROFILE/survival_analysis/bash_scripts/array_survival_run/error/%A_%a.err

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/CAIA}
SURVIVAL_DIR="$PROJECT_ROOT/COMPASS/PROFILE/survival_analysis"
MANIFEST=${MANIFEST:-$SURVIVAL_DIR/bash_scripts/slurm_manifests/survival_tasks.tsv}
ROWS_PER_TASK=${ROWS_PER_TASK:-1}
INPUTS_DIR=${INPUTS_DIR:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/prediction_inputs}
OUTPUT_DIR=${OUTPUT_DIR:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis}

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project root not found: $PROJECT_ROOT"
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p \
  COMPASS/PROFILE/survival_analysis/bash_scripts/array_survival_run/output \
  COMPASS/PROFILE/survival_analysis/bash_scripts/array_survival_run/error \
  "$OUTPUT_DIR"

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate survlatent_ode || { echo "Failed to activate conda env survlatent_ode"; exit 1; }

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TOTAL_ROWS=$(wc -l < "$MANIFEST")
START_LINE=$((SLURM_ARRAY_TASK_ID * ROWS_PER_TASK + 1))
END_LINE=$((START_LINE + ROWS_PER_TASK - 1))

if [[ "$START_LINE" -gt "$TOTAL_ROWS" ]]; then
  echo "No rows assigned to task ${SLURM_ARRAY_TASK_ID} (start=${START_LINE}, total=${TOTAL_ROWS})"
  exit 0
fi
if [[ "$END_LINE" -gt "$TOTAL_ROWS" ]]; then
  END_LINE="$TOTAL_ROWS"
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: processing manifest rows ${START_LINE}-${END_LINE}"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "INPUTS_DIR=$INPUTS_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"

FAILED=0

for LINE_NUM in $(seq "$START_LINE" "$END_LINE"); do
  TASK_LINE=$(sed -n "${LINE_NUM}p" "$MANIFEST")
  if [[ -z "${TASK_LINE}" ]]; then
    echo "Skipping empty manifest row ${LINE_NUM}"
    continue
  fi

  IFS=$'\t' read -r MODEL LANDMARK CONFIG EXTRA_FIELD <<< "$TASK_LINE"
  if [[ -n "${EXTRA_FIELD:-}" ]]; then
    echo "Unsupported manifest row ${LINE_NUM}: expected 3 tab-separated fields, got more"
    exit 1
  fi

  MODEL=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
  echo "Running row ${LINE_NUM}: model=${MODEL}, landmark=${LANDMARK}, config=${CONFIG}"
  ROW_OUTPUT_DIR="$OUTPUT_DIR/slurm_runs/${MODEL}/landmark_${LANDMARK}/${CONFIG}"
  mkdir -p "$ROW_OUTPUT_DIR"

  case "$MODEL" in
    cox)
      if ! python "$SURVIVAL_DIR/cox_aggregated.py" \
        --inputs-dir "$INPUTS_DIR" \
        --output-dir "$ROW_OUTPUT_DIR" \
        --landmark-days "$LANDMARK" \
        --analysis "$CONFIG" \
        --endpoints platinum death \
        --n-folds "${N_FOLDS:-5}"
      then
        echo "[error] row ${LINE_NUM} failed: model=${MODEL}, landmark=${LANDMARK}, config=${CONFIG}"
        FAILED=1
      fi
      ;;
    xgboost|xgb)
      if ! python "$SURVIVAL_DIR/landmark_xgboost.py" \
        --inputs-dir "$INPUTS_DIR" \
        --output-dir "$ROW_OUTPUT_DIR" \
        --landmark-days "$LANDMARK" \
        --endpoints platinum death \
        --n-folds "${N_FOLDS:-5}"
      then
        echo "[error] row ${LINE_NUM} failed: model=${MODEL}, landmark=${LANDMARK}, config=${CONFIG}"
        FAILED=1
      fi
      ;;
    deephit|dynamic_deephit)
      if ! python "$SURVIVAL_DIR/dynamic_deephit.py" \
        --inputs-dir "$INPUTS_DIR" \
        --output-dir "$ROW_OUTPUT_DIR" \
        --landmark-day "$LANDMARK" \
        --config "$CONFIG" \
        --n-folds "${N_FOLDS:-5}" \
        ${CUDA_FLAG:-}
      then
        echo "[error] row ${LINE_NUM} failed: model=${MODEL}, landmark=${LANDMARK}, config=${CONFIG}"
        FAILED=1
      fi
      ;;
    *)
      echo "Unsupported model in manifest row ${LINE_NUM}: $MODEL"
      exit 1
      ;;
  esac
done

conda deactivate
exit "$FAILED"
