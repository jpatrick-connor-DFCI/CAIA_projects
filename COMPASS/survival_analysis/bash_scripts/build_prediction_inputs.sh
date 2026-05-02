#!/bin/bash

#SBATCH --job-name=profile_surv_inputs
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=COMPASS/PROFILE/survival_analysis/bash_scripts/build_prediction_inputs/output/%j.out
#SBATCH --error=COMPASS/PROFILE/survival_analysis/bash_scripts/build_prediction_inputs/error/%j.err

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/CAIA}
SURVIVAL_DIR="$PROJECT_ROOT/COMPASS/PROFILE/survival_analysis"
DATA=${DATA:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/longitudinal_prediction_data.csv}
V3_LABELS_PATH=${V3_LABELS_PATH:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/v3_outputs/LLM_v3_labels.tsv}
INPUTS_DIR=${INPUTS_DIR:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/prediction_inputs}
LANDMARK_DAYS=${LANDMARK_DAYS:-"0 90"}
TIME_UNIT_DAYS=${TIME_UNIT_DAYS:-7}
TEST_FRAC=${TEST_FRAC:-0.20}
VAL_FRAC=${VAL_FRAC:-0.20}
MIN_PATIENT_COVERAGE=${MIN_PATIENT_COVERAGE:-0.20}

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project root not found: $PROJECT_ROOT"
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p \
  COMPASS/PROFILE/survival_analysis/bash_scripts/build_prediction_inputs/output \
  COMPASS/PROFILE/survival_analysis/bash_scripts/build_prediction_inputs/error \
  "$INPUTS_DIR"

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate survlatent_ode || { echo "Failed to activate conda env survlatent_ode"; exit 1; }

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Building prediction inputs"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  DATA=$DATA"
echo "  V3_LABELS_PATH=$V3_LABELS_PATH"
echo "  INPUTS_DIR=$INPUTS_DIR"
echo "  LANDMARK_DAYS=$LANDMARK_DAYS"

python "$SURVIVAL_DIR/build_prediction_inputs.py" \
  --data "$DATA" \
  --v3-labels-path "$V3_LABELS_PATH" \
  --output-dir "$INPUTS_DIR" \
  --landmark-days $LANDMARK_DAYS \
  --time-unit-days "$TIME_UNIT_DAYS" \
  --test-frac "$TEST_FRAC" \
  --val-frac "$VAL_FRAC" \
  --min-patient-coverage "$MIN_PATIENT_COVERAGE"

conda deactivate
