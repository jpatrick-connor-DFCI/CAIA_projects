#!/bin/bash

#SBATCH --job-name=profile_genomic_inputs
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=COMPASS/survival_analysis/bash_scripts/build_genomic_inputs/output/%j.out
#SBATCH --error=COMPASS/survival_analysis/bash_scripts/build_genomic_inputs/error/%j.err

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/CAIA}
SURVIVAL_DIR="$PROJECT_ROOT/COMPASS/survival_analysis"
DATA=${DATA:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/longitudinal_prediction_data.csv}
V3_LABELS_PATH=${V3_LABELS_PATH:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/v3_outputs/LLM_v3_labels.tsv}
SOMATIC_PATH=${SOMATIC_PATH:-/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/clinical_and_genomic_features/complete_somatic_data_df.csv.gz}
INPUTS_DIR=${INPUTS_DIR:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/prediction_inputs}
TIME_UNIT_DAYS=${TIME_UNIT_DAYS:-7}
MIN_PATIENT_COVERAGE=${MIN_PATIENT_COVERAGE:-0.20}

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project root not found: $PROJECT_ROOT"
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p \
  COMPASS/survival_analysis/bash_scripts/build_genomic_inputs/output \
  COMPASS/survival_analysis/bash_scripts/build_genomic_inputs/error \
  "$INPUTS_DIR/genomic"

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate survlatent_ode || { echo "Failed to activate conda env survlatent_ode"; exit 1; }

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Building genomic prediction inputs"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  DATA=$DATA"
echo "  V3_LABELS_PATH=$V3_LABELS_PATH"
echo "  SOMATIC_PATH=$SOMATIC_PATH"
echo "  INPUTS_DIR=$INPUTS_DIR (genomic outputs in <INPUTS_DIR>/genomic)"

python "$SURVIVAL_DIR/build_genomic_inputs.py" \
  --data "$DATA" \
  --v3-labels-path "$V3_LABELS_PATH" \
  --somatic-path "$SOMATIC_PATH" \
  --inputs-dir "$INPUTS_DIR" \
  --time-unit-days "$TIME_UNIT_DAYS" \
  --min-patient-coverage "$MIN_PATIENT_COVERAGE"

conda deactivate
