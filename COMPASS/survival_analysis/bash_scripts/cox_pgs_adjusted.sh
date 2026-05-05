#!/bin/bash

#SBATCH --job-name=profile_cox_pgs_adjusted
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=COMPASS/survival_analysis/bash_scripts/cox_pgs_adjusted/output/%j.out
#SBATCH --error=COMPASS/survival_analysis/bash_scripts/cox_pgs_adjusted/error/%j.err

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/CAIA}
SURVIVAL_DIR="$PROJECT_ROOT/COMPASS/survival_analysis"
AGGREGATED_PATTERN=${AGGREGATED_PATTERN:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/prediction_inputs/aggregated_landmark{landmark}.csv}
GERMLINE_PATH=${GERMLINE_PATH:-/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/clinical_and_genomic_features/complete_germline_data_df.csv.gz}
OUTPUT_DIR=${OUTPUT_DIR:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis}
LANDMARKS=${LANDMARKS:-"0 90"}
ENDPOINTS=${ENDPOINTS:-"platinum death"}

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project root not found: $PROJECT_ROOT"
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p \
  COMPASS/survival_analysis/bash_scripts/cox_pgs_adjusted/output \
  COMPASS/survival_analysis/bash_scripts/cox_pgs_adjusted/error

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate survlatent_ode || { echo "Failed to activate conda env survlatent_ode"; exit 1; }

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running PGS-adjusted univariate Cox"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  AGGREGATED_PATTERN=$AGGREGATED_PATTERN"
echo "  GERMLINE_PATH=$GERMLINE_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  LANDMARKS=$LANDMARKS"
echo "  ENDPOINTS=$ENDPOINTS"

python "$SURVIVAL_DIR/cox_pgs_adjusted.py" \
  --aggregated-csv-pattern "$AGGREGATED_PATTERN" \
  --germline-path "$GERMLINE_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --landmarks $LANDMARKS \
  --endpoints $ENDPOINTS

conda deactivate