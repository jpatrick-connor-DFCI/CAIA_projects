#!/bin/bash
# Launch wrapper for build_genomic_inputs.sh.
#
# Usage:
#   bash COMPASS/survival_analysis/bash_scripts/launch_build_genomic_inputs.sh
#   SOMATIC_PATH=/path/to/somatic.csv.gz bash COMPASS/survival_analysis/bash_scripts/launch_build_genomic_inputs.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/CAIA}"
SURVIVAL_DIR="$PROJECT_ROOT/COMPASS/survival_analysis"
INPUTS_DIR="${INPUTS_DIR:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/prediction_inputs}"

mkdir -p \
  "$SURVIVAL_DIR/bash_scripts/build_genomic_inputs/output" \
  "$SURVIVAL_DIR/bash_scripts/build_genomic_inputs/error" \
  "$INPUTS_DIR/genomic"

echo "Submitting genomic-input build"
echo "Project root: $PROJECT_ROOT"
echo "Inputs dir:   $INPUTS_DIR (genomic outputs in <INPUTS_DIR>/genomic)"

cd "$PROJECT_ROOT"

sbatch \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",INPUTS_DIR="$INPUTS_DIR" \
  "$SURVIVAL_DIR/bash_scripts/build_genomic_inputs.sh"
