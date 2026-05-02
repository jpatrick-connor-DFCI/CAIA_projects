#!/bin/bash
# Launch wrapper for build_prediction_inputs.sh.
#
# Usage:
#   bash COMPASS/PROFILE/survival_analysis/bash_scripts/launch_build_prediction_inputs.sh
#   LANDMARK_DAYS="0 90" bash COMPASS/PROFILE/survival_analysis/bash_scripts/launch_build_prediction_inputs.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/CAIA}"
SURVIVAL_DIR="$PROJECT_ROOT/COMPASS/PROFILE/survival_analysis"
INPUTS_DIR="${INPUTS_DIR:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/prediction_inputs}"

mkdir -p \
  "$SURVIVAL_DIR/bash_scripts/build_prediction_inputs/output" \
  "$SURVIVAL_DIR/bash_scripts/build_prediction_inputs/error" \
  "$INPUTS_DIR"

echo "Submitting prediction-input build"
echo "Project root: $PROJECT_ROOT"
echo "Inputs dir:   $INPUTS_DIR"

cd "$PROJECT_ROOT"

sbatch \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",INPUTS_DIR="$INPUTS_DIR" \
  "$SURVIVAL_DIR/bash_scripts/build_prediction_inputs.sh"
