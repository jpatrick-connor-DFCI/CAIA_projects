#!/bin/bash

#SBATCH --job-name=profile_v3_llm
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=COMPASS/v3/bash_scripts/run_v3_pipeline/output/%j.out
#SBATCH --error=COMPASS/v3/bash_scripts/run_v3_pipeline/error/%j.err

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/CAIA}
V3_DIR="$PROJECT_ROOT/COMPASS/v3"
OUTPUT_DIR=${OUTPUT_DIR:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/v3_outputs}
MRN_FILE=${MRN_FILE:-/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/mrn_lists/longitudinal_mrns.csv}
NOTE_BUNDLE_PATH=${NOTE_BUNDLE_PATH:-$OUTPUT_DIR/LLM_v3_prostate_note_bundle.json.gz}
MODEL=${MODEL:-gpt-4o}
MAX_WORKERS=${MAX_WORKERS:-8}
MAX_RETRIES=${MAX_RETRIES:-3}
MAX_NOTES_PER_PATIENT=${MAX_NOTES_PER_PATIENT:-30}
LIMIT_MRNS=${LIMIT_MRNS:-}
OVERWRITE=${OVERWRITE:-0}

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project root not found: $PROJECT_ROOT"
  exit 1
fi
if [[ ! -f "$MRN_FILE" ]]; then
  echo "MRN file not found: $MRN_FILE"
  exit 1
fi
if [[ ! -f "$NOTE_BUNDLE_PATH" ]]; then
  echo "Note bundle not found: $NOTE_BUNDLE_PATH"
  echo "Compile it first with COMPASS/v3/compile_prostate_note_bundle.py"
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p \
  COMPASS/v3/bash_scripts/run_v3_pipeline/output \
  COMPASS/v3/bash_scripts/run_v3_pipeline/error \
  "$OUTPUT_DIR"

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate DFCI_GPT_LLM || { echo "Failed to activate conda env DFCI_GPT_LLM"; exit 1; }

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running PROFILE v3 LLM pipeline"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  MRN_FILE=$MRN_FILE"
echo "  NOTE_BUNDLE_PATH=$NOTE_BUNDLE_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  MODEL=$MODEL"
echo "  MAX_WORKERS=$MAX_WORKERS"
echo "  MAX_NOTES_PER_PATIENT=$MAX_NOTES_PER_PATIENT"

EXTRA_ARGS=()
if [[ -n "$LIMIT_MRNS" ]]; then
  EXTRA_ARGS+=(--limit-mrns "$LIMIT_MRNS")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

python "$V3_DIR/run_v3_pipeline.py" \
  --mrn-file "$MRN_FILE" \
  --note-bundle-path "$NOTE_BUNDLE_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --max-workers "$MAX_WORKERS" \
  --max-retries "$MAX_RETRIES" \
  --max-notes-per-patient "$MAX_NOTES_PER_PATIENT" \
  "${EXTRA_ARGS[@]}"

conda deactivate