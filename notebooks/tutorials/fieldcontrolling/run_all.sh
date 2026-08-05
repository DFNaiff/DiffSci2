#!/usr/bin/env bash
# Execute the whole tutorial series in order, in place.
#
# WARNING: this uses `nbconvert --inplace`, which OVERWRITES the .ipynb files and
# any outputs you produced interactively. Do not run it while you have the
# notebooks open in Jupyter. For surgical edits that preserve your work, use
# apply_fixes.py instead.
#
#   conda activate ddpm_env
#   cd notebooks/tutorials/fieldcontrolling
#   bash run_all.sh              # all five notebooks
#   bash run_all.sh 03 04        # only these
#
# Each notebook skips its training step if the checkpoint already exists, so
# re-running is cheap. Logs go to fcdata/logs/.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p fcdata/logs

NOTEBOOKS=(
  00-vae-training-2d
  01-porosity-field-and-gaussian-process
  02-scalar-porosity-pretraining
  03-field-controlled-training
  04-field-controlled-inference
  05-chunk-decoding
)

if [ "$#" -gt 0 ]; then
  SELECTED=()
  for want in "$@"; do
    for nb in "${NOTEBOOKS[@]}"; do
      [[ "$nb" == "$want"* ]] && SELECTED+=("$nb")
    done
  done
  NOTEBOOKS=("${SELECTED[@]}")
fi

for nb in "${NOTEBOOKS[@]}"; do
  echo "============================================================"
  echo "  ${nb}   $(date '+%H:%M:%S')"
  echo "============================================================"
  start=$SECONDS
  jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    "${nb}.ipynb" 2>&1 | tee "fcdata/logs/${nb}.log"
  echo "  -> done in $(( (SECONDS - start) / 60 ))m $(( (SECONDS - start) % 60 ))s"
  echo
done

echo "All notebooks executed. Figures in fcdata/figures/."
