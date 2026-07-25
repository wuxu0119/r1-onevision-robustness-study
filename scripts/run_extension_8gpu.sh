#!/usr/bin/env bash
set -euo pipefail

# The extension is resumable. Re-running skips records whose full
# (row, condition, prompt, image mode, replicate) key is already present.

config="configs/extension.json"

if pgrep -u "${USER}" -f "scripts/02_run_inference.py" >/dev/null; then
  echo "An inference process owned by ${USER} is already running. Stop here and inspect it."
  exit 1
fi

gpu_count="$(nvidia-smi --list-gpus | wc -l)"
if [[ "${gpu_count}" -lt 8 ]]; then
  echo "This pipeline expects 8 visible GPUs, but found ${gpu_count}."
  exit 1
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  2>/dev/null | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo "At least one GPU already has a compute process. Do not overlap jobs."
  exit 1
fi

run_and_retry() {
  local phase="$1"
  local model_key="$2"
  bash scripts/run_8gpu_phase.sh "${phase}" "${model_key}" "${config}"
  bash scripts/run_8gpu_phase.sh \
    "${phase}" "${model_key}" "${config}" --retry-errors
}

for model_key in base sft rl; do
  run_and_retry controls "${model_key}"
done

for model_key in base sft rl; do
  run_and_retry gate_generalization "${model_key}"
done

run_and_retry heldout_gate rl
run_and_retry confidence rl
run_and_retry multiview_budgeted rl

python scripts/03_score_results.py \
  --results-dir results/raw \
  --output-dir results/scored
python scripts/07_analyze_extension.py \
  --scored-dir results/scored \
  --output-dir results/analysis/extension

tar -czf r1_onevision_extension_results.tar.gz \
  configs/extension.json \
  results/raw/controls__*.jsonl \
  results/raw/gate_generalization__*.jsonl \
  results/raw/heldout_gate__*.jsonl \
  results/raw/confidence__*.jsonl \
  results/raw/multiview_budgeted__*.jsonl \
  results/scored/controls__*.csv \
  results/scored/gate_generalization__*.csv \
  results/scored/heldout_gate__*.csv \
  results/scored/confidence__*.csv \
  results/scored/multiview_budgeted__*.csv \
  results/analysis/extension \
  logs/controls__*.log \
  logs/gate_generalization__*.log \
  logs/heldout_gate__*.log \
  logs/confidence__*.log \
  logs/multiview_budgeted__*.log

echo "extension experiments completed: r1_onevision_extension_results.tar.gz"
