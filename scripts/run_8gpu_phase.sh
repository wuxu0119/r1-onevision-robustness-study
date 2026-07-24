#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_8gpu_phase.sh <phase> <model-key>"
  exit 2
fi

phase="$1"
model_key="$2"
num_gpus=8

mkdir -p logs
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

pids=()
for gpu_id in $(seq 0 $((num_gpus - 1))); do
  log_path="logs/${phase}__${model_key}__gpu${gpu_id}.log"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python scripts/02_run_inference.py \
      --config configs/experiment.json \
      --phase "${phase}" \
      --model-key "${model_key}" \
      --shard-id "${gpu_id}" \
      --num-shards "${num_gpus}" \
      >"${log_path}" 2>&1 &
  pids+=("$!")
  echo "started GPU ${gpu_id}, log: ${log_path}"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "At least one shard failed. Inspect logs before retrying."
  exit 1
fi
echo "completed phase=${phase} model=${model_key}"

