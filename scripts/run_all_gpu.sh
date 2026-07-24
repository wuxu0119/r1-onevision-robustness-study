#!/usr/bin/env bash
set -euo pipefail

# Every call is resumable. Re-running this script skips completed records.

bash scripts/run_8gpu_phase.sh reproduce base
bash scripts/run_8gpu_phase.sh reproduce sft
bash scripts/run_8gpu_phase.sh reproduce rl

bash scripts/run_8gpu_phase.sh robustness base
bash scripts/run_8gpu_phase.sh robustness sft
bash scripts/run_8gpu_phase.sh robustness rl

bash scripts/run_8gpu_phase.sh improvement rl

python scripts/03_score_results.py \
  --results-dir results/raw \
  --output-dir results/scored
python scripts/04_analyze_results.py \
  --scored-dir results/scored \
  --output-dir results/analysis
python scripts/05_make_figures.py \
  --analysis-dir results/analysis \
  --output-dir results/figures
python scripts/06_build_review_sheet.py \
  --input results/analysis/manual_error_annotations.csv \
  --output-dir results/review \
  --limit 100

tar -czf r1_onevision_results.tar.gz results logs configs/experiment.json
echo "all experiments completed: r1_onevision_results.tar.gz"

