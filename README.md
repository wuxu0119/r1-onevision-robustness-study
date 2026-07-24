# R1-Onevision Robustness Study

Reproducibility and robustness study of multimodal reasoning with
R1-Onevision. This project evaluates when the released models fail under
controlled visual perturbations and tests a lightweight inference-time
failure detector.

- Repository: https://github.com/wuxu0119/r1-onevision-robustness-study
- Author: Xu Wu
- Course: Visual Media

## Target paper

Yi Yang et al., "R1-Onevision: Advancing Generalized Multimodal Reasoning
through Cross-Modal Formalization," ICCV 2025.

- Paper: https://openaccess.thecvf.com/content/ICCV2025/html/Yang_R1-Onevision_Advancing_Generalized_Multimodal_Reasoning_through_Cross-Modal_Formalization_ICCV_2025_paper.html
- Official code: https://github.com/Fancy-MLLM/R1-Onevision
- Benchmark: https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision-Bench
- SFT checkpoint: https://huggingface.co/Fancy-MLLM/R1-Onevision-7B
- RL checkpoint: https://huggingface.co/Fancy-MLLM/R1-Onevision-7B-RL

## Questions

1. Can the reported gain from rule-based RL over Qwen2.5-VL-7B be reproduced?
2. Are the released checkpoints stable under semantics-preserving visual
   changes?
3. Can prompt disagreement detect unreliable answers without additional
   training?

## Experiment design

- Reproduction: all 942 R1-Onevision-Bench questions, evaluated with the base,
  SFT, and RL checkpoints.
- Robustness: a seed-fixed stratified subset of 300 questions.
- Failure condition A: downsample to 50% or 25%, then bilinear resize to the
  original size.
- Failure condition B: compose the target with one or three unrelated
  benchmark images.
- Improvement: run the paper-format prompt and an evidence-first prompt on the
  same global image. Answer only when the normalized final answers agree;
  otherwise abstain and send the case to manual review.
- Exploratory ablation: global image plus four overlapping quadrants. This
  exceeded 40GB on some inputs and is reported as a resource limitation rather
  than scored as an incorrect answer.

## Main results

| Model | Accuracy on 942 questions |
| --- | ---: |
| Qwen2.5-VL-7B-Instruct | 33.44% |
| R1-Onevision-7B (SFT) | 34.71% |
| R1-Onevision-7B-RL | 37.37% |

The paired RL-minus-base difference is +3.93 percentage points (10,000-repeat
paired bootstrap 95% CI: +0.42 to +7.54; exact McNemar p=0.036). The paper
reports a +4.1-point difference (32.1% to 36.2%).

For SFT, one unrelated panel reduces accuracy from 39.0% to 29.7% on the
300-question subset. Under the dual-prompt agreement gate, selective accuracy
is 46.6%-51.0% at 40.7%-47.7% coverage across the three improvement
conditions.

## Environment

- Ubuntu 20.04.5
- 8 x NVIDIA A100 PCIe 40GB
- Python 3.11.15
- PyTorch 2.6.0+cu124
- Transformers 4.49.0
- BF16 with PyTorch SDPA

## Setup

```bash
conda create -n r1ov python=3.11 pip -y
conda activate r1ov
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Prepare the benchmark metadata:

```bash
python scripts/01_prepare_manifest.py --config configs/experiment.json
```

Run one smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/00_smoke_test.py \
  --model-id Fancy-MLLM/R1-Onevision-7B-RL
```

Run all resumable 8-GPU phases:

```bash
bash scripts/run_all_gpu.sh
```

To retry only records whose previous attempt has a non-empty `error` field:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/02_run_inference.py \
  --config configs/experiment.json \
  --phase improvement \
  --model-key rl \
  --shard-id 0 \
  --num-shards 8 \
  --retry-errors
```

## Output layout

```text
configs/                  experiment definitions
data/                     benchmark cache and derived images (not committed)
notes/                    paper and protocol notes
report/                   concise aggregate-results documentation
scripts/                  inference, scoring, analysis, and review utilities
src/r1eval/               reusable implementation
tests/                    unit tests
results/raw/              resumable JSONL generations
results/scored/           deterministic scores
results/analysis/         summary tables and failure-detector metrics
results/review/           local qualitative-review page
```

Only small, verified summary tables and unit-test fixtures are tracked in this
repository. Benchmark data, model weights, raw generations, and scored
per-sample outputs are intentionally excluded.

## Evaluation caveats

The paper uses GPT-4o-mini to extract and judge answers. This repository uses a
deterministic local scorer to avoid a proprietary API, so absolute accuracy can
differ, especially for the 159 free-form questions. Model comparisons use
paired samples. Per-record CUDA errors are kept in the JSONL for auditability
and are reported separately; they must not be silently interpreted as ordinary
wrong answers.

## License and data

This repository contains evaluation code and small derived summaries. Model
weights and R1-Onevision-Bench are downloaded from their official sources and
retain their original licenses.

## Citation

This repository evaluates the following work:

```bibtex
@inproceedings{Yang_2025_ICCV,
  author    = {Yang, Yi and He, Xiaoxuan and Pan, Hongkun and Jiang, Xiyan
               and Deng, Yan and Yang, Xingtao and Lu, Haoyu and Yin, Dacheng
               and Rao, Fengyun and Zhu, Minfeng and Zhang, Bo and Chen, Wei},
  title     = {R1-Onevision: Advancing Generalized Multimodal Reasoning
               through Cross-Modal Formalization},
  booktitle = {Proceedings of the IEEE/CVF International Conference on
               Computer Vision (ICCV)},
  month     = {October},
  year      = {2025},
  pages     = {2376--2385}
}
```
