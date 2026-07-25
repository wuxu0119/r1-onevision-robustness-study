# R1-Onevision Reproducibility and Measurement Study

This repository contains the evaluation code and compact, verified summary
tables for a reproducibility and measurement study of R1-Onevision. The study
asks whether checkpoint rankings and multimodal-reasoning conclusions remain
stable when the answer parser, visual composition, prompt, and inference budget
are controlled.

- Repository: https://github.com/wuxu0119/r1-onevision-robustness-study
- Author: Xu Wu
- Target paper: Yi Yang et al., “R1-Onevision: Advancing Generalized
  Multimodal Reasoning through Cross-Modal Formalization,” ICCV 2025

The repository is limited to source code, protocol notes, tests, and compact
aggregate summaries. Manuscript files, raw model generations, model weights,
and benchmark images are not included.

## Target resources

- [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_R1-Onevision_Advancing_Generalized_Multimodal_Reasoning_through_Cross-Modal_Formalization_ICCV_2025_paper.html)
- [Official code](https://github.com/Fancy-MLLM/R1-Onevision)
- [R1-Onevision-Bench](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision-Bench)
- [SFT checkpoint](https://huggingface.co/Fancy-MLLM/R1-Onevision-7B)
- [RL checkpoint](https://huggingface.co/Fancy-MLLM/R1-Onevision-7B-RL)
- [Qwen2.5-VL-7B-Instruct baseline](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

## Research questions

1. Are clean-benchmark differences among the Base, SFT, and RL checkpoints
   reproducible under a deterministic local scorer?
2. How stable are their answers under task-label-preserving changes in
   resolution and image composition?
3. How much of an apparent distractor effect remains after target scale,
   canvas size, panel type, position, and random seed are controlled?
4. Can answer agreement support selective prediction, and how does a
   dual-prompt gate compare with same-prompt stochastic consistency?
5. Does a near-pixel-budget-matched multi-view input improve accuracy without
   reintroducing the original out-of-memory failure?

## Experiment scope

The core archive contains 10,926 rows:

- clean reproduction on all 942 questions for Base, SFT, and RL;
- a paired 300-question stress test for all three checkpoints under clean,
  50% and 25% downsampling, and one or three unrelated panels;
- an initial RL-only prompt and multi-view study.

The extension adds 26,142 rows:

- fixed-canvas controls for target-only resizing, blank panels, noise panels,
  left/right target positions, and three semantic-distractor seeds;
- evidence-first prompting for all three checkpoints and all five visual
  conditions;
- a clean evidence-first pass over all 942 questions, with the 642 questions
  outside the prompt-development subset reserved for held-out evaluation;
- eight stochastic RL generations per item under clean, 25% downsampling, and
  three-distractor conditions;
- a budgeted comparison between one 864×864 global view and five 384×384
  views.

All 26,142 extension rows completed without a runtime error. The initial
unbudgeted five-view branch produced 680 CUDA out-of-memory records; those
records are retained as resource failures and are not interpreted as ordinary
wrong answers.

## Scorer repair

An audit found that the original regular expression could miss an explicit
answer split across lines, such as `The correct answer is:` followed by
`\boxed{B}`, while a loose tail fallback could also accept a non-final letter.
The repaired multiple-choice extractor applies the following precedence:

1. the final `<answer>...</answer>` block;
2. the latest explicit final/correct answer phrase or direct boxed choice;
3. a conservative bare or leading choice;
4. otherwise manual review.

Thirteen adversarial regression tests cover line-broken boxes, conflicting
earlier and later answers, Markdown labels, option-list decoys, multi-answer
rejection, and unchanged numeric/free-response behavior. The repair was
applied uniformly to both archives without overwriting the original raw data.

## Main findings after uniform rescoring

| Checkpoint | Clean accuracy (942) | 95% bootstrap CI |
| --- | ---: | ---: |
| Qwen2.5-VL-7B-Instruct | 36.94% | [33.86, 40.02] |
| R1-Onevision-7B (SFT) | 33.97% | [31.00, 37.05] |
| R1-Onevision-7B-RL | 37.26% | [34.18, 40.34] |

The repaired RL-minus-Base difference is +0.32 percentage points (paired
bootstrap 95% CI [−3.18, +3.82], exact McNemar *p*=0.906). The study therefore
does not claim a significant RL-over-Base advantage under this local scoring
protocol.

On the complete 3×5 robustness matrix, none of the 12 clean-to-perturbed paired
comparisons survives Holm correction. Accuracy alone hides substantial answer
turnover: clean/perturbed answer agreement ranges from 37.7% to 56.3%.

The fixed-canvas study changes the distractor interpretation. For SFT, mean
semantic-panel accuracy is 6.83 points below clean input (95% CI
[−11.56, −2.28]), but only 3.00 points below a matched blank-panel control
(95% CI [−6.44, +0.50]). Thus, layout and target-scale changes explain part of
the apparent distractor penalty, while the isolated semantic effect remains
uncertain.

On the held-out 642-question clean complement, the RL dual-prompt agreement
gate accepts 288 items (44.86% coverage) and reaches 48.26% selective accuracy
(95% CI [42.46, 54.04]), versus 39.25% accuracy before abstention. This is a
useful selective-prediction trade-off, but not a fully established failure
detector: coverage is below one half, and same-prompt stochastic agreement also
selects higher-accuracy subsets.

The budgeted multi-view experiment finishes all 3,600 generations without an
out-of-memory error, but provides no accuracy gain. Averaged over six matched
condition/prompt groups, accuracy is 33.89% for the global view and 33.33% for
five views; every paired 95% interval includes zero.

Full tables, confidence intervals, paired flip counts, and interpretation
boundaries are in [`RESULTS.md`](RESULTS.md).

## Environment

- Ubuntu 20.04.5
- 8 × NVIDIA A100 PCIe 40 GB
- Python 3.11.15
- PyTorch 2.6.0+cu124
- Transformers 4.49.0
- BF16 inference with PyTorch SDPA

## Setup

```bash
conda create -n r1ov python=3.11 pip -y
conda activate r1ov
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Prepare the benchmark manifest and run a one-GPU smoke test:

```bash
python scripts/01_prepare_manifest.py --config configs/experiment.json
CUDA_VISIBLE_DEVICES=0 python scripts/00_smoke_test.py \
  --model-id Fancy-MLLM/R1-Onevision-7B-RL
```

Run the resumable core and extension pipelines:

```bash
bash scripts/run_all_gpu.sh
bash scripts/run_extension_8gpu.sh
```

The runners start one shard per visible GPU, preserve per-record errors, and
skip completed keys on restart. Before launching, verify that the selected GPUs
are unused and follow the local resource-sharing policy.

Rescore existing raw JSONL files with the repaired extractor and regenerate the
summary tables:

```bash
python scripts/03_score_results.py \
  --results-dir results/raw \
  --output-dir results/scored
python scripts/04_analyze_results.py \
  --scored-dir results/scored \
  --output-dir results/analysis
python scripts/07_analyze_extension.py \
  --scored-dir results/scored \
  --output-dir results/analysis/extension
```

## Manual error analysis

The repository can create a fixed-seed annotation sheet, but no completed
human taxonomy is claimed:

```bash
python scripts/08_sample_manual_errors.py
python scripts/06_build_review_sheet.py \
  --input results/analysis/manual_error_sample_100.csv
# After independent labels and adjudication:
python scripts/09_summarize_annotations.py
```

Annotation columns are blank by design until human labels are supplied.

## Repository layout

```text
configs/                  core and extension experiment definitions
notes/                    paper and protocol notes
scripts/                  inference, scoring, analysis, and review utilities
src/r1eval/               data, perturbation, prompting, modeling, and scoring code
tests/                    unit and regression tests
results/analysis/         small verified aggregate tables only
RESULTS.md                complete human-readable aggregate results
```

Benchmark data, derived experiment images, model weights, raw generations,
scored per-item outputs, logs, archives, and report documents are excluded.

## Evaluation boundaries

- The target work uses model-assisted answer extraction/judging, whereas this
  repository uses a deterministic local scorer. Absolute accuracy is therefore
  not directly interchangeable with the paper’s reported values.
- Multiple-choice extraction is automated. Ambiguous and free-response outputs
  that cannot be scored conservatively remain marked for manual review rather
  than silently imputed.
- Bootstrap intervals resample paired question IDs. Exact McNemar tests compare
  directional correctness flips; the 12 robustness tests use Holm correction.
- The dual-prompt held-out split is held out only from prompt development, not
  from checkpoint training. Cross-checkpoint and perturbed-condition gate
  analyses remain exploratory.
- Fixed-canvas controls reduce composition confounds but cannot guarantee
  identical internal visual-token allocation.
- The budgeted multi-view comparison matches nominal pixels approximately, not
  the model’s hidden token count or compute exactly.

## Citation

```bibtex
@inproceedings{Yang_2025_ICCV,
  author    = {Yang, Yi and He, Xiaoxuan and Pan, Hongkun and Jiang, Xiyan
               and Deng, Yan and Yang, Xingtao and Lu, Haoyu and Yin, Dacheng
               and Rao, Fengyun and Zhu, Minfeng and Zhang, Bo and Chen, Wei},
  title     = {R1-Onevision: Advancing Generalized Multimodal Reasoning
               through Cross-Modal Formalization},
  booktitle = {Proceedings of the IEEE/CVF International Conference on
               Computer Vision (ICCV)},
  year      = {2025},
  pages     = {2376--2385}
}
```
