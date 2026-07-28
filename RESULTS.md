# Verified Results after Uniform Scorer Repair

This document reports the complete aggregate evidence used by the study. All
values below were recomputed from the repaired deterministic scorer. Original
raw JSONL files and original scored CSVs were preserved; no answer was manually
imputed.

Unless stated otherwise, confidence intervals use 10,000 bootstrap resamples
of question IDs. Paired comparisons use the same questions on both sides.
McNemar *p*-values are exact and two-sided. The 12 primary robustness
comparisons use Holm family-wise correction.

## 1. Data integrity and scorer audit

| Archive | Rows | Runtime errors | Old correct | Repaired correct | Old manual review | Repaired manual review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Core | 10,926 | 680 | 3,584 | 3,631 | 1,312 | 1,219 |
| Extension | 26,142 | 0 | 8,947 | 8,976 | 1,052 | 1,034 |

The 680 core errors are CUDA out-of-memory records from the initial
unbudgeted multi-view branch. They are kept as resource failures and excluded
from model-accuracy conclusions. All extension groups have their expected row
counts, no runtime errors, and no duplicate experiment keys.

The original answer extractor could miss explicit choices separated by a line
break and could accept a stray trailing letter. The repaired precedence is:

1. final `<answer>...</answer>` block;
2. latest explicit final/correct answer phrase or direct `\boxed{A}`–`\boxed{E}`;
3. conservative bare or leading choice;
4. manual review otherwise.

Across both archives, 178 rows changed from incorrect to correct because an
explicit answer was recovered; 102 changed from correct to incorrect because
an unsafe tail match was removed or a later explicit answer took precedence.
All 13 dedicated adversarial scorer tests pass.

The remaining core manual-review set contains 680 runtime errors, 363
non-error multiple-choice outputs, and 176 non-error free-response outputs.
The extension retains 1,034 non-error manual-review rows (671 multiple-choice
and 363 free-response). These cases are not silently converted.

## 2. Clean reproduction on all 942 questions

| Checkpoint | Correct | Accuracy | 95% CI | Manual review | Mean output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base: Qwen2.5-VL-7B-Instruct | 348 | 36.94% | [33.86, 40.02] | 63 | 476.4 |
| R1-Onevision-7B (SFT) | 320 | 33.97% | [31.00, 37.05] | 90 | 961.0 |
| R1-Onevision-7B-RL | 351 | 37.26% | [34.18, 40.34] | 29 | 529.0 |

### Paired checkpoint comparisons

| Comparison | Delta | 95% CI | Left-only correct | Right-only correct | Exact McNemar *p* |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT − Base | −2.97 pt | [−6.58, +0.53] | 159 | 131 | 0.113 |
| RL − SFT | +3.29 pt | [−0.11, +6.79] | 123 | 154 | 0.071 |
| RL − Base | +0.32 pt | [−3.18, +3.82] | 141 | 144 | 0.906 |

The repaired results do not support a significant RL-over-Base advantage under
this local scoring protocol. The evaluator audit materially changes the
checkpoint comparison, showing that checkpoint ranking cannot be separated
from answer extraction without validating the scorer.

## 3. Complete 3×5 robustness matrix

The robustness subset contains the same 300 seed-fixed questions in every
cell. “Downsample-50” and “Downsample-25” reduce the target before bilinear
resizing to its original dimensions. “Distractor-1” and “Distractor-3” append
unrelated benchmark panels.

| Model | Clean | Downsample-50 | Downsample-25 | Distractor-1 | Distractor-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 36.67% | 38.33% | 34.00% | 35.00% | 32.67% |
| SFT | 38.00% | 34.33% | 36.33% | 29.67% | 38.33% |
| RL | 33.00% | 36.67% | 40.67% | 35.00% | 36.67% |

### All 12 clean-to-perturbed paired comparisons

“Agreement” requires the same non-empty normalized answer before and after the
change. The delta is perturbed minus clean.

| Model | Perturbation | Delta | 95% CI | Correct→wrong | Wrong→correct | Answer agreement | Holm *p* |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | Downsample-50 | +1.67 pt | [−3.33, +6.67] | 26 | 31 | 56.33% | 1.000 |
| Base | Downsample-25 | −2.67 pt | [−8.00, +2.33] | 35 | 27 | 51.33% | 1.000 |
| Base | Distractor-1 | −1.67 pt | [−7.33, +4.00] | 41 | 36 | 44.67% | 1.000 |
| Base | Distractor-3 | −4.00 pt | [−9.33, +1.33] | 41 | 29 | 40.67% | 1.000 |
| SFT | Downsample-50 | −3.67 pt | [−9.00, +2.00] | 41 | 30 | 47.67% | 1.000 |
| SFT | Downsample-25 | −1.67 pt | [−7.33, +4.00] | 41 | 36 | 44.00% | 1.000 |
| SFT | Distractor-1 | −8.33 pt | [−14.00, −3.00] | 49 | 24 | 40.00% | 0.056 |
| SFT | Distractor-3 | +0.33 pt | [−5.67, +6.33] | 42 | 43 | 37.67% | 1.000 |
| RL | Downsample-50 | +3.67 pt | [−1.67, +9.00] | 29 | 40 | 47.33% | 1.000 |
| RL | Downsample-25 | +7.67 pt | [+2.33, +13.00] | 24 | 47 | 46.00% | 0.094 |
| RL | Distractor-1 | +2.00 pt | [−3.67, +7.67] | 33 | 39 | 42.33% | 1.000 |
| RL | Distractor-3 | +3.67 pt | [−2.33, +9.67] | 37 | 48 | 40.67% | 1.000 |

None of the 12 paired effects survives Holm correction. More importantly,
answer agreement is only 37.67%–56.33% even when mean accuracy barely moves.
For example, SFT under three distractors changes by only +0.33 points, yet 85
of 300 correctness states flip and only 37.67% of final answers remain the
same. Mean accuracy therefore conceals substantial item-level instability.

The unadjusted SFT Distractor-1 decrease and RL Downsample-25 increase are not
interpreted as isolated causal effects. Both require composition controls and
multiplicity correction; the latter also contains many wrong-to-correct
chance flips and does not demonstrate improved visual reasoning.

## 4. Fixed-canvas composition controls

The original distractor operation changes more than semantic content: it can
alter target scale, layout, nominal image size, and visual-token allocation.
The extension therefore compares target-only resizing with matched blank,
noise, and semantic panels at left/right positions. Semantic results average
three independent distractor seeds at both positions (six trials per item).

### Descriptive accuracy

| Model | Clean | Target-only rescaled | Matched blank mean | Matched noise mean | Semantic-panel mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 36.67% | 35.67% | 34.83% | 34.67% | 36.61% |
| SFT | 38.00% | 33.67% | 34.17% | 32.50% | 31.17% |
| RL | 33.00% | 35.67% | 36.33% | 34.17% | 34.94% |

### Paired semantic-panel effects

| Model | Comparison | Mean delta | 95% CI |
| --- | --- | ---: | ---: |
| Base | Semantic mean − clean | −0.06 pt | [−4.50, +4.39] |
| Base | Semantic mean − matched blank | +1.78 pt | [−1.61, +5.17] |
| SFT | Semantic mean − clean | −6.83 pt | [−11.56, −2.28] |
| SFT | Semantic mean − matched blank | −3.00 pt | [−6.44, +0.50] |
| RL | Semantic mean − clean | +1.94 pt | [−2.67, +6.50] |
| RL | Semantic mean − matched blank | −1.39 pt | [−4.89, +2.06] |

SFT remains worse with semantic panels than with the original clean input, but
the semantic-minus-matched-blank interval crosses zero. The supported
interpretation is therefore narrower than “semantic distractors cause the
entire drop”: resizing/layout accounts for part of the penalty, and the
residual semantic contribution is unresolved at this sample size. Base and RL
show no interval excluding zero in either comparison.

The mean semantic-versus-matched-blank answer agreement is also low: 46.78%
for Base, 35.56% for SFT, and 44.06% for RL. Composition controls reduce a
causal confound; they do not remove the underlying prediction instability.

## 5. Dual-prompt agreement as selective prediction

The gate compares the paper-format prompt with a fixed evidence-first prompt.
It answers only when both normalized final answers are non-empty and identical;
otherwise it abstains.

### Held-out clean complement

The evidence-first prompt was developed using the original 300-question
subset. The remaining 642 clean questions are evaluated separately.

| Split | N | Accepted | Coverage | Baseline accuracy | Selective accuracy | Random rejection at same coverage | Rejected-set error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Held-out complement | 642 | 288 | 44.86% [40.97, 48.75] | 39.25% | 48.26% [42.46, 54.04] | 39.25% [35.07, 43.40] | 68.08% [63.06, 72.84] |

Agreement selects a more accurate subset than random rejection at the same
coverage, but accepts fewer than half of the items. This is best described as
a promising training-free abstention heuristic, not as a fully validated
failure detector.

The paired item-bootstrap uplift over uniform same-count random acceptance is
+9.01 percentage points (95% CI [+4.98, +13.32]). It remains +7.63 points
([+3.82, +11.67]) after matching accepted counts by answer type and +7.23
points ([+3.40, +11.13]) after matching answer type and benchmark domain.
The rejected subset has 68.08% error, 7.33 points ([+4.03, +10.92]) above the
full held-out error rate. These are exploratory selection references, not
independent competing detectors.

### Cross-checkpoint and perturbation check on the 300-item design subset

These 15 groups are exploratory because they reuse the prompt-development
subset. They are included in full to avoid selective reporting.

| Model | Condition | Baseline | Coverage | Selective accuracy |
| --- | --- | ---: | ---: | ---: |
| Base | Clean | 36.67% | 41.67% | 52.00% |
| Base | Downsample-50 | 38.33% | 40.67% | 54.92% |
| Base | Downsample-25 | 34.00% | 41.33% | 50.00% |
| Base | Distractor-1 | 35.00% | 42.00% | 50.00% |
| Base | Distractor-3 | 32.67% | 41.67% | 45.60% |
| SFT | Clean | 38.00% | 40.00% | 58.33% |
| SFT | Downsample-50 | 34.33% | 40.00% | 54.17% |
| SFT | Downsample-25 | 36.33% | 41.33% | 49.19% |
| SFT | Distractor-1 | 29.67% | 40.67% | 45.90% |
| SFT | Distractor-3 | 38.33% | 37.00% | 56.76% |
| RL | Clean | 33.00% | 44.00% | 46.97% |
| RL | Downsample-50 | 36.67% | 40.67% | 55.74% |
| RL | Downsample-25 | 40.67% | 47.33% | 52.11% |
| RL | Distractor-1 | 35.00% | 42.33% | 46.46% |
| RL | Distractor-3 | 36.67% | 41.00% | 46.34% |

The pattern is checkpoint- and condition-dependent. This table does not by
itself establish checkpoint-general detection because the prompt and operating
rule were not selected on an independent development set for Base and SFT.

## 6. Same-prompt stochastic consistency baselines

For RL, eight stochastic decodes were generated for each of 300 items in three
conditions. A two-run agreement gate is the cost-matched comparator to the
two-prompt gate. The eight-run method accepts only a unique majority whose vote
share is at least 0.75.

| Condition | Method | Accepted | Coverage | Reference accuracy | Selective accuracy (95% CI) |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean | Single stochastic run | 300 | 100.00% | 34.33% | 34.33% [29.00, 39.67] |
| Clean | Two-run same-prompt agreement | 127 | 42.33% | 34.33% | 47.24% [38.64, 56.00] |
| Clean | Eight-run majority, vote ≥0.75 | 89 | 29.67% | 36.67% | 62.92% [52.75, 72.83] |
| Distractor-3 | Single stochastic run | 300 | 100.00% | 32.33% | 32.33% [27.00, 37.67] |
| Distractor-3 | Two-run same-prompt agreement | 108 | 36.00% | 32.33% | 50.00% [40.35, 59.57] |
| Distractor-3 | Eight-run majority, vote ≥0.75 | 78 | 26.00% | 38.33% | 60.26% [49.33, 71.25] |
| Downsample-25 | Single stochastic run | 300 | 100.00% | 32.67% | 32.67% [27.33, 38.00] |
| Downsample-25 | Two-run same-prompt agreement | 118 | 39.33% | 32.67% | 53.39% [44.44, 62.40] |
| Downsample-25 | Eight-run majority, vote ≥0.75 | 86 | 28.67% | 36.33% | 65.12% [54.93, 75.00] |

“Reference accuracy” is first-run accuracy for the two-run gate and full-set
majority-vote accuracy for the eight-run gate. The stochastic baselines show
that agreement is not unique to the evidence-first prompt. Eight-run
confidence attains higher selective accuracy but at 8× generation cost and
only 26.0%–29.7% coverage. These results use the 300-item design subset and
remain exploratory.

## 7. Budgeted multi-view comparison

The original global-plus-four-quadrants experiment was not a fair accuracy
comparison because 680 of 1,800 multi-view attempts exceeded 40 GB, including
all 600 three-distractor multi-view attempts. The replacement approximately
matches nominal pixels: one 864×864 global view (746,496 pixels) versus five
384×384 views (737,280 pixels).

All 3,600 budgeted generations finish without an error.

| Condition | Prompt | Global | Five-view | Delta | 95% CI | Exact McNemar *p* |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean | Paper | 35.33% | 34.67% | −0.67 pt | [−6.67, +5.33] | 0.916 |
| Clean | Evidence-first | 36.00% | 33.33% | −2.67 pt | [−8.67, +3.33] | 0.445 |
| Downsample-25 | Paper | 35.33% | 35.00% | −0.33 pt | [−6.00, +5.33] | 1.000 |
| Downsample-25 | Evidence-first | 32.00% | 33.00% | +1.00 pt | [−5.00, +6.67] | 0.822 |
| Distractor-3 | Paper | 33.00% | 32.33% | −0.67 pt | [−6.67, +5.33] | 0.915 |
| Distractor-3 | Evidence-first | 31.67% | 31.67% | +0.00 pt | [−6.00, +6.00] | 1.000 |

Across the six equally sized groups, global accuracy is 33.89% and five-view
accuracy is 33.33%. Mean latency rises from 13.89 to 14.34 seconds, and the
maximum recorded peak memory rises from 17.70 to 17.79 GB. No paired confidence
interval excludes zero. Budgeting solves the engineering failure but does not
produce evidence of an accuracy benefit.

## 8. Fail-closed reliability-signal requalification

A later 10,080-call engineering pilot completed with all expected unique
records and zero infrastructure failures, but failed the frozen signal-validity
gates: 123/126 parseability cells and 5/6 verbal-confidence cells were below
threshold. Multiple-choice parseability was 89.83%; free-response
parseability was 59.62%. The authenticated decision was **NO-GO**, and no
efficacy analysis was performed.

A stricter Stage-B0 v7 excluded-canary audit then compared the compact option
scorer with the full-sequence oracle. It returned **NO-GO** with 46 role-level
equivalence failures (23 Base, 15 SFT, 8 RL), five aggregate-equivalence
failures, and two aggregate winner mismatches. No launch lock was issued and
no formal Stage-B0 efficacy run started.

These are validity results rather than model-performance results. Full
accounting and the future oracle-only workload are documented in
[`RELIABILITY_AUDIT.md`](RELIABILITY_AUDIT.md).

## 9. Interpretation boundaries

- This is a reproducibility and measurement study, not a new training method.
- Absolute scores depend on the local deterministic evaluator and are not
  directly interchangeable with model-judged values in the target paper.
- No completed human error taxonomy is claimed. Qualitative examples are case
  studies, not frequency estimates.
- The 642-question gate evaluation is held out from prompt development, but
  not from checkpoint pretraining or post-training.
- The 300-item cross-checkpoint gate, stochastic consistency, and
  perturbation analyses are exploratory.
- Fixed-canvas blank/noise/semantic controls reduce, but cannot eliminate,
  differences in hidden visual tokenization.
- Same-coverage random rejection is a necessary baseline, not proof of optimal
  calibration.
- The 0.75 eight-run threshold is a reported operating point; the full
  risk–coverage table should be consulted rather than selecting one point
  after seeing test performance.
- Nominal pixel matching in the multi-view study is an engineering proxy, not
  exact matching of hidden token count, FLOPs, or wall-clock cost.

## 10. Machine-readable summaries

The repository tracks only compact aggregate files under
`results/analysis/`, including:

- `reproduction_models.csv`
- `reproduction_paired_deltas.csv`
- `reproduction_by_domain.csv`
- `robustness_paired.csv`
- `full_group_matrix.csv`
- `scorer_repair_summary.json`
- `ifsr_stage_a_audit_summary.json`
- `ifsr_stage_b0_v7_audit_summary.json`
- `extension/control_condition_summary.csv`
- `extension/control_paired_contrasts.csv`
- `extension/semantic_seed_aggregate.csv`
- `extension/clean_heldout_gate.csv`
- `extension/dual_prompt_gate_all_checkpoints.csv`
- `extension/dual_prompt_gate_by_item_type.csv`
- `extension/heldout_gate_by_group.csv`
- `extension/self_consistency_risk_coverage.csv`
- `extension/inference_cost_summary.csv`
- `extension/multiview_budgeted_summary.csv`
- `extension/multiview_budgeted_paired.csv`
- `extension/metadata.json`

Raw generations, model weights, benchmark images, per-item scored outputs,
logs, and manuscript files are not committed.
