# Verified Results

All numbers below were recomputed locally from 10,926 scored records. There are
30 expected experimental groups and no duplicate experiment keys.

## Full benchmark reproduction (n=942)

| Model | Accuracy | 95% bootstrap CI | Mean output tokens |
| --- | ---: | ---: | ---: |
| Base | 33.44% | [30.36, 36.41] | 476.4 |
| SFT | 34.71% | [31.63, 37.79] | 961.0 |
| RL | 37.37% | [34.29, 40.45] | 529.0 |

Paired comparisons:

| Comparison | Delta | 95% CI | Exact McNemar p |
| --- | ---: | ---: | ---: |
| SFT - Base | +1.27 pt | [-2.23, +4.88] | 0.525 |
| RL - SFT | +2.65 pt | [-0.74, +6.16] | 0.149 |
| RL - Base | +3.93 pt | [+0.42, +7.54] | 0.036 |

## Selected robustness findings (n=300 paired)

| Model / condition | Clean | Perturbed | Delta | Answer agreement |
| --- | ---: | ---: | ---: | ---: |
| SFT / distractor_1 | 39.0% | 29.7% | -9.33 pt | 40.7% |
| SFT / downsample_50 | 39.0% | 34.7% | -4.33 pt | 49.0% |
| Base / distractor_3 | 33.7% | 31.3% | -2.33 pt | 36.3% |
| RL / downsample_25 | 33.0% | 40.3% | +7.33 pt | 46.0% |

The RL downsample gain does not imply better visual reasoning: 25
clean-correct answers become wrong, while 47 clean-wrong answers become
correct. The low agreement indicates a shifted and unstable decision path.

## Dual-prompt agreement gate

The system answers only if the paper-format prompt and evidence-first prompt
produce the same normalized final answer.

| Condition | Baseline accuracy | Coverage | Selective accuracy | Failure F1 |
| --- | ---: | ---: | ---: | ---: |
| Clean | 33.0% | 44.3% | 46.6% | 0.707 |
| Distractor-3 | 36.7% | 40.7% | 46.7% | 0.679 |
| Downsample-25 | 40.3% | 47.7% | 51.0% | 0.649 |

## Resource failure

The global-plus-four-quadrants branch produced 680 CUDA OOM records out of
1,800 multi-view attempts (37.8%) on A100 40GB. All 600
distractor_3/multi-view attempts failed. These records are reported as a
resource limitation and excluded from model-accuracy claims.
