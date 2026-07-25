# Manual error annotation guide

This document is a codebook, not evidence that annotation has already been
completed. Any report of category frequencies must be based on a filled
annotation table and must state the sampling and reliability protocol.

For a systematic study, draw a fixed-seed stratified sample of at least 100
incorrect RL predictions: 20 each from clean, downsample-50, downsample-25,
distractor-1, and distractor-3. Open the stored image, read the question and
generated output, and inspect the reference answer before assigning a label.

## Labels

- `P` - Perception: a visible number, symbol, label, or object is transcribed
  incorrectly or omitted.
- `F` - Formalization: the relevant elements are perceived, but their spatial,
  mathematical, or logical relation is represented incorrectly.
- `R` - Reasoning: the necessary visual facts are correct, but a later
  inference, equation, or deduction is incorrect.
- `A` - Answer formatting/extraction: the reasoning reaches the correct result
  but the final machine-readable answer is absent or wrong.
- `U` - Uncertain/mixed: more than one stage fails or the cause cannot be
  assigned reliably.

## Annotation rule

Choose the earliest clearly observable failure in the pipeline. For example, if
the model reads 8 as 3 and then correctly computes from 3, label the sample `P`,
not `R`.

Do not infer an internal attention mechanism from fluent reasoning text. Record
only observable errors in the image description, formal representation,
inference, or answer string. Use `U` whenever the evidence does not support a
single category.

## Reliability

Preferred protocol:

1. Two annotators label the same shuffled sample independently.
2. Resolve disagreements only after saving the independent labels.
3. Report raw agreement, Cohen's kappa, the number of disagreements, and the
   adjudicated category counts.

If a second annotator is unavailable, the same annotator may relabel the
shuffled sample after a delay. Report this explicitly as intra-rater rather
than inter-rater agreement.
