# Manual error annotation guide

Annotate at least 50 failures after opening the stored image and reading the
generated output.

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

