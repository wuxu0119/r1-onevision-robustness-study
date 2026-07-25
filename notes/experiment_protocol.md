# Recorded experiment protocol

This file documents the original study plan and the later extension. It was
not registered with an external time-stamped service, so the final report does
not describe the analyses as preregistered. Confirmatory and exploratory
claims are separated explicitly below.

## Hypotheses

H1. Accuracy decreases monotonically as visual resolution is reduced, with
errors concentrated in perception and formalization rather than arithmetic.

H2. Adding unrelated panels may harm performance. Because panel composition
also changes target scale, canvas geometry, and position, the original
distractor comparison alone cannot identify the causal mechanism.

H3. SFT and RL improve clean reasoning accuracy more than they improve
robustness to corrupted visual evidence.

H4. Explicit evidence extraction with multi-view inspection increases
robustness and makes failures more detectable, even when it does not improve
every raw answer.

## Data

Use all 942 R1-Onevision-Bench samples for clean reproduction.

Use a fixed, category-stratified 300-sample subset for robustness and
improvement experiments. The subset is selected once with seed 20260723 and
saved as a manifest.

## Models

- Base: Qwen/Qwen2.5-VL-7B-Instruct
- SFT: Fancy-MLLM/R1-Onevision-7B
- RL: Fancy-MLLM/R1-Onevision-7B-RL

Record the downloaded model revision hashes in every output file.

For SFT inference, load the model weights from
`Fancy-MLLM/R1-Onevision-7B` but the processor from its declared base model,
`Qwen/Qwen2.5-VL-7B-Instruct`. The SFT repository's processor configuration
uses the development-only class name `Qwen2_5_VLImageProcessor`, which stable
Transformers 4.49 does not recognize. Record the processor source in every
output row.

## Failure condition A: resolution degradation

Downsample each image to 50% or 25% of its original width and height, then
resize it back to the original size with bilinear interpolation. The semantic
question and ground-truth answer are unchanged.

## Failure condition B: distractor panels

Compose the target image with one or three unrelated benchmark images. Place
the target panel at a deterministic random position. The question still refers
only to the target content, so the correct answer is unchanged.

## Improvement

Evidence-first prompting asks the model to:

1. transcribe relevant visual values and labels;
2. state spatial or logical relations;
3. identify uncertain observations;
4. reason only from the listed evidence;
5. output a final answer in a machine-readable tag.

Global-plus-quadrants supplies the full image and four overlapping local views.
Agreement between the direct and evidence-first answers is evaluated as an
exploratory abstention heuristic, not as a validated failure detector.

## Primary metrics

- exact/deterministic answer accuracy;
- parse rate;
- accuracy drop relative to clean input;
- clean-to-perturbed answer consistency;
- accepted-set accuracy, rejected-set error rate, and coverage;
- selective accuracy among samples where methods agree;
- generation latency and output-token count.

## Manual error audit

If a systematic taxonomy is reported, manually classify at least 100
fixed-seed, condition-stratified failures:

- P: perception/transcription error;
- F: relation or formalization error;
- R: reasoning error after correct evidence;
- A: answer extraction or formatting error;
- U: uncertain or mixed.

The annotation is performed while viewing the input image, question,
ground-truth answer, generated evidence, and generated reasoning. Category
frequencies must not be reported until the table is filled and annotation
agreement has been measured.

## Statistical reporting

Report paired differences on the same samples. Final reported intervals use
10,000 paired bootstrap resamples. Apply a Holm correction across the 12
pre-defined model-by-perturbation tests. Avoid claiming improvement when the
interval includes zero or when a nominal p-value does not survive the stated
multiple-comparison procedure.

## Post-hoc extension

The extension was designed after auditing the original results and is therefore
exploratory.

- A fixed-canvas factorial design separates target-only resizing, blank/noise
  panels, target position (left/right), and three non-overlapping semantic
  distractor seeds. The design is run for Base, SFT, and RL.
- Eight stochastic runs of the RL checkpoint are generated for clean,
  distractor-3, and downsample-25 inputs. The first two runs form a
  cost-matched same-prompt agreement baseline; all eight runs define discrete
  majority-confidence operating points for risk-coverage analysis.
- Evidence-first prompting is run for all five visual conditions and all three
  checkpoints.
- A clean evidence-first run covers all 942 questions. The prompt is evaluated
  separately on the 642-row complement that was not used during development,
  providing held-out clean validation.
- The OOM-prone five-view experiment is repeated with one 864x864 global view
  versus five 384x384 views. Their nominal pixel budgets are 746,496 and
  737,280 respectively. This is a practical visual-token proxy rather than a
  guarantee of identical internal token counts.
- Except for the 642-row clean complement, extension analyses use the original
  300-item subset and remain exploratory.
