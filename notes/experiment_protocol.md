# Preregistered experiment protocol

## Hypotheses

H1. Accuracy decreases monotonically as visual resolution is reduced, with
errors concentrated in perception and formalization rather than arithmetic.

H2. Adding unrelated panels harms performance because the model selects and
formalizes irrelevant evidence.

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
Agreement between the direct and evidence-first answers acts as a failure
detector.

## Primary metrics

- exact/deterministic answer accuracy;
- parse rate;
- accuracy drop relative to clean input;
- clean-to-perturbed answer consistency;
- failure-detection precision, recall, and F1;
- selective accuracy among samples where methods agree;
- generation latency and output-token count.

## Manual error audit

After inference, manually classify at least 50 baseline failures:

- P: perception/transcription error;
- F: relation or formalization error;
- R: reasoning error after correct evidence;
- A: answer extraction or formatting error;
- U: uncertain or mixed.

The annotation is performed while viewing the input image, question,
ground-truth answer, generated evidence, and generated reasoning.

## Statistical reporting

Report paired differences on the same samples. Use 2,000 paired bootstrap
resamples for 95% confidence intervals. Avoid claiming improvement when the
interval includes zero.
