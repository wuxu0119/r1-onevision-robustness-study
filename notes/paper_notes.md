# R1-Onevision paper notes

## Bibliographic status

- Yi Yang et al.
- ICCV 2025 main proceedings, pp. 2376-2385.
- The method, model, data, and benchmark are publicly available.

## Main claim

R1-Onevision argues that multimodal reasoning requires models to extract,
structure, and verify visual information rather than treating an image as a
generic caption. The paper uses cross-modal formalization to construct
reasoning supervision and then post-trains Qwen2.5-VL with SFT and rule-based
RL.

## Cross-modal reasoning data pipeline

1. Curate data from natural scenes, science, mathematics, OCR, charts, and
   documents.
2. Convert images to formal textual representations:
   - circuit schematics to SPICE;
   - flowcharts to PlantUML or Mermaid;
   - tables to CSV or JSON;
   - natural scenes to captions plus Grounding-DINO boxes;
   - text-heavy images through EasyOCR plus layout restoration.
3. Use DeepSeek-R1 and role-playing prompts to generate detailed reasoning.
4. Use GPT-4o to remove inaccurate or inconsistent reasoning steps.
5. Fine-tune Qwen2.5-VL on approximately 155k samples.
6. Apply rule-based reinforcement learning, including experiments on CLEVR.

## Important interpretation for this project

The formal representation is mainly part of training-data construction. The
released inference example sends an image and question directly to the model.
Therefore, the visual formalization step is latent at inference time and cannot
be directly inspected.

Our redesign makes this step explicit:

1. extract structured evidence;
2. expose uncertain facts;
3. inspect both the global image and local quadrants;
4. use disagreement between direct and evidence-first answers as a failure
   signal.

## Official benchmark

R1-Onevision-Bench contains 942 samples and five categories:

- Mathematics
- Physics
- Chemistry
- Biology
- Deduction

It includes grade/difficulty labels and both multiple-choice and open-answer
problems. The paper reports results by average accuracy, grade, and category.

## Published results to reproduce

On R1-Onevision-Bench, the paper reports:

- Qwen2.5-VL-7B: 32.1 average accuracy
- R1-Onevision-7B: 36.2 average accuracy
- Qwen2.5-VL-72B: 52.0 average accuracy

On mathematical benchmarks, the paper reports that SFT+RL improves the
Qwen2.5-VL-7B baseline on MathVision from 25.4 to 29.9 and on MathVerse from
43.6 to 46.4.

These published values are reference values only. They must not be presented as
our reproduced results.

## Paper limitations relevant to our study

- The official evaluation uses GPT-4o-mini to extract and judge answers,
  reducing full reproducibility.
- The paper does not isolate perception/formalization errors from reasoning
  errors at inference time.
- Robustness to resolution loss and irrelevant visual panels is not
  systematically evaluated.
- The SFT model card and separate RL checkpoint can be confusing; experiments
  must record exact Hugging Face model IDs and revisions.
- Long generated reasoning can be fluent even when the initial visual fact is
  incorrect.

## Exact inference prompt reported in the appendix

The RL model is evaluated with:

> First output the thinking process in <think> </think> tags and then output
> the final answer in <answer> </answer> tags.

The official generation configuration for the RL checkpoint uses temperature
0.1, top-k 1, top-p 0.001, and repetition penalty 1.05.

