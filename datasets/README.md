# Datasets & Configuration Assets

This directory captures the Crescendo-inspired dataset decisions that power the zero-shot refusal benchmark.

Files:
- `crescendo_subset_plan.md` – rationale for the subset of Crescendo-style scenarios we adopt for distill-model analysis.
- `prompt_taxonomy.json` – machine-readable taxonomy spanning harmful, borderline, benign, and perturbation categories.
- `scoring_scheme.md` – grading rubric for refusals, safety correctness, helpful helpfulness, and calibration.
- `slice_definitions.json` – definition of core vs diagnostic slices plus perturbation tags.
- `inference_config.yaml` – canonical zero-shot inference parameters for DeepSeek-Distill-Llama-3.1-8B.
