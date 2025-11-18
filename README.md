# Zero-Shot Refusal Baseline for DeepSeek-Distill-Llama-3.1-8B

This repository scaffolds a novel benchmark inspired by the Microsoft Crescendo paper to study zero-shot refusal behavior in **DeepSeek-Distill-Llama-3.1-8B**. It includes:

- 🧱 **Dataset assets** describing the tailored Crescendo-style subset, taxonomy, scoring scheme, and slice definitions.
- 🗒️ **Prompt resources** with a machine-readable schema and seed prompts covering harmful, benign, borderline, and perturbation cases.
- 🧮 **Scripts** for running the baseline model, labeling refusals, computing metrics, perturbation diagnostics, and producing visualizations.
- 📊 **Results & visualization stubs** for storing generations, metrics, and downstream figures.
- 📚 **Documentation** capturing design decisions, related work, human-eval protocol, error analyses, future interventions, and a technical write-up.

> ⚠️ The repo intentionally separates prompts, datasets, and evaluation artifacts so each can evolve independently while staying reproducible.

## Layout

```
datasets/   # Derived Crescendo subset plan, taxonomy, scoring & slice specs
prompts/    # Prompt schema + zero-shot refusal benchmark prompts (JSONL)
docs/       # Design docs, protocols, literature surveys, analyses
scripts/    # Runner, labeling, metrics, perturbation, visualization utilities
results/    # Placeholder for generations, metrics, and figures
```

## Getting started

1. Inspect design docs in `docs/` to understand goals, threat model, and evaluation plan.
2. Review `prompts/prompts_refusal_benchmark.jsonl` for the benchmark prompts and expand as needed.
3. Install dependencies (see each script header) and run `scripts/run_baseline.py` to collect zero-shot generations.
4. Process outputs with `scripts/auto_label_refusals.py`, then score via `scripts/evaluate_metrics.py` and `scripts/visualize_results.py`.
5. Follow `docs/reproducibility.md` for packaging and sharing the benchmark + baseline findings.
