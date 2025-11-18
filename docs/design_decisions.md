# Design Decisions

## Benchmark Philosophy
- **Teacher-student mismatch focus**: Many Crescendo-style datasets assume full-strength base models. We explicitly search for distillation regressions (lost safety coverage, inconsistent refusals).
- **Core vs diagnostic separation**: Core prompts drive headline metrics; diagnostics enable causal analysis (paraphrases, personas, translations, context order, etc.).
- **Balanced harm/benign coverage**: For every harmful slice we pair benign or borderline items on the same topic to detect over-refusal.

## Prompt Construction Rules
1. **Single-turn encoding** even for multi-step narratives; diagnostics replicate context manipulations.
2. **Language diversity** kept small but purposeful—if the model fails in EN, test again in ES to detect localized safety gaps.
3. **Metadata-first**: prompts are written *after* schema fields are defined to avoid metadata drift.

## Evaluation Architecture
- **Deterministic inference** with low temperature ensures reproducibility and clearer comparison to future safety interventions.
- **Structured outputs** (JSONL) for raw generations allow downstream labeling, metrics, and visualization without re-running inference.
- **Layered analysis**: automatic heuristics → human adjudication subset → qualitative case studies.

## Tooling Choices
- **Transformers + Safetensors** for local inference; fallback huggingface hub loads, optionally replaced by API wrappers.
- **Pydantic schemas** protect prompt/result structure.
- **Polars/Pandas** for analytics; Altair for visual summaries (reproducible Vega-Lite specs).

## Extension Strategy
- New prompt slices should extend taxonomy JSON and include rationales in `notes`.
- Additional models can reuse scripts by swapping `inference_config.yaml` entries.
- Visualizations accept aggregated metrics CSVs, so they remain model-agnostic.
