# Reasoning Safety Benchmarks (Child Abuse, Firearm, Phishing)

This project instantiates the action items from the meeting with Pradeep:

1. **Align datasets with the reference paper's categories** for child abuse, firearm, and phishing scenarios. Fill any gaps by designing new prompts.
2. **Prepare reasoning-model (non-CB) baselines** using standardized scripts so results can be shared as soon as runs finish.
3. **Capture benchmark outputs** and share them with the safety review team once metrics meet the agreed reporting format.

## Directory Structure

```
reasoning_benchmarks/
├── datasets/        # Category-specific dataset specs, gap analyses, data prep procedures
├── prompts/         # Prompt schema + benchmark prompt files per category
├── scripts/         # Runner + evaluation scaffolding for reasoning models
├── docs/            # Meeting-aligned TODO list, baseline alignment notes, sharing checklist
└── README.md        # This file
```

## Benchmark Execution

1. **Run category-specific benchmarks**:
   ```bash
   python reasoning_benchmarks/scripts/run_reasoning_benchmark.py \
       --config reasoning_benchmarks/datasets/inference_config.yaml \
       --prompts reasoning_benchmarks/prompts/[CATEGORY]_prompts.jsonl \
       --out reasoning_benchmarks/results/[CATEGORY]_generations.jsonl
   ```

2. **Validate against baselines**:
   ```bash
   python scripts/evaluate_metrics.py \
       --generations reasoning_benchmarks/results/[CATEGORY]_generations.jsonl \
       --out reasoning_benchmarks/results/[CATEGORY]_metrics.json
   ```

## Category Coverage

- **Child Abuse**: Direct + indirect grooming scenarios
- **Firearms**: Weapon modification + acquisition
- **Phishing**: Email + credential harvesting
- **Bio-weapons**: Synthesis + dual-use
- **Multi-turn**: Contextual escalation tests

## Deliverables Checklist

- [ ] Child abuse benchmark results
- [ ] Firearm benchmark results
- [ ] Phishing benchmark results
- [ ] Bio-weapon coverage analysis
- [ ] Multi-turn scenario report
