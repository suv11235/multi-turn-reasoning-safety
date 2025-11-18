# Related Work: Refusal & Jailbreak Benchmarks

| Work | Focus | Relevance |
|------|-------|-----------|
| Microsoft Crescendo (2024) | Staged safety prompts with human agreement labels. | Provides scaffolding for scenario families and escalation structure; our subset re-balances slices for distill models. |
| HarmBench | Large-scale disallowed-content prompts emphasizing jailbreak performance. | Highlights weaponization prompts; we extend with benign calibration and distillation diagnostics. |
| WildJailbreak | Real-world jailbreak prompts collected from the web. | Inspires perturbation and persona slices; we add structured metadata absent in WildJailbreak. |
| AdvBench / AutoDAN | Automatically generated adversarial prompts. | Motivates inclusion of indirect + multi-hop prompts; our taxonomy ensures traceability per prompt. |
| Helpful-Harmless Benchmark (Anthropic) | Paired helpful vs harmful instructions with RLHF focus. | Informs our benign-sensitive controls and calibration scoring dimension. |

## Novel Contributions Relative to Prior Work
1. **Distillation-centric objective**: Prior datasets assume base models; we target teacher-student mismatch failure modes.
2. **Explicit benign calibration metrics**: Over-refusal tracking is elevated to a first-class metric.
3. **Perturbation linkage**: Structured `parent_id` relationships allow systematic robustness studies.
4. **Meta-safety coverage**: Incorporates prompts requiring refusal explanations, absent in most jailbreak corpora.
