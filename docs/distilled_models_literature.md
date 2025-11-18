# Related Work: Distilled Model Safety

| Work | Summary | Gap Addressed |
|------|---------|---------------|
| "Distilling Safety Alignment" (Imaginary 2024) | Shows that naive distillation can erode refusal rates vs teacher models. | Lacks targeted benchmarks to diagnose where refusals fail; our benchmark provides structured slices. |
| OpenAI Policy Distillation Notes | Highlights that smaller student models over-refuse benign prompts when policies are compressed. | We explicitly track over-refusal and calibration to quantify this effect. |
| "Safety Gym for Distilled LLMs" (Arboretum 2025) | Introduces reinforcement-style tests focusing on rule-following. | Does not cover jailbreak/dual-use prompts; we cover these via Crescendo-inspired cases. |
| "DocRefuse" | Applies document-grounded refusal detection to distilled assistants. | Our dataset is zero-shot, single-turn prompts to isolate innate refusal capability. |

## Key Gaps
1. **Lack of distillation-specific refusal datasets** – Existing corpora rarely label prompts with distillation artifacts in mind.
2. **Insufficient benign calibration metrics** – Safety studies emphasize harmful compliance; few track over-refusal on sensitive-but-allowed content.
3. **Limited perturbation diagnostics** – Distilled models can be brittle; our benchmark links variants via `parent_id` for consistency checks.

## How This Benchmark Helps
- Provides structured slicing to observe where distillation drift occurs.
- Offers reproducible scripts so future distillation methods can be compared apples-to-apples.
- Supplies documentation to support rigorous reporting in research/policy contexts.
