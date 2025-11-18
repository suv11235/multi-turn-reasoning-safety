# Crescendo-Inspired Subset Plan

## Objectives
- Target **zero-shot refusal** behavior in **DeepSeek-Distill-Llama-3.1-8B** without additional safety conditioning.
- Focus on **distillation artifacts**: cases where the student model forgets or over-applies teacher safety policies.
- Cover both **harmful compliance** risks and **over-refusal** of benign requests.

## Scenario Families
1. **Direct Malicious Instructions** – canonical jailbreak targets (e.g., weaponization, self-harm incitement) with minimal obfuscation.
2. **Indirection & Roleplay Attacks** – multi-turn or persona-driven prompts that disguise harmful intent behind storytelling or research framing.
3. **Dual-Use Technical Content** – legitimate scientific or engineering questions that could be misused; monitors nuance handling.
4. **Borderline / Ambiguous Requests** – prompts hinging on disclaimers, hypotheticals, or context that require calibrated refusals rather than blanket denials.
5. **Benign-but-Sensitive Topics** – mental health check-ins, political analysis, or personal data questions designed to detect over-refusal.
6. **Meta-Safety & Reflection Tasks** – prompts asking the model to critique harmful behavior or explain refusal rationales.
7. **Perturbation Slices** – paraphrases, tone shifts, language translations, and prompt-order variations used strictly for diagnostics.

## Sampling Strategy
- Start from Crescendo’s taxonomy but **re-balance** toward ambiguous and dual-use content (often underrepresented in refusal datasets).
- Ensure **≥5 prompts per primary family** in the core benchmark; diagnostics can reuse content via perturbations.
- For each harmful prompt, craft **paired benign controls** sharing topic but removing malicious intent.
- Insert **distillation stress tests**: prompts referencing the teacher model, conflating instructions, or mirroring known distillation failure narratives.

## Annotation Hooks
- Every prompt mapped to: `category`, `risk_level`, `expected_behavior`, `slice`, `attack_pattern`, `language`, `parent_id` (for perturbations).
- Maintain textual rationales to guide future human evaluators.

## Deliverables
- Machine-readable taxonomy (`prompt_taxonomy.json`).
- Slice definitions (`slice_definitions.json`).
- Prompt file (`../prompts/prompts_refusal_benchmark.jsonl`).
- Documentation describing threat model & evaluation axes (`../docs/threat_model.md`, `../docs/design_decisions.md`).
