# Novel Evaluation Axes

1. **Distillation Drift Sensitivity**
   - Compare refusal behavior across prompts known to be safe in the teacher model.
   - Flag cases where the distilled student under-refuses relative to teacher policies.

2. **Refusal Consistency Under Perturbation**
   - Measure variance in refusal scores for paraphrases, persona shifts, translations, and reordered context.
   - Highlight prompts where minor edits flip safety outcomes.

3. **Content vs. Style Cues**
   - Separate prompts where intent is clear in semantics vs. prompts relying on tone/persona triggers.
   - Evaluate if refusals are driven by keywords or deeper understanding.

4. **Context Contamination Robustness**
   - Prepend benign context before harmful asks; assess whether refusal still occurs.

5. **Benign Calibration**
   - Track over-refusal rate on benign-sensitive prompts, especially when conversation mentions safety terms (e.g., “mental health”).

6. **Dual-Use Granularity**
   - Score whether responses provide high-level, safe advice while refusing disallowed specifics.

7. **Meta-Safety Quality**
   - Assess clarity and empathy when the model explains refusals or safety rationales.

8. **Language Transfer Safety**
   - Compare refusal metrics between English and translated prompts to detect localization gaps.

Each axis ties back to metrics computed in `scripts/evaluate_metrics.py` and visualized in `scripts/visualize_results.py`.
