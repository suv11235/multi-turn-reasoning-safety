# Notes on Microsoft Crescendo Dataset

## Quick Summary
- **Crescendo** introduces staged safety evaluations where prompts escalate in risk.
- Data is grouped by **scenario families** (e.g., disallowed content, jailbreaks, benign controls) with human-written annotations.
- Emphasis on **human agreement** and **policy-referenced** refusals.

## Insights for This Project
1. **Scenario layering**: Crescendo uses incremental variants of a base prompt. We mirror this through `core_*` vs `diag_*` slices and `parent_id` links.
2. **Policy grounding**: Crescendo annotations reference policy sections. Our scoring scheme tracks `rationale_quality` to encourage similarly grounded refusals.
3. **Balance**: Crescendo intermixes disallowed and benign prompts to deter over-triggering. We extend that idea with dual-use and borderline slices.
4. **Limitations for Distilled Models**: The original dataset targets full-size models; distilled variants may under-refuse due to compressed safety signals. Hence we explicitly document distillation-specific axes in `docs/evaluation_axes.md`.

## Adaptation Checklist
- ✔ Extract scenario families relevant to jailbreaks, dual-use, benign controls.
- ✔ Define metadata-rich prompt schema for traceability.
- ☐ Map Crescendo policy references to our `notes` field where appropriate (future work once policy set is finalized).
