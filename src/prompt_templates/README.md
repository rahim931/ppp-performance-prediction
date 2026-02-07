# Prompt Templates

This folder contains all prompt templates used by the LLM pipelines.

Templates are plain text files with placeholders that are filled
by the pipeline runners.

They are intentionally kept separate from code to ensure:
- reproducibility
- easy prompt iteration
- clear documentation of experimental setups

---

## Templates

### 1) `cap.md` – Core Abstraction Prompting (CAP)

Purpose:
- Extract the *core algorithmic idea* of a heuristic from its Python code.

Input to the prompt:
- Full heuristic code (`{code}`)

Output (JSON, enforced by the prompt):
```json
{
  "core_idea": "<concise abstraction>"
}
```
Notes:
Focuses on what the heuristic does, not implementation details
Does NOT evaluate performance
Used once per heuristic
Output is reused by the PPP pipeline

### 2) ppp_with_refs.md – Performance Prediction Prompting (PPP)

Purpose:
- Predict the objective value of a target heuristic based on similarity to reference heuristics.

Input to the prompt:
- Dataset / application type
- Core ideas of reference heuristics + their objective values
- Core idea of the target heuristic

Output (JSON, enforced by the prompt):
```json
{
  "prediction": <float>,
  "confidence": <float>,
  "justification": "<short explanation>"
}
```
Notes:
- Lower values are better for both prediction and confidence
- The number of reference heuristics (k) is controlled in the pipeline code
- Designed to work with small k (e.g. 3) as well as larger settings

---

### General Notes
- All templates enforce machine-readable JSON output
- Templates should not contain any API- or model-specific assumptions
- Any change to a template changes the experimental setup and should be documented accordingly
