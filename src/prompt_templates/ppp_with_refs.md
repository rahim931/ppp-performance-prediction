You are an expert in algorithm analysis and heuristic optimization.

You are given several reference heuristics with known performance values (objective).
Lower objective values are BETTER.
The target heuristic has NO known objective value.

You MUST base your prediction primarily on similarity of algorithmic ideas and mechanisms.

Dataset / App Type: {raw_app_type}

Reference Heuristics (core ideas + objective):

{references_block}

Target Heuristic:
Core Idea:
{target_core}

Task:
- Predict the objective value of the target heuristic.
- Also predict a confidence score for this prediction.
- For BOTH values: lower is better.

Output format (JSON only, single object on last line):
{
  "prediction": <float>,
  "confidence": <float>,
  "justification": "<short explanation>"
}
