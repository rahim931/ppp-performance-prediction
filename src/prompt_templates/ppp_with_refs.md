You are an expert in algorithm analysis and heuristic optimization.

You are given several reference heuristics with known performance values (objectives).
Lower objective values indicate better performance.

Your task is to predict the performance of a target heuristic by
COMPARING its algorithmic idea to the provided reference heuristics.

IMPORTANT CONSTRAINTS:
- The predicted objective MUST lie within the range
  [min(reference objectives), max(reference objectives)].
- Do NOT extrapolate beyond the reference scores.
- Base your prediction on semantic and algorithmic similarity.

Dataset / App Type: {raw_app_type}

Task Objective Range (global, for this app type):
- min_objective: {task_min}
- max_objective: {task_max}

Reference Heuristics:
Each reference consists of a core algorithmic idea and its known objective value.

{references_block}

Target Heuristic:
Core Algorithmic Idea:
{target_core}

Prediction Task:
1. Predict the objective value of the target heuristic.
   - The prediction MUST lie within [{task_min}, {task_max}].
   - Do NOT extrapolate outside this range.
2. Predict a confidence score in [0, 1].
   - The confidence represents how semantically similar the target heuristic
     is to the most relevant reference heuristic.
   - Higher confidence means higher similarity.

Output format (JSON only, single object on last line):
{{
  "prediction": <float>,
  "confidence": <float>,
  "justification": "<short explanation referencing the closest reference>"
}}
