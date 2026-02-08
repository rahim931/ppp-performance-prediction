You are an expert in algorithm analysis and heuristic optimization.

Here are example heuristics and their objective values.
Lower objective values are better.

Dataset / App Type: {raw_app_type}

Reference Heuristics:
{references_block}

Target Heuristic (code):
```python
{target_code}

```
Task:
1. Compare the target code to the reference codes and identify which reference(s) it is most similar to semantically.
2. Predict the target objective by interpolating between the most relevant reference objectives.
3. Output a confidence in [0, 1]. Higher means more confident.
4. The predicted objective MUST lie within [{task_min}, {task_max}].
Output format (JSON only, single object on last line):
{{
"prediction": <float>,
"confidence": <float>
}}
