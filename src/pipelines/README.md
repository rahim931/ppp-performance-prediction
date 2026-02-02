# Pipelines

This folder contains the executable pipelines for the LLM-based experiments
using a two-stage setup:

CAP → PPP (Core Abstraction Prompting → Performance Prediction Prompting)

The pipelines are responsible for:
- loading the unified heuristic dataset
- building prompts from templates
- calling the LLM via the LLMClient interface
- parsing LLM outputs
- writing structured CSV result files

No API-specific logic (IMI, HuggingFace, OpenAI, etc.) is implemented here.
All pipelines are API-agnostic and work with any LLMClient implementation
(e.g. MockLLMClient for development).

---

## Overview

### 1) CAP Pipeline (`cap_runner.py`)

**CAP = Core Abstraction Prompting**

Purpose:
- Convert heuristic *code* into a concise, abstract description of the
  core algorithmic idea using an LLM.

Input:
- `all_heuristics_dataset.pkl` (preferred) or `.csv`
- Relevant columns:
  - `heuristic_id`
  - `raw_app_type`
  - `strategy`
  - `code`
  - `objective`
  - `parse_ok`
  - `is_timeout`

Process:
1. Iterate over heuristics
2. Build a CAP prompt using `prompt_templates/cap.md`
3. Call the LLM
4. Parse the JSON output `{ "core_idea": ... }`
5. Write results to a CAP CSV file

Output:
- CSV file with one row per heuristic containing:
  - heuristic_id
  - core algorithmic idea (LLM output)
  - LLM metadata (model, latency, prompt hash, etc.)

The CAP output is reused as input for the PPP pipeline.

---

### 2) PPP Pipeline (`ppp_runner.py`)

**PPP = Performance Prediction Prompting**

Purpose:
- Predict the objective value of a target heuristic based on similarity
  to reference heuristics with known performance values.

Input:
- Unified heuristic dataset (objective values)
- CAP results (core algorithmic ideas)

Reference selection:
- A small number of reference heuristics is selected per target heuristic.
- Default setting: **k = 3**
  - one strong heuristic (low objective)
  - one medium heuristic
  - one weak heuristic (high objective)
- References are selected within the same `raw_app_type`.

Process:
1. Load core ideas from CAP results
2. Select k reference heuristics (stratified by objective)
3. Build a PPP prompt using `prompt_templates/ppp_with_refs.md`
4. Call the LLM
5. Parse predicted objective value and confidence
6. Write results to a PPP CSV file

Output:
- CSV file with:
  - heuristic_id
  - predicted objective value (LLM output)
  - confidence (lower is better)
  - LLM metadata

---

## Notes

- CAP and PPP results are stored separately and merged later during analysis.
- Classical similarity methods (AST features, TF-IDF cosine, n-gram Jaccard, etc.)
  are computed by another team and are NOT part of these pipelines.
- All pipelines support resume mode and filtering based on dataset metadata.
