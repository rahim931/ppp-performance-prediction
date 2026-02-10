# AST Similarity Grouping (CSV Export for LLM Similarity Alignment)

This script generates **5–10 similarity groups per application type** from a global `parse_ok` heuristic dataset.

Each group contains:
- **1 main heuristic**
- **3–5 heuristics** that are most structurally similar to the main one  
  (based on **AST structural features**)

The output is a **single CSV** that can be used to check:
- whether **LLM similarity scores** recover the same “similar” sets, and
- whether **structurally similar heuristics** also exhibit similar objective values.

---

## Motivation

Our classical baselines (**AST / TF-IDF / Jaccard**) produce **pairwise similarity or distance scores**,  
rather than a single scalar prediction per heuristic.

Therefore, we provide **group-based outputs** for a consistency / sanity check:

- If **AST similarity** indicates that a set of heuristics is structurally similar,
- we verify whether **LLM similarity** also identifies them as similar.

Additionally, we inspect whether their **objective values** are close.

---

## Input

The script expects a **global, filtered, parseable dataset**, for example:


### Required Columns
- `heuristic_id`
- `raw_app_type`
- `strategy`
- `code`
- `objective`

If a `parse_ok` column exists, only rows with `parse_ok = True` are kept.

---

## What the Script Does

For **each `raw_app_type` (task)**:

1. **Stratify by objective** into quantile bins  
   - Default: **3 bins** → `low / mid / high`

2. **Group within each bin by strategy**  
   - e.g. `e1`, `e2`, `i1`, `m1`, `m2`

3. **Sample main heuristics**
   - Default: **10 groups per task**
   - Reproducible via a fixed random seed

4. For each **main heuristic**:
   - Parse its code into an **AST**
   - Compute a **compressed structural AST feature**
     - node-type count vector
   - Find the **top-K nearest neighbors**
     - Default: `K = 5`
     - Similarity metric: **cosine similarity**

5. **Export results as CSV**
   - Each row represents a **main–neighbor relation**

---

## Outputs

The script writes **three files** under `OUTPUT_DIR`.

### 1️⃣ `ast_groups.csv` (Main Deliverable)

Each row represents **one main heuristic and one similar neighbor**.

#### Key Columns
- `app_type`  
  (e.g. `bin_greedy`, `cvrp_lns`, …)
- `group_id`  
  (starts at 1 per `app_type`)
- `objective_bin`  
  (`low / mid / high`)

**Main heuristic information**
- `main_heuristic_id`
- `main_strategy`
- `main_objective`
- `main_obj_rank`
- `main_obj_percentile`
- `main_is_good_top10pct`  
  (whether the main is in the top 10% by objective rank)

**Neighbor information**
- `neighbor_rank`  
  (1..K by similarity)
- `neighbor_heuristic_id`
- `neighbor_strategy`
- `neighbor_objective`
- `ast_cosine_sim`  
  (AST structural similarity; closer to 1 = more similar)
- `neighbor_obj_diff`  
  (`neighbor objective - main objective`)
- `neighbor_is_better_than_main`  
  (whether the neighbor outperforms the main)

---

### 2️⃣ `selected_mains.csv`

List of sampled **main heuristics**,  
provided for **traceability and reproducibility**.

---

### 3️⃣ `summary.csv`

Per-task summary, including:
- dataset size
- number of mains selected
- number of output rows
- which strategies and objective bins were covered

---

## Performance Labeling

This script **does not predict performance**.  
It only adds **lightweight, interpretable labels**.

- By default, objectives are treated as **minimization**
  - smaller is better
- `main_is_good_top10pct = True`
  - the main heuristic is in the **top 10%** within that task
- `neighbor_is_better_than_main = True`
  - the neighbor has a **better objective** than the main

If any task uses **maximization**, adjust `MINIMIZE_BY_APP_TYPE` accordingly.

---

## Requirement Fit

This output satisfies the team requirements:

- **5–10 groups per app type** ✅  
- **1 main + 3–5 similar heuristics per group** ✅  
- **CSV with group IDs and heuristic IDs** ✅  
- **Based on AST structural similarity** ✅  
- **Supports alignment checks vs LLM similarity and objective consistency** ✅
