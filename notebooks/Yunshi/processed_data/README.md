
# Evolutionary Heuristic Design Data Pipeline

## Overview

This pipeline processes heuristic-generation outputs stored as JSON files and constructs clean, auditable datasets for downstream analysis.

It recursively scans the project data directory, applies exception-based filtering rules, extracts relevant sub-fields from each JSON file, and produces both a global dataset and task-specific datasets.

All outputs are saved in both `.pkl` and `.csv` formats.

---

## What the Pipeline Does

1. **Recursive scanning**

   * Traverses the project `data/` directory.
   * Automatically discovers all folders named `all_programs`.

2. **Filtering**

   * Excludes files whose filenames contain `_Exception`.
   * Excludes JSON files with `"exception": true` at the top level.
   * Drops records with missing offspring data, invalid objectives, empty code, or duplicate heuristic IDs.

3. **Extraction**

   * Extracts key fields from each valid JSON file, including heuristic metadata, generated code, and objective values.

4. **Dataset construction**

   * Merges all cleaned records into a single global dataset.
   * Splits the dataset into per-task datasets using normalized task names.

---

## Input Data Layout

The pipeline assumes the following directory structure under the project root:

```
ppp-performance-prediction/
├── data/
│   ├── bin_greedy/
│   │   └── <instance>/
│   │       └── all_programs/
│   │           └── *.json
│   ├── cvrp_lns/
│   │   └── <instance>/
│   │       └── all_programs/
│   │           └── *.json
│   ├── premarshalling_astar/
│   └── puzzle_astar/
```

Only directories named `all_programs` are scanned. All other folders are ignored.

---

## Output Files

### Global Dataset

```
data/processed/
├── all_heuristics_dataset.pkl
├── all_heuristics_dataset.csv
```

This dataset contains all cleaned heuristic records across all tasks.

### Per-task Datasets

```
data/processed/per_task/
├── BinPacking.pkl
├── BinPacking.csv
├── CVRP.pkl
├── CVRP.csv
├── Premarshalling.pkl
├── Premarshalling.csv
├── SlidingPuzzle.pkl
└── SlidingPuzzle.csv
```

Each per-task file contains only the records belonging to a single normalized task.

---

## Dataset Schema

The global and per-task datasets share the same schema.
Each row corresponds to a single heuristic instance.

| Column name      | Description                                                                           |
| ---------------- | ------------------------------------------------------------------------------------- |
| `heuristic_id`   | Unique identifier of the heuristic (derived from offspring ID, fallback to filename). |
| `raw_app_type`   | Raw task name inferred from the directory structure.                                  |
| `instance_scale` | Instance-scale directory name (problem size or variant).                              |
| `filename`       | Source JSON filename.                                                                 |
| `strategy`       | Strategy identifier parsed from the filename.                                         |
| `algorithm`      | Algorithm name or tag stored in the JSON metadata.                                    |
| `code`           | Generated heuristic source code.                                                      |
| `objective`      | Objective value reported by the heuristic.                                            |
| `task_name`      | Normalized task name used for grouping (e.g., BinPacking, CVRP).                      |
| `is_timeout`     | Boolean flag indicating timeout-like behavior under task-specific thresholds.         |

---

## Execution Assumptions

* The pipeline is typically executed from within the `notebooks/` directory.
* The project root is inferred relative to the current working directory.
* The raw input data must be located under `data/`.
* Output files are written to `data/processed/`.

---

## Notes

* All outputs are deterministic given the same input data.
* Audit statistics are printed at the end of execution to summarize filtering and retention.
* Both `.pkl` and `.csv` formats are provided for flexibility in downstream usage.

