# AST Structural Feature Analysis for SlidingPuzzle

This script performs **structural AST-based analysis** on heuristics from the **SlidingPuzzle (puzzle_astar)** task.

It extracts **compressed AST feature vectors** from heuristic code, computes **structural similarity** between heuristics, and evaluates how well structural similarity aligns with **performance similarity**.

The analysis is intended as a **sanity / consistency check**, rather than a predictive model.

---

## Overview

Given a filtered heuristic dataset for SlidingPuzzle, the script:

- Parses heuristic code into **Abstract Syntax Trees (ASTs)**
- Extracts **lightweight structural features** from ASTs
- Computes pairwise **structural distances** (cosine)
- Computes pairwise **performance distances** (absolute objective difference)
- Evaluates alignment between structure and performance using:
  - Pearson correlation
  - Mantel test
  - Overlap@K

---

## Input

### Dataset

The script expects a **filtered and parseable dataset**, preferably in pickle format:


(Optional, for inspection only)

### Required Columns

- `heuristic_id` — unique identifier of the heuristic
- `code` — Python source code of the heuristic
- `objective` — scalar performance value (assumed minimization)

Rows with missing or non-numeric objectives are removed automatically.

---

## AST Feature Representation

For each heuristic, the code is parsed into an AST and summarized using **compressed structural features**, including:

### Node Statistics
- Total number of AST nodes
- Per-node-type counts (bag-of-nodes representation)

### Tree Shape Statistics
- Maximum AST depth
- Average AST depth
- Maximum branching factor
- Average branching factor

Heuristics that fail AST parsing are automatically filtered out.

---

## Distance Computation

### Structural Distance
- Feature vectors are compared using **cosine distance**
- Result: `distance_struct.npy`

### Performance Distance
- Pairwise absolute difference of objective values
- Result: `distance_perf.npy`

---

## Alignment Metrics

The script computes several metrics to assess whether **structural similarity reflects performance similarity**.

### Pearson Correlation
- Computed between the upper triangles of the two distance matrices

### Mantel Test
- Pearson correlation between distance matrices
- Permutation-based p-value (default: 999 permutations)

### Overlap@K
- For each heuristic:
  - Top-K nearest neighbors by structural distance
  - Top-K nearest neighbors by performance distance
- Reports average overlap ratio
- Default: `K = 10`

---

## Outputs

All outputs are written to:


### Generated Files

- `features.parquet`  
  AST feature vectors for all parseable heuristics

- `distance_struct.npy`  
  Pairwise structural distance matrix (cosine)

- `distance_perf.npy`  
  Pairwise performance distance matrix (absolute difference)

- `metrics.json`  
  Summary statistics, including:
  - Pearson correlation
  - Mantel statistic and p-value
  - Overlap@10
  - Dataset size and parsing success rate

---

## Interpretation Notes

- This script **does not train a model** and **does not predict performance**
- All metrics are intended for **diagnostic and exploratory analysis**
- Objectives are treated as **minimization** by default
- Structural similarity does not imply causal performance similarity

---

## Configuration

Key parameters can be adjusted at the top of the script:

- `OVERLAP_K` — neighborhood size for overlap analysis
- `MANTEL_PERMUTATIONS` — number of permutations in Mantel test
- `RANDOM_SEED` — reproducibility control

---

## Intended Use

This analysis is designed to support:
- Evaluation of AST-based similarity measures
- Comparison against LLM-based similarity or ranking methods
- Structural sanity checks before downstream modeling
