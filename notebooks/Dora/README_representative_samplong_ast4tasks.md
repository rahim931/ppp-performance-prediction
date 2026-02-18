# Representative Sampling for Heuristic Performance Analysis

## 📖 Overview
This repository contains the sampling and visualization pipeline for evaluating code similarity metrics in evolutionary heuristic design. 

The primary goal of this module is to perform **Representative Sampling** on the global `parse_ok` dataset. By creating a balanced subset of heuristics, we can:
* **Visualize Similarity**: Generate clearer heatmaps (AST, Jaccard, TF-IDF).
* **Inspect Structural Clusters**: Identify how code structure relates to performance.
* **Enable Fair Comparison**: Provide a consistent 800-sample dataset for LLM (PPP) and Classical Method evaluation.

---

## ⚙️ Sampling Strategy
To avoid data compression and bias, we implement a **Stratified Mixed Sampling** strategy for each task (BinPacking, CVRP, Premarshalling, SlidingPuzzle):

| Proportion | Category | Purpose |
| :--- | :--- | :--- |
| **30%** | **Top-Performing** | Ensure coverage of the best solutions. |
| **20%** | **Bottom-Performing** | Include poor solutions and potential timeouts for contrast. |
| **50%** | **Global Random** | Maintain structural diversity and represent the overall distribution. |

**Why this design?** Pure random sampling often misses extreme behaviors, while pure top-K sampling creates artificial similarity that inflates correlation metrics. This mixed strategy provides a realistic "stress test" for both LLMs and classical metrics.

---

## 📂 Project Structure & Input
### Input Requirements
The scripts expect a global dataset (`all_heuristics_dataset.pkl`) containing:
- `parse_ok == True` (Filtered valid Python code)
- `objective` (Numerical performance values)
- `raw_app_type` (Task classification)

### Output Files
The pipeline generates task-specific CSVs in `outputs/representative_sampling/`:
- `bin_greedy_sampled_200.csv`
- `cvrp_lns_sampled_200.csv`
- `premarshalling_astar_sampled_200.csv`
- `puzzle_astar_sampled_200.csv`
- **`all_tasks_representative_samples_800.csv`** (The unified evaluation set)

---

## 🔬 Evaluation Metrics
After sampling, we evaluate the following metrics to compare **Classical Metrics** vs. **LLM Predictions**:

1. **Pearson/Spearman Correlation**: Linear and rank relationship between similarity and performance.
2. **MAE (Mean Absolute Error)**: Prediction accuracy relative to the ground truth objective.
3. **NDCG@10**: Ability to correctly rank the top 10 most efficient heuristics.
4. **Heatmap Comparison**: Side-by-side visual analysis of AST distance vs. Performance distance.

---

## ▶️ How to Run
1. **Generate Samples**:
   ```bash
   python scripts/representative_sampling_4tasks.py