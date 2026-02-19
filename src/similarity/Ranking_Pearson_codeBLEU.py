import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

# =========================
# SETTINGS
# =========================

INPUT_FILE = "codebleu_results_group2_full.csv"

TOP_K_LIST = [5, 10]

MINIMIZE_BY_TASK = {
    "bin_greedy": True,
    "cvrp_lns": True,
    "premarshalling_astar": True,
    "puzzle_astar": True,
}

# =========================
# Overlap function (same as your first script)
# =========================

def compute_topk_overlap(pred_scores, true_scores, k, minimize=True):
    if minimize:
        true_order = np.argsort(true_scores)
    else:
        true_order = np.argsort(-true_scores)

    pred_order = np.argsort(-pred_scores)

    true_topk = set(true_order[:k])
    pred_topk = set(pred_order[:k])

    return len(true_topk & pred_topk) / k


# =========================
# Load data
# =========================

df = pd.read_csv(INPUT_FILE)

results = []

# ============================================================
# Group by task_name (important!)
# ============================================================

for task_name, task_data in df.groupby("task_name"):

    print("\n===================================")
    print("Task:", task_name)

    # --------------------------------------------------------
    # Get unique heuristic ids
    # --------------------------------------------------------
    heuristics = pd.unique(
        task_data[["heuristic_id_1", "heuristic_id_2"]].values.ravel()
    )

    heuristics = list(heuristics)
    n = len(heuristics)

    if n < 3:
        continue

    id_to_idx = {h: i for i, h in enumerate(heuristics)}

    # --------------------------------------------------------
    # Build similarity matrix from distance
    # similarity = 1 - distance
    # --------------------------------------------------------
    sim_matrix = np.zeros((n, n))

    for _, row in task_data.iterrows():

        i = id_to_idx[row["heuristic_id_1"]]
        j = id_to_idx[row["heuristic_id_2"]]

        sim = 1.0 - row["d_codebleu"]

        sim_matrix[i, j] = sim
        sim_matrix[j, i] = sim

    # diagonal self-similarity
    np.fill_diagonal(sim_matrix, 1.0)

    # --------------------------------------------------------
    # Convert similarity matrix → prediction score
    # --------------------------------------------------------
    pred_scores = sim_matrix.mean(axis=1)

    # --------------------------------------------------------
    # Get true objectives
    # --------------------------------------------------------
    # We reconstruct objectives from original CSV logic:
    # cb_d_perf = |obj1 - obj2|
    # So we must reload objective from raw dataset

    # IMPORTANT:
    # If you still have original dataset, load it.
    # Otherwise you must have objective per heuristic stored somewhere.

    # For safety, let's assume you still have:
    raw_df = pd.read_csv("all_tasks_representative_samples_800.csv")
    raw_task = raw_df[raw_df["task_name"] == task_name]

    obj_map = dict(zip(raw_task["heuristic_id"], raw_task["objective"]))
    objectives = np.array([obj_map[h] for h in heuristics])

    minimize = MINIMIZE_BY_TASK.get(task_name, True)

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------
    spearman_val, _ = spearmanr(pred_scores, objectives)
    pearson_val, _ = pearsonr(pred_scores, objectives)

    metric_row = {
        "task": task_name,
        "method": "CodeBLEU",
        "spearman": spearman_val,
        "pearson": pearson_val,
    }

    # --------------------------------------------------------
    # Overlap@K
    # --------------------------------------------------------
    for k in TOP_K_LIST:
        overlap_val = compute_topk_overlap(
            pred_scores,
            objectives,
            k,
            minimize=minimize
        )
        metric_row[f"top{k}_overlap"] = overlap_val

    results.append(metric_row)


# =========================
# Save results
# =========================

results_df = pd.DataFrame(results)
results_df.to_csv("codebleu_ranking_prediction.csv", index=False)

print("\nDONE")
print(results_df)
