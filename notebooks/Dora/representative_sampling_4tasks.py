# -*- coding: utf-8 -*-
"""
Representative Sampling for 4 Tasks
- Mixed sampling: top / bottom / random
- Automatically processes 4 tasks
- Saves per-task sampled CSV
"""

import os
import numpy as np
import pandas as pd


# =========================
# PATHS (只改這裡)
# =========================
INPUT_PKL = r"E:\Python\HRT\Project_4\all_heuristics_dataset.pkl"
OUTPUT_DIR = r"E:\Python\HRT\Project_4\outputs\representative_sampling"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# SETTINGS
# =========================
SAMPLE_SIZE = 200   # 每個 task 抽樣數量
SEED = 42

TASK_KEYS = [
    "bin_greedy",
    "cvrp_lns",
    "premarshalling_astar",
    "puzzle_astar"
]

# 是否 objective 越小越好
MINIMIZE_BY_TASK = {
    "bin_greedy": True,
    "cvrp_lns": True,
    "premarshalling_astar": True,
    "puzzle_astar": True,
}


# =========================
# Representative Sampling
# =========================
def representative_sampling(df_task, k=200, minimize=True, seed=42):
    """
    混合抽樣：
    30% top
    20% bottom
    50% random
    """
    np.random.seed(seed)

    df_task = df_task.copy()
    df_task["objective"] = pd.to_numeric(df_task["objective"], errors="coerce")
    df_task = df_task.dropna(subset=["objective"])

    # 排序
    df_sorted = df_task.sort_values(
        "objective",
        ascending=minimize
    ).reset_index(drop=True)

    n = len(df_sorted)
    k = min(k, n)

    k_top = int(k * 0.3)
    k_bottom = int(k * 0.2)
    k_random = k - k_top - k_bottom

    top_samples = df_sorted.head(k_top)
    bottom_samples = df_sorted.tail(k_bottom)

    remaining_pool = df_sorted.drop(top_samples.index).drop(bottom_samples.index)

    if len(remaining_pool) > 0:
        random_samples = remaining_pool.sample(
            n=min(k_random, len(remaining_pool)),
            random_state=seed
        )
    else:
        random_samples = pd.DataFrame()

    final_sample = pd.concat([
        top_samples,
        bottom_samples,
        random_samples
    ])

    final_sample = final_sample.sample(frac=1, random_state=seed).reset_index(drop=True)

    return final_sample


# =========================
# MAIN
# =========================
def main():

    print("Loading dataset...")
    df = pd.read_pickle(INPUT_PKL)

    summary_rows = []

    for task in TASK_KEYS:

        print(f"\nProcessing task: {task}")

        df_task = df[df["raw_app_type"] == task].copy()

        if df_task.empty:
            print("  No data found, skipping.")
            continue

        minimize = MINIMIZE_BY_TASK.get(task, True)

        df_sample = representative_sampling(
            df_task,
            k=SAMPLE_SIZE,
            minimize=minimize,
            seed=SEED
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{task}_sampled_{SAMPLE_SIZE}.csv"
        )

        df_sample.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"  Saved: {output_path}")
        print(f"  Sample size: {len(df_sample)}")

        summary_rows.append({
            "task": task,
            "original_rows": len(df_task),
            "sampled_rows": len(df_sample),
            "minimize_objective": minimize
        })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "sampling_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nDONE")
    print("Summary saved:", summary_path)


if __name__ == "__main__":
    main()
