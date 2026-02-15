import pandas as pd
import numpy as np
from codebleu import calc_codebleu
from scipy.stats import pearsonr
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


# ============================================================
# 1. CodeBLEU distance
# ============================================================
def codebleu_distance(pred: str, ref: str, lang="python"):
    """
    1 - CodeBLEU as distance:
    0.0 -> identical code
    1.0 -> completely different
    """
    try:
        result = calc_codebleu(
            references=[ref],
            predictions=[pred],
            lang=lang,
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None
        )
        return 1.0 - float(result["codebleu"])
    except Exception:
        return 1.0


# ============================================================
# 2. Worker (for multiprocessing)
# ============================================================
def worker(args):
    idx, main_code, neighbor_code, main_obj, neighbor_obj = args

    d_sim = codebleu_distance(neighbor_code, main_code)
    cb_score = 1.0 - d_sim
    d_perf = abs(float(neighbor_obj) - float(main_obj))

    return idx, d_sim, cb_score, d_perf


# ============================================================
# 3. Main
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    groups_df = pd.read_csv("ast_groups.csv")

    # ALL heuristics (main + neighbor)
    all_df = pd.read_csv("all_heuristics_dataset.csv")

    # Build lookup tables
    code_dict = all_df.set_index("heuristic_id")["code"].to_dict()
    obj_dict = all_df.set_index("heuristic_id")["objective"].to_dict()

    # --------------------------------------------------------
    # Build tasks
    # --------------------------------------------------------
    tasks = []
    for idx, row in groups_df.iterrows():
        m_id = row["main_heuristic_id"]
        n_id = row["neighbor_heuristic_id"]

        if m_id in code_dict and n_id in code_dict:
            tasks.append((
                idx,
                code_dict[m_id],
                code_dict[n_id],
                obj_dict[m_id],
                obj_dict[n_id]
            ))

    print(f"Total group pairs : {len(groups_df)}")
    print(f"Valid code pairs  : {len(tasks)}")

    if len(tasks) == 0:
        raise RuntimeError("No valid (main, neighbor) pairs with code found.")

    # --------------------------------------------------------
    # Parallel CodeBLEU computation
    # --------------------------------------------------------
    results = []
    with ProcessPoolExecutor() as executor:
        for r in tqdm(executor.map(worker, tasks), total=len(tasks)):
            results.append(r)

    res_df = pd.DataFrame(
        results,
        columns=["idx", "cb_d_sim", "codebleu_score", "cb_d_perf"]
    )

    final_df = (
        groups_df
        .merge(res_df, left_index=True, right_on="idx")
        .drop(columns=["idx"])
    )

    final_df.to_csv("codebleu_results_analysis.csv", index=False)

    # ============================================================
    # 4. Task-level statistics
    # ============================================================
    print("\n" + "=" * 72)
    print(f"{'Task (App Type)':<25} | {'Pearson R':<10} | {'Overlap@3':<10}")
    print("-" * 72)

    for app_type, task_data in final_df.groupby("app_type"):

        # -----------------------
        # Pearson (group-level)
        # -----------------------
        r_vals = []
        for _, group in task_data.groupby("group_id"):
            if len(group) >= 2:
                if group["cb_d_sim"].nunique() < 2 or group["cb_d_perf"].nunique() < 2:
                    continue                                                #Minimum sample size satisfied
                r, _ = pearsonr(group["cb_d_sim"], group["cb_d_perf"])
                if not np.isnan(r):
                    r_vals.append(r)

        avg_r = np.mean(r_vals) if r_vals else np.nan

        # -----------------------
        # Overlap@K
        # -----------------------
        K = 3
        overlaps = []

        for _, group in task_data.groupby("group_id"):
            if len(group) >= K:
                top_sim = set(group.nsmallest(K, "cb_d_sim").index)
                top_perf = set(group.nsmallest(K, "cb_d_perf").index)
                overlaps.append(len(top_sim & top_perf) / K)

        avg_overlap = np.mean(overlaps) if overlaps else np.nan

        r_str = f"{avg_r:>9.4f}" if not np.isnan(avg_r) else "   N/A   "
        o_str = f"{avg_overlap:>9.2%}" if not np.isnan(avg_overlap) else "   N/A   "

        print(f"{app_type:<25} | {r_str} | {o_str}")

    print("=" * 72)
