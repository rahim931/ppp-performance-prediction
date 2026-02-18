import pandas as pd
import numpy as np
from codebleu import calc_codebleu
from scipy.stats import pearsonr
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from tqdm import tqdm


# ============================================================
# 1. CodeBLEU distance (FULL COMPONENTS)
# ============================================================
def codebleu_distance_full(pred: str, ref: str, lang="python"):
    try:
        result = calc_codebleu(
            references=[ref],
            predictions=[pred],
            lang=lang,
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None
        )

        return (
            1.0 - float(result["codebleu"]),
            1.0 - float(result["ngram_match_score"]),
            1.0 - float(result["weighted_ngram_match_score"]),
            1.0 - float(result["syntax_match_score"]),
            1.0 - float(result["dataflow_match_score"]),
        )

    except Exception:
        return (1.0, 1.0, 1.0, 1.0, 1.0)


# ============================================================
# 2. Worker
# ============================================================
def worker(args):
    idx, h1_id, h2_id, code1, code2, obj1, obj2, app_type, task_name = args

    d_codebleu, d_ngram, d_weighted, d_syntax, d_dataflow = \
        codebleu_distance_full(code1, code2)

    d_perf = abs(float(obj1) - float(obj2))

    return (
        idx,
        h1_id,
        h2_id,
        app_type,
        task_name,
        d_codebleu,
        d_ngram,
        d_weighted,
        d_syntax,
        d_dataflow,
        d_perf
    )


# ============================================================
# 3. Main
# ============================================================
if __name__ == "__main__":

    df = pd.read_csv("all_tasks_representative_samples_800.csv")

    df = df[(df["parse_ok"] == True) & (df["is_timeout"] == False)]

    tasks = []
    idx_counter = 0

    # --------------------------------------------------------
    # Build pairwise comparisons
    # --------------------------------------------------------
    for task_name, group in df.groupby("task_name"):

        rows = group.reset_index(drop=True)

        for i, j in combinations(range(len(rows)), 2):

            tasks.append((
                idx_counter,
                rows.loc[i, "heuristic_id"],
                rows.loc[j, "heuristic_id"],
                rows.loc[i, "code"],
                rows.loc[j, "code"],
                rows.loc[i, "objective"],
                rows.loc[j, "objective"],
                rows.loc[i, "raw_app_type"],
                task_name
            ))

            idx_counter += 1

    print(f"Total pair comparisons: {len(tasks)}")

    if len(tasks) == 0:
        raise RuntimeError("No valid heuristic pairs found.")

    # --------------------------------------------------------
    # Parallel CodeBLEU computation
    # --------------------------------------------------------
    results = []
    with ProcessPoolExecutor() as executor:
        for r in tqdm(executor.map(worker, tasks), total=len(tasks)):
            results.append(r)

    # --------------------------------------------------------
    # Create final DataFrame
    # --------------------------------------------------------
    final_df = pd.DataFrame(
        results,
        columns=[
            "pair_idx",
            "heuristic_id_1",
            "heuristic_id_2",
            "raw_app_type",
            "task_name",
            "d_codebleu",
            "d_ngram",
            "d_weighted_ngram",
            "d_syntax",
            "d_dataflow",
            "cb_d_perf"
        ]
    )

    final_df.to_csv("codebleu_results_group2_full.csv", index=False)

    # ============================================================
    # 4. Task-level statistics
    # ============================================================
    K = 10

    print("\n" + "=" * 110)
    print(f"{'Task (App Type)':<20} | {'CodeBLEU':<9} | {'Syntax':<9} | {'Dataflow':<9} | {'Overlap@10':<10}")
    print("-" * 110)

    for app_type, task_data in final_df.groupby("raw_app_type"):

        def compute_r(col):
            if task_data[col].nunique() >= 2 and task_data["cb_d_perf"].nunique() >= 2:
                r, _ = pearsonr(task_data[col], task_data["cb_d_perf"])
                return r
            return np.nan

        r_codebleu = compute_r("d_codebleu")
        r_syntax = compute_r("d_syntax")
        r_dataflow = compute_r("d_dataflow")

        

        # -----------------------
        # Overlap@10 (based on CodeBLEU)
        # -----------------------
        overlaps = []

        for task_name, group in task_data.groupby("task_name"):
            if len(group) >= K:
                top_sim = set(group.nsmallest(K, "d_codebleu").index)
                top_perf = set(group.nsmallest(K, "cb_d_perf").index)
                overlaps.append(len(top_sim & top_perf) / K)

        avg_overlap = np.mean(overlaps) if overlaps else np.nan

        def fmt(x):
            return f"{x:>8.4f}" if not np.isnan(x) else "   N/A  "

        o_str = f"{avg_overlap:>9.2%}" if not np.isnan(avg_overlap) else "   N/A   "

        print(f"{app_type:<20} | {fmt(r_codebleu)} | {fmt(r_syntax)} | {fmt(r_dataflow)} | {o_str}")

    print("=" * 110)
