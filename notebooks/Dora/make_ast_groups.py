# -*- coding: utf-8 -*-
"""
Create representative AST-similarity groups per task:
- Stratify by objective (quantiles -> low/mid/high)
- Within each bin, group by strategy and sample main heuristics
- For each main, find top-K nearest neighbors using AST node-type feature vectors (compressed)
- Export ONE CSV with group_id + heuristic ids, plus summary files.

Author: (you)
"""

import os
import ast
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# PATHS (只改這裡)
# =========================
FRIEND_GLOBAL_PKL = r"E:\Python\HRT\Project_4\all_heuristics_dataset.pkl"
OUTPUT_DIR = r"E:\Python\HRT\Project_4\outputs\ast_groups_for_llm_check"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# SETTINGS
# =========================
RANDOM_SEED = 42
GROUPS_PER_TASK = 10          # 每個 app_type 5–10 組（預設 10）
NEIGHBORS_PER_GROUP = 5       # 每組 3–5 個相似（預設 5）
OBJECTIVE_BINS = 3            # low/mid/high
TOP_GOOD_PCT = 0.10           # 「表現好」定義為 Top 10%

# objective 越小越好通常成立；若某 task 是越大越好，改成 False
MINIMIZE_BY_APP_TYPE = {
    "bin_greedy": True,
    "cvrp_lns": True,
    "premarshalling_astar": True,
    "puzzle_astar": True,
}

# 你想處理的四個 task（以 raw_app_type 分組最穩）
TASK_KEYS = ["bin_greedy", "cvrp_lns", "premarshalling_astar", "puzzle_astar"]


# =========================
# Helpers
# =========================
def safe_str(x) -> str:
    return "" if x is None else str(x)

def ast_node_counts(code: str) -> Dict[str, int]:
    """AST feature: count node types (compressed structural vector)."""
    try:
        tree = ast.parse(code)
    except Exception:
        return {}
    counts: Dict[str, int] = {}
    for node in ast.walk(tree):
        t = type(node).__name__
        counts[t] = counts.get(t, 0) + 1
    return counts

def compute_objective_rank(df: pd.DataFrame, minimize: bool) -> pd.Series:
    """Rank objective within df. rank=1 is best."""
    if minimize:
        return df["objective"].rank(method="min", ascending=True).astype(int)
    else:
        return df["objective"].rank(method="min", ascending=False).astype(int)

def is_better(a: float, b: float, minimize: bool) -> bool:
    """Return True if a is better than b."""
    return (a < b) if minimize else (a > b)

def stratify_objective_bins(df: pd.DataFrame, bins: int) -> pd.Series:
    """
    Create bins = 3 -> low/mid/high using quantiles.
    If duplicates make qcut fail, fallback to cut.
    """
    labels = ["low", "mid", "high"] if bins == 3 else [f"bin{i}" for i in range(bins)]
    try:
        return pd.qcut(df["objective"], q=bins, labels=labels, duplicates="drop")
    except Exception:
        return pd.cut(df["objective"], bins=bins, labels=labels, include_lowest=True)

def pick_mains_stratified(df: pd.DataFrame, groups_target: int, seed: int) -> pd.DataFrame:
    """
    Pick main heuristics using:
    objective_bin -> strategy -> sample,
    with coverage constraint to reduce redundancy.
    """
    rng = random.Random(seed)
    df = df.copy()

    df["objective_bin"] = stratify_objective_bins(df, OBJECTIVE_BINS).astype(str)

    # Build strata list
    strata = []
    for (obin, strat), sub in df.groupby(["objective_bin", "strategy"]):
        sub_ids = sub["heuristic_id"].tolist()
        if len(sub_ids) > 0:
            strata.append((obin, strat, sub))

    # Shuffle strata to avoid deterministic bias across bins/strategies
    rng.shuffle(strata)

    picked_rows = []
    used_ids = set()

    # Round-robin across strata until we have enough mains
    while len(picked_rows) < groups_target:
        progressed = False
        for (obin, strat, sub) in strata:
            if len(picked_rows) >= groups_target:
                break
            # candidates not used yet
            candidates = sub[~sub["heuristic_id"].isin(used_ids)]
            if candidates.empty:
                continue
            # sample one
            row = candidates.sample(n=1, random_state=rng.randint(0, 10**9)).iloc[0]
            picked_rows.append(row)
            used_ids.add(row["heuristic_id"])
            progressed = True
            if len(picked_rows) >= groups_target:
                break
        if not progressed:
            # No more unique candidates; stop early
            break

    mains = pd.DataFrame(picked_rows).reset_index(drop=True)
    # Ensure group_id starts at 1
    mains["group_id"] = np.arange(1, len(mains) + 1)
    return mains


def neighbors_by_ast_cosine(df: pd.DataFrame, mains: pd.DataFrame, k: int) -> List[dict]:
    """
    Compute AST node-type vectors for all rows in df,
    then for each main find top-k cosine-similar neighbors.
    """
    # Build feature dicts
    feat_dicts = [ast_node_counts(code) for code in df["code"].tolist()]
    # Vectorize
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(feat_dicts)  # shape: (N, F)

    # Precompute objective ranks/percentiles
    minimize = MINIMIZE_BY_APP_TYPE.get(df["raw_app_type"].iloc[0], True)
    obj_rank = compute_objective_rank(df, minimize=minimize)
    df = df.copy()
    df["obj_rank"] = obj_rank
    df["obj_percentile"] = (df["obj_rank"] / len(df)).astype(float)

    # Map id -> index
    id_to_idx = {hid: i for i, hid in enumerate(df["heuristic_id"].tolist())}

    rows_out: List[dict] = []

    for _, m in mains.iterrows():
        mid = m["heuristic_id"]
        if mid not in id_to_idx:
            continue
        mi = id_to_idx[mid]

        # cosine similarity of main to all
        sims = cosine_similarity(X[mi], X).ravel()
        # exclude self
        sims[mi] = -1.0

        # top-k indices
        top_idx = np.argpartition(-sims, kth=min(k, len(sims)-1))[:k]
        # sort by similarity
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        main_row = df.iloc[mi]
        main_obj = float(main_row["objective"])
        main_rank = int(main_row["obj_rank"])
        main_pct = float(main_row["obj_percentile"])
        main_good = main_pct <= TOP_GOOD_PCT

        for rnk, j in enumerate(top_idx, start=1):
            nb_row = df.iloc[int(j)]
            nb_obj = float(nb_row["objective"])
            rows_out.append({
                "app_type": safe_str(df["raw_app_type"].iloc[0]),
                "group_id": int(m["group_id"]),
                "objective_bin": safe_str(m.get("objective_bin", "")),
                "main_heuristic_id": safe_str(mid),
                "main_strategy": safe_str(main_row.get("strategy", "")),
                "main_objective": main_obj,
                "main_obj_rank": main_rank,
                "main_obj_percentile": round(main_pct, 6),
                "main_is_good_top10pct": bool(main_good),

                "neighbor_rank": int(rnk),
                "neighbor_heuristic_id": safe_str(nb_row["heuristic_id"]),
                "neighbor_strategy": safe_str(nb_row.get("strategy", "")),
                "neighbor_objective": nb_obj,

                "ast_cosine_sim": float(sims[int(j)]),
                "neighbor_obj_diff": nb_obj - main_obj,
                "neighbor_is_better_than_main": bool(is_better(nb_obj, main_obj, minimize=minimize)),
            })

    return rows_out


# =========================
# MAIN
# =========================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Loading friend's global dataset:", FRIEND_GLOBAL_PKL)
    df = pd.read_pickle(FRIEND_GLOBAL_PKL)

    # Basic checks
    required_cols = ["heuristic_id", "raw_app_type", "strategy", "code", "objective"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Ensure types
    df = df.copy()
    df["heuristic_id"] = df["heuristic_id"].astype(str)
    df["strategy"] = df["strategy"].astype(str)
    df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
    df = df.dropna(subset=["objective"])
    df["code"] = df["code"].astype(str)

    # Optional: if dataset has parse_ok already, you can enforce it
    if "parse_ok" in df.columns:
        df = df[df["parse_ok"] == True].copy()

    outputs_all = []
    summaries = []
    mains_all = []

    print("\nCreating groups per task...")
    for task_key in TASK_KEYS:
        dft = df[df["raw_app_type"] == task_key].copy()
        if dft.empty:
            print(f"  [SKIP] No rows found for {task_key}")
            continue

        # Basic cleaning
        dft = dft[dft["code"].str.strip() != ""].copy()
        dft = dft.drop_duplicates(subset=["heuristic_id"], keep="first").copy()

        minimize = MINIMIZE_BY_APP_TYPE.get(task_key, True)

        print(f"\n--- Task: {task_key} ---")
        print(f"Rows available: {len(dft)} | minimize objective: {minimize}")

        # Pick mains
        mains = pick_mains_stratified(dft, groups_target=GROUPS_PER_TASK, seed=RANDOM_SEED)
        if mains.empty:
            print("  [WARN] No mains selected (check strata).")
            continue

        # Add objective bin info already computed in mains
        mains_all.append(mains.assign(app_type=task_key))

        # Build groups using AST cosine neighbors
        group_rows = neighbors_by_ast_cosine(dft, mains, k=NEIGHBORS_PER_GROUP)
        outputs_all.extend(group_rows)

        # Summary
        summaries.append({
            "app_type": task_key,
            "rows_total": int(len(dft)),
            "mains_selected": int(len(mains)),
            "neighbors_per_main": int(NEIGHBORS_PER_GROUP),
            "rows_in_output": int(len(group_rows)),
            "objective_bins_used": ",".join(sorted(set(mains["objective_bin"].tolist()))),
            "strategies_used": ",".join(sorted(set(mains["strategy"].tolist()))),
        })

        print(f"Selected mains: {len(mains)} | Output rows: {len(group_rows)}")

    # Save outputs
    out_csv = os.path.join(OUTPUT_DIR, "ast_groups.csv")
    sum_csv = os.path.join(OUTPUT_DIR, "summary.csv")
    mains_csv = os.path.join(OUTPUT_DIR, "selected_mains.csv")

    out_df = pd.DataFrame(outputs_all)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    pd.DataFrame(summaries).to_csv(sum_csv, index=False, encoding="utf-8-sig")

    if mains_all:
        pd.concat(mains_all, ignore_index=True).to_csv(mains_csv, index=False, encoding="utf-8-sig")

    print("\nDONE")
    print("Saved:", out_csv)
    print("Saved:", sum_csv)
    print("Saved:", mains_csv)


if __name__ == "__main__":
    main()
