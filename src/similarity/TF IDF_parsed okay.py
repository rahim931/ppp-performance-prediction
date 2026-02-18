# -*- coding: utf-8 -*-
"""
Per-task TF-IDF + Cosine analysis from global parse_ok dataset.

- Loads: E:\Python\HRT\Project_4\all_heuristics_dataset.pkl
- Prints unique values for potential task columns (so you can verify naming)
- Splits into 4 tasks and runs TF-IDF+cosine per task
- Saves per-task matrices + metrics.json + counts.json

Tasks you want:
  - bin_greedy
  - cvrp_lns
  - premarshalling_astar
  - puzzle_astar

  - raw_app_type: e.g., "bin_greedy", "cvrp_lns", ...
  - task_name: e.g., "BinPacking", "CVRP", "Premarshalling", "SlidingPuzzle"
We will auto-detect and map both possibilities.
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from skbio.stats.distance import mantel


# =========================
# PATHS
# =========================
INPUT_PKL = r"E:\Python\HRT\Project_4\all_heuristics_dataset.pkl"
OUTPUT_ROOT = r"E:\Python\HRT\Project_4\outputs\tfidf_per_task_from_friend_parseok"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


# =========================
# SETTINGS
# =========================
TFIDF_MAX_FEATURES = 20000
TOKEN_PATTERN = r"[A-Za-z_][A-Za-z_0-9]*"
MANTEL_PERMUTATIONS = 999
TOPK = 10

# You want 4 tasks; we support both raw_app_type naming and normalized task_name.
RAW_APP_TASKS = ["bin_greedy", "cvrp_lns", "premarshalling_astar", "puzzle_astar"]
TASK_NAME_MAP = {
    "bin_greedy": "BinPacking",
    "cvrp_lns": "CVRP",
    "premarshalling_astar": "Premarshalling",
    "puzzle_astar": "SlidingPuzzle",
}


# =========================
# HELPERS
# =========================
def overlap_at_k(sim_a: np.ndarray, sim_b: np.ndarray, k: int = 10) -> float:
    """
    Mean neighbor overlap@k between two similarity matrices.
    Assumes higher values = more similar.
    """
    n = sim_a.shape[0]
    overlaps = []
    for i in range(n):
        nn_a = np.argsort(-sim_a[i])[1:k + 1]  # exclude self
        nn_b = np.argsort(-sim_b[i])[1:k + 1]
        overlaps.append(len(set(nn_a) & set(nn_b)) / k)
    return float(np.mean(overlaps))


def run_tfidf_cosine_metrics(df_task: pd.DataFrame, out_dir: str) -> None:
    """
    Run TF-IDF + cosine similarity, compute distance correlations + overlap@10,
    and save outputs to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    df_task = df_task.reset_index(drop=True)
    n = len(df_task)

    # Basic sanity
    if n < 3:
        print(f"[WARN] Too few rows ({n}). Skip.")
        return

    codes = df_task["code"].astype(str).tolist()
    objectives = df_task["objective"].astype(float).values

    print(f"  Rows: {n}")

    # TF-IDF
    vectorizer = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN,
        max_features=TFIDF_MAX_FEATURES
    )
    X = vectorizer.fit_transform(codes)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"  TF-IDF vocab size: {vocab_size}")

    # Similarity (for overlap)
    sim_code = cosine_similarity(X)  # diag ~ 1

    # Performance distance & similarity
    dist_perf = np.abs(objectives[:, None] - objectives[None, :]).astype(np.float64)
    sim_perf = (-dist_perf).astype(np.float64)  # for overlap

    # Distances (for Mantel)
    dist_code = (1.0 - sim_code).astype(np.float64)
    np.fill_diagonal(dist_code, 0.0)
    np.fill_diagonal(dist_perf, 0.0)

    # Correlations on upper triangle (distance matrices)
    iu = np.triu_indices(n, k=1)
    pearson_r, pearson_p = pearsonr(dist_code[iu], dist_perf[iu])

    mantel_r, mantel_p, _ = mantel(
        dist_code,
        dist_perf,
        method="pearson",
        permutations=MANTEL_PERMUTATIONS
    )

    # Overlap@k (similarity matrices)
    overlap_k = overlap_at_k(sim_code, sim_perf, k=TOPK)

    # Save matrices
    np.save(os.path.join(out_dir, "tfidf_cosine_similarity.npy"), sim_code)
    np.save(os.path.join(out_dir, "code_distance_tfidf_cosine.npy"), dist_code)
    np.save(os.path.join(out_dir, "performance_distance.npy"), dist_perf)

    # Save IDs for alignment/debug
    df_task[["heuristic_id"]].to_csv(os.path.join(out_dir, "ids.csv"), index=False)

    # Save metrics
    metrics = {
        "num_heuristics": int(n),
        "pearson_r_distance": float(pearson_r),
        "pearson_p_distance": float(pearson_p),
        "mantel_r_distance": float(mantel_r),
        "mantel_p_distance": float(mantel_p),
        "overlap_at_{}".format(TOPK): float(overlap_k),
        "tfidf_vocab_size": int(vocab_size),
        "tfidf_max_features": int(TFIDF_MAX_FEATURES),
        "token_pattern": TOKEN_PATTERN,
        "mantel_permutations": int(MANTEL_PERMUTATIONS),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, "counts.json"), "w", encoding="utf-8") as f:
        json.dump({"num_heuristics": int(n)}, f, indent=2)

    print(f"  Pearson r (dist): {pearson_r:.6f} (p={pearson_p:.3e})")
    print(f"  Mantel  r (dist): {mantel_r:.6f} (p={mantel_p:.3e})")
    print(f"  Overlap@{TOPK}    : {overlap_k:.6f}")
    print(f"  Saved to: {out_dir}")


def print_unique_preview(df: pd.DataFrame, col: str, max_show: int = 30) -> None:
    if col not in df.columns:
        print(f"- {col}: (column not found)")
        return
    vals = df[col].dropna().astype(str).unique().tolist()
    vals_sorted = sorted(vals)
    print(f"- {col}: {len(vals_sorted)} unique")
    for v in vals_sorted[:max_show]:
        print(f"    {v}")
    if len(vals_sorted) > max_show:
        print(f"    ... (showing first {max_show})")


# =========================
# LOAD GLOBAL DATASET
# =========================
print("Loading friend's global parse_ok dataset...")
df = pd.read_pickle(INPUT_PKL)

required_cols = {"heuristic_id", "code", "objective"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in pkl: {missing}")

print(f"Total rows in global dataset: {len(df)}")
print("\n[Column check: unique values preview]")
for c in ["raw_app_type", "task_name", "instance_scale", "strategy"]:
    print_unique_preview(df, c)

# =========================
# CHOOSE GROUPING COLUMN
# =========================
# Prefer raw_app_type if it contains the raw task names;
# otherwise use task_name (normalized).
use_col = None

if "raw_app_type" in df.columns:
    raw_vals = set(df["raw_app_type"].dropna().astype(str).unique())
    if any(t in raw_vals for t in RAW_APP_TASKS):
        use_col = "raw_app_type"

if use_col is None and "task_name" in df.columns:
    task_vals = set(df["task_name"].dropna().astype(str).unique())
    normalized_targets = set(TASK_NAME_MAP.values())
    if any(t in task_vals for t in normalized_targets):
        use_col = "task_name"

if use_col is None:
    raise ValueError(
        "Cannot auto-detect task column.\n"
        "Expected 'raw_app_type' with values like bin_greedy/cvrp_lns/... or "
        "'task_name' with values like BinPacking/CVRP/Premarshalling/SlidingPuzzle."
    )

print(f"\nUsing grouping column: {use_col}")

# Build the 4 groups based on detected column
groups = {}
if use_col == "raw_app_type":
    for raw_task in RAW_APP_TASKS:
        groups[raw_task] = df[df["raw_app_type"].astype(str) == raw_task].copy()
else:
    # use task_name normalized
    for raw_task, norm_name in TASK_NAME_MAP.items():
        groups[raw_task] = df[df["task_name"].astype(str) == norm_name].copy()

# =========================
# RUN PER-TASK ANALYSIS
# =========================
print("\n================ PER-TASK TF-IDF + COSINE ================")

summary_rows = []

for raw_task, df_task in groups.items():
    print("\n----------------------------------------------------------")
    print(f"Task key (requested): {raw_task}")
    print(f"Rows found: {len(df_task)}")

    out_dir = os.path.join(OUTPUT_ROOT, raw_task)
    if len(df_task) == 0:
        print("  [WARN] No rows found for this task. Skipping.")
        summary_rows.append({
            "task": raw_task,
            "rows": 0,
            "status": "missing"
        })
        continue

    run_tfidf_cosine_metrics(df_task, out_dir)

    # load metrics back for summary
    with open(os.path.join(out_dir, "metrics.json"), "r", encoding="utf-8") as f:
        m = json.load(f)

    summary_rows.append({
        "task": raw_task,
        "rows": m["num_heuristics"],
        "pearson_r_distance": m["pearson_r_distance"],
        "mantel_r_distance": m["mantel_r_distance"],
        "overlap_at_{}".format(TOPK): m["overlap_at_{}".format(TOPK)],
        "tfidf_vocab_size": m["tfidf_vocab_size"],
        "status": "ok"
    })

# Save summary
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_ROOT, "summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\n================ DONE ================")
print(f"Saved per-task outputs under: {OUTPUT_ROOT}")
print(f"Summary saved to: {summary_path}")
