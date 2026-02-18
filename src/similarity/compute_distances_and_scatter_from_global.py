# -*- coding: utf-8 -*-
"""
From global parse_ok dataset (all_heuristics_dataset.pkl), compute:

Per task:
  - distance_perf.npy  : |objective_i - objective_j|
Per method (TF-IDF / AST / Jaccard):
  - distance_struct.npy: code/structure distance matrix (method-specific)
  - scatter plot (sampled pairs): x=distance_struct, y=distance_perf
    + regression line + Pearson r in title

Outputs:
  outputs/recompute_from_global/<task>/<method>/
      distance_perf.npy
      distance_struct.npy   (if computed full matrix)
      scatter_<method>_<task>.png
      pair_samples.csv      (optional; sampled pairs used for scatter)

Notes:
  - Full NxN Jaccard is expensive; default uses sampling for scatter only.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import ast


# =========================
# PATHS
# =========================
GLOBAL_PKL = r"E:\Python\HRT\Project_4\all_heuristics_dataset.pkl"
OUT_ROOT = r"E:\Python\HRT\Project_4\outputs\recompute_from_global"

os.makedirs(OUT_ROOT, exist_ok=True)


# =========================
# SETTINGS
# =========================
TASK_COL = "raw_app_type"   # bin_greedy, cvrp_lns, premarshalling_astar, puzzle_astar
CODE_COL = "code"
OBJ_COL  = "objective"
ID_COL   = "heuristic_id"

TASK_KEYS = ["bin_greedy", "cvrp_lns", "premarshalling_astar", "puzzle_astar"]

RANDOM_SEED = 42

# full N×N matrices can be large; still ok for ~5k with float32
DTYPE = np.float32

# Scatter sampling (recommended)
# number of random pairs sampled for scatter
SCATTER_NUM_PAIRS = 20000   # adjust if too slow/too dense

# If you really want full Jaccard distance matrix (NOT recommended for ~5k)
JACCARD_FULL_MATRIX = False
JACCARD_MAX_FULL_N = 1500     # safety guard

# For heatmap / scatter, sometimes you want similarity not distance;
# here we follow your rule: x=distance_struct, y=distance_perf.
# (distance small => more similar)
USE_DISTANCE_ON_X = True


# =========================
# Utilities
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def ast_node_counts(code: str):
    """Compressed AST feature: count node types."""
    try:
        tree = ast.parse(code)
    except Exception:
        return {}
    counts = {}
    for node in ast.walk(tree):
        t = type(node).__name__
        counts[t] = counts.get(t, 0) + 1
    return counts

def compute_perf_distance(objectives: np.ndarray) -> np.ndarray:
    """Full NxN |obj_i - obj_j| in float32."""
    obj = objectives.astype(np.float32)
    # broadcasting
    D = np.abs(obj[:, None] - obj[None, :]).astype(DTYPE)
    np.fill_diagonal(D, 0.0)
    return D

def linear_regression_line(x, y):
    """Return slope/intercept for y = a*x + b (simple least squares)."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    xm = x.mean()
    ym = y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0, ym
    a = (((x - xm) * (y - ym)).sum()) / denom
    b = ym - a * xm
    return a, b

def sample_pairs(n: int, num_pairs: int, seed: int):
    """Sample (i,j) pairs with i<j."""
    rng = np.random.default_rng(seed)
    # sample indices uniformly
    i = rng.integers(0, n, size=num_pairs, dtype=np.int64)
    j = rng.integers(0, n, size=num_pairs, dtype=np.int64)
    mask = i != j
    i = i[mask]
    j = j[mask]
    # enforce ordering
    ii = np.minimum(i, j)
    jj = np.maximum(i, j)
    # remove self-pairs
    mask2 = ii != jj
    ii = ii[mask2]
    jj = jj[mask2]
    return ii, jj

def plot_scatter_and_save(x, y, title, save_path):
    """Scatter + regression line + Pearson r in title."""
    # Pearson r
    r = float(pearsonr(x, y)[0]) if len(x) > 2 else float("nan")
    a, b = linear_regression_line(x, y)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=6)  # do NOT set colors to comply with your style preference
    # regression line
    xs = np.linspace(x.min(), x.max(), 200)
    ys = a * xs + b
    plt.plot(xs, ys, linewidth=2)

    plt.title(f"{title}\nPearson r = {r:.4f}")
    plt.xlabel("Structural distance (method)")
    plt.ylabel("Performance distance |obj_i - obj_j|")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return r

def compute_struct_distance_tfidf(codes: list[str]) -> np.ndarray:
    """TF-IDF cosine distance: D = 1 - cosine_similarity(tfidf)."""
    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        max_features=None
    )
    X = vec.fit_transform(codes)  # sparse
    S = cosine_similarity(X)      # dense
    D = (1.0 - S).astype(DTYPE)
    np.fill_diagonal(D, 0.0)
    return D

def compute_struct_distance_ast(codes: list[str]) -> np.ndarray:
    """AST node-type cosine distance."""
    feat_dicts = [ast_node_counts(c) for c in codes]
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(feat_dicts)
    S = cosine_similarity(X)
    D = (1.0 - S).astype(DTYPE)
    np.fill_diagonal(D, 0.0)
    return D

def jaccard_similarity_set(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0

def compute_struct_distance_jaccard_full(codes: list[str]) -> np.ndarray:
    """
    WARNING: O(N^2). Only use for small N.
    Token Jaccard distance: 1 - |A∩B|/|A∪B|
    """
    tokens = [set(c.split()) for c in codes]  # simple whitespace tokenization
    n = len(tokens)
    D = np.zeros((n, n), dtype=DTYPE)
    for i in range(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            sim = jaccard_similarity_set(tokens[i], tokens[j])
            dist = 1.0 - sim
            D[i, j] = dist
            D[j, i] = dist
    return D

def scatter_from_sampling_perf_and_method(df_task: pd.DataFrame, method: str, out_dir: str,
                                          D_perf: np.ndarray | None,
                                          D_struct: np.ndarray | None):
    """
    Produce scatter using sampled pairs.
    If full matrices are available, use them directly.
    Otherwise compute x on-the-fly for the sampled pairs (e.g., Jaccard).
    """
    ensure_dir(out_dir)
    n = len(df_task)
    num_pairs = min(SCATTER_NUM_PAIRS, n * (n - 1) // 2)
    ii, jj = sample_pairs(n, num_pairs, seed=RANDOM_SEED)

    objectives = df_task[OBJ_COL].to_numpy(dtype=np.float32)

    # y: performance distance
    if D_perf is not None:
        y = D_perf[ii, jj]
    else:
        y = np.abs(objectives[ii] - objectives[jj]).astype(np.float32)

    # x: structural distance
    codes = df_task[CODE_COL].astype(str).tolist()
    if D_struct is not None:
        x = D_struct[ii, jj]
    else:
        # compute per pair (used for Jaccard if not full)
        if method.lower() == "jaccard":
            token_sets = [set(c.split()) for c in codes]
            x_list = []
            for a, b in zip(ii, jj):
                sim = jaccard_similarity_set(token_sets[int(a)], token_sets[int(b)])
                x_list.append(1.0 - sim)
            x = np.array(x_list, dtype=np.float32)
        else:
            raise ValueError("D_struct is None but method is not supported for pairwise on-the-fly compute.")

    # Save sampled pairs for audit
    pair_csv = os.path.join(out_dir, f"pair_samples_{method}.csv")
    pd.DataFrame({
        "i": ii,
        "j": jj,
        "heuristic_i": df_task[ID_COL].iloc[ii].to_numpy(),
        "heuristic_j": df_task[ID_COL].iloc[jj].to_numpy(),
        "distance_struct": x,
        "distance_perf": y
    }).to_csv(pair_csv, index=False, encoding="utf-8-sig")

    fig_path = os.path.join(out_dir, f"scatter_{method}.png")
    r = plot_scatter_and_save(
        x=x,
        y=y,
        title=f"{df_task[TASK_COL].iloc[0]} | {method} | sampled pairs={len(x)}",
        save_path=fig_path
    )
    return r, fig_path, pair_csv


# =========================
# MAIN
# =========================
def main():
    print("Loading global dataset:", GLOBAL_PKL)
    df = pd.read_pickle(GLOBAL_PKL)

    # Basic checks & enforce parse_ok if exists
    needed = [ID_COL, TASK_COL, CODE_COL, OBJ_COL]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str)
    df[CODE_COL] = df[CODE_COL].astype(str)
    df[OBJ_COL] = pd.to_numeric(df[OBJ_COL], errors="coerce")
    df = df.dropna(subset=[OBJ_COL])
    df = df[df[CODE_COL].str.strip() != ""]

    if "parse_ok" in df.columns:
        df = df[df["parse_ok"] == True].copy()
        print(f"Using parse_ok==True rows: {len(df)}")

    summary_rows = []

    for task in TASK_KEYS:
        df_task = df[df[TASK_COL] == task].copy()
        if df_task.empty:
            print(f"[SKIP] {task}: no rows")
            continue

        df_task = df_task.drop_duplicates(subset=[ID_COL], keep="first").reset_index(drop=True)
        n = len(df_task)
        print("\n" + "=" * 80)
        print(f"Task: {task} | N={n}")

        task_root = os.path.join(OUT_ROOT, task)
        ensure_dir(task_root)

        # ---- performance distance (shared across methods) ----
        perf_path = os.path.join(task_root, "distance_perf.npy")
        if os.path.exists(perf_path):
            print("Loading existing distance_perf.npy")
            D_perf = np.load(perf_path)
        else:
            print("Computing distance_perf.npy (full NxN)...")
            objectives = df_task[OBJ_COL].to_numpy(dtype=np.float32)
            D_perf = compute_perf_distance(objectives)
            np.save(perf_path, D_perf)
            print("Saved:", perf_path)

        # ---- METHOD 1: TF-IDF ----
        tfidf_dir = os.path.join(task_root, "tfidf")
        ensure_dir(tfidf_dir)
        tfidf_struct_path = os.path.join(tfidf_dir, "distance_struct.npy")

        if os.path.exists(tfidf_struct_path):
            print("TF-IDF: loading existing distance_struct.npy")
            D_tfidf = np.load(tfidf_struct_path)
        else:
            print("TF-IDF: computing full distance_struct.npy ...")
            codes = df_task[CODE_COL].tolist()
            D_tfidf = compute_struct_distance_tfidf(codes)
            np.save(tfidf_struct_path, D_tfidf)
            print("Saved:", tfidf_struct_path)

        # copy perf into method folder for “stored together” requirement
        np.save(os.path.join(tfidf_dir, "distance_perf.npy"), D_perf)

        r_tfidf, fig_tfidf, pairs_tfidf = scatter_from_sampling_perf_and_method(
            df_task=df_task, method="tfidf",
            out_dir=tfidf_dir, D_perf=D_perf, D_struct=D_tfidf
        )

        # ---- METHOD 2: AST ----
        ast_dir = os.path.join(task_root, "ast")
        ensure_dir(ast_dir)
        ast_struct_path = os.path.join(ast_dir, "distance_struct.npy")

        if os.path.exists(ast_struct_path):
            print("AST: loading existing distance_struct.npy")
            D_ast = np.load(ast_struct_path)
        else:
            print("AST: computing full distance_struct.npy ...")
            codes = df_task[CODE_COL].tolist()
            D_ast = compute_struct_distance_ast(codes)
            np.save(ast_struct_path, D_ast)
            print("Saved:", ast_struct_path)

        np.save(os.path.join(ast_dir, "distance_perf.npy"), D_perf)

        r_ast, fig_ast, pairs_ast = scatter_from_sampling_perf_and_method(
            df_task=df_task, method="ast",
            out_dir=ast_dir, D_perf=D_perf, D_struct=D_ast
        )

        # ---- METHOD 3: Jaccard ----
        jacc_dir = os.path.join(task_root, "jaccard")
        ensure_dir(jacc_dir)
        np.save(os.path.join(jacc_dir, "distance_perf.npy"), D_perf)

        jacc_struct_path = os.path.join(jacc_dir, "distance_struct.npy")
        D_jacc = None

        if JACCARD_FULL_MATRIX and n <= JACCARD_MAX_FULL_N:
            if os.path.exists(jacc_struct_path):
                print("Jaccard: loading existing distance_struct.npy")
                D_jacc = np.load(jacc_struct_path)
            else:
                print("Jaccard: computing FULL NxN distance_struct.npy (slow) ...")
                codes = df_task[CODE_COL].tolist()
                D_jacc = compute_struct_distance_jaccard_full(codes)
                np.save(jacc_struct_path, D_jacc)
                print("Saved:", jacc_struct_path)
        else:
            print("Jaccard: skipping full NxN matrix (sampling scatter only).")

        r_jacc, fig_jacc, pairs_jacc = scatter_from_sampling_perf_and_method(
            df_task=df_task, method="jaccard",
            out_dir=jacc_dir, D_perf=D_perf, D_struct=D_jacc
        )

        summary_rows.extend([
            {"task": task, "method": "tfidf",   "pearson_r_sampled": r_tfidf, "scatter_png": fig_tfidf},
            {"task": task, "method": "ast",     "pearson_r_sampled": r_ast,   "scatter_png": fig_ast},
            {"task": task, "method": "jaccard", "pearson_r_sampled": r_jacc,  "scatter_png": fig_jacc},
        ])

    summary_csv = os.path.join(OUT_ROOT, "summary_scatter_sampled.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print("\nDONE. Summary saved to:", summary_csv)
    print("All outputs under:", OUT_ROOT)


if __name__ == "__main__":
    main()
