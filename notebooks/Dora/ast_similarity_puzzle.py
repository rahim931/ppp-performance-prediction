# -*- coding: utf-8 -*-
"""
AST Feature Vector (Structural, Compressed) analysis for SlidingPuzzle (puzzle_astar)

Inputs:
- E:\\Python\\HRT\\Project_4\\SlidingPuzzle.pkl   (preferred)
  (Optional) E:\\Python\\HRT\\Project_4\\SlidingPuzzle.csv (for manual inspection)

Outputs (in outputs/puzzle_ast/):
- features.parquet                  : AST feature vectors
- distance_struct.npy               : structural distance matrix (cosine)
- distance_perf.npy                 : performance distance matrix (abs diff)
- metrics.json                      : Pearson, Mantel, overlap@10, counts
"""

import os
import json
import ast
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr


# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = r"E:\Python\HRT\Project_4"

# Use your filtered dataset directly
PKL_PATH = os.path.join(PROJECT_ROOT, "SlidingPuzzle.pkl")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "puzzle_ast")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metrics config
OVERLAP_K = 10
MANTEL_PERMUTATIONS = 999  # increase for more stable p-value (e.g., 4999) if time allows
RANDOM_SEED = 42


# -----------------------------
# AST Feature Extractor (Structural, Compressed)
# -----------------------------
class ASTFeatureExtractor:
    """
    Structural (compressed) features:
      - node type counts (bag-of-nodes)
      - tree depth statistics
      - branching statistics
    """

    def _iter_child_nodes(self, node):
        for child in ast.iter_child_nodes(node):
            yield child

    def _walk_with_depth(self, node, depth=0):
        yield node, depth
        for child in self._iter_child_nodes(node):
            yield from self._walk_with_depth(child, depth + 1)

    def extract(self, code: str) -> Dict[str, float]:
        code = code if isinstance(code, str) else ""
        code = code.strip()

        if not code:
            return {"__parse_ok__": 0}

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"__parse_ok__": 0}
        except Exception:
            return {"__parse_ok__": 0}

        node_types = Counter()
        depths: List[int] = []
        branchings: List[int] = []

        for node, d in self._walk_with_depth(tree, 0):
            node_types[type(node).__name__] += 1
            depths.append(d)
            children = list(self._iter_child_nodes(node))
            branchings.append(len(children))

        feats: Dict[str, float] = {"__parse_ok__": 1}
        feats["num_nodes"] = float(sum(node_types.values()))
        feats["max_depth"] = float(max(depths)) if depths else 0.0
        feats["avg_depth"] = float(np.mean(depths)) if depths else 0.0
        feats["max_branching"] = float(max(branchings)) if branchings else 0.0
        feats["avg_branching"] = float(np.mean(branchings)) if branchings else 0.0

        for k, v in node_types.items():
            feats[f"node_{k}"] = float(v)

        return feats


def build_feature_matrix(feature_dicts: List[Dict[str, float]]) -> pd.DataFrame:
    feat_df = pd.DataFrame(feature_dicts).fillna(0)
    for c in feat_df.columns:
        feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce").fillna(0)
    return feat_df


def upper_triangle_values(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


# -----------------------------
# Mantel test (Pearson correlation between distance matrices + permutation p-value)
# -----------------------------
def mantel_test(D1: np.ndarray, D2: np.ndarray, permutations: int = 999, seed: int = 42) -> Tuple[float, float]:
    """
    Simple Mantel test:
      - statistic: Pearson correlation between upper triangles of D1 and D2
      - p-value: permutation test by permuting labels of D2
    """
    rng = np.random.default_rng(seed)

    v1 = upper_triangle_values(D1)
    v2 = upper_triangle_values(D2)

    r_obs, _ = pearsonr(v1, v2)

    # Permutations: shuffle labels of D2 (permute rows/cols together)
    more_extreme = 0
    n = D2.shape[0]
    for _ in range(permutations):
        perm = rng.permutation(n)
        D2p = D2[np.ix_(perm, perm)]
        v2p = upper_triangle_values(D2p)
        r_perm, _ = pearsonr(v1, v2p)
        if abs(r_perm) >= abs(r_obs):
            more_extreme += 1

    # Add 1 smoothing (common in permutation p-values)
    p_value = (more_extreme + 1) / (permutations + 1)
    return float(r_obs), float(p_value)


# -----------------------------
# Overlap@K
# -----------------------------
def overlap_at_k(D_struct: np.ndarray, D_perf: np.ndarray, k: int = 10) -> float:
    """
    For each item i:
      - structural neighbors: indices of smallest k distances in D_struct[i] (excluding itself)
      - performance neighbors: indices of smallest k distances in D_perf[i] (excluding itself)
      - overlap = |intersection| / k
    Average over i.
    """
    n = D_struct.shape[0]
    if n <= 1:
        return 0.0

    k_eff = min(k, n - 1)
    overlaps = []

    for i in range(n):
        # Exclude self by setting it to +inf
        ds = D_struct[i].copy()
        dp = D_perf[i].copy()
        ds[i] = np.inf
        dp[i] = np.inf

        nn_struct = np.argpartition(ds, k_eff)[:k_eff]
        nn_perf = np.argpartition(dp, k_eff)[:k_eff]

        inter = len(set(nn_struct.tolist()).intersection(set(nn_perf.tolist())))
        overlaps.append(inter / k_eff)

    return float(np.mean(overlaps))


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading filtered dataset:", PKL_PATH)
    if not os.path.exists(PKL_PATH):
        raise FileNotFoundError(f"Cannot find: {PKL_PATH}")

    df = pd.read_pickle(PKL_PATH)

    # --- Sanity checks ---
    print("Columns:", list(df.columns))
    print("Rows:", len(df))

    required = ["heuristic_id", "code", "objective"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Clean basics
    df = df.copy()
    df = df.dropna(subset=["code", "objective"])
    df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
    df = df.dropna(subset=["objective"]).reset_index(drop=True)

    print("After basic cleaning:", len(df))
    if len(df) == 0:
        raise ValueError("No rows left after basic cleaning. Check dataset content.")

    # --- Extract AST features ---
    extractor = ASTFeatureExtractor()
    feature_dicts = [extractor.extract(code) for code in df["code"].tolist()]
    feat_df = build_feature_matrix(feature_dicts)

    # Filter parse failures
    parse_ok_mask = feat_df["__parse_ok__"] == 1
    df_ok = df[parse_ok_mask.values].reset_index(drop=True)
    feat_ok = feat_df[parse_ok_mask.values].reset_index(drop=True)

    n_total = int(len(df))
    n_parse_ok = int(len(df_ok))
    n_parse_fail = n_total - n_parse_ok

    print(f"Parse OK: {n_parse_ok} / {n_total} (fail: {n_parse_fail})")
    if n_parse_ok < 5:
        raise ValueError("Too few parseable codes. AST parsing may be failing due to non-Python code.")

    # Save feature vectors
    out_feat_path = os.path.join(OUTPUT_DIR, "features.parquet")
    feat_out = feat_ok.copy()
    feat_out.insert(0, "heuristic_id", df_ok["heuristic_id"].values)
    feat_out.to_parquet(out_feat_path, index=False)
    print("Saved features:", out_feat_path)

    # --- Distance matrices ---
    X = feat_ok.drop(columns=["__parse_ok__"], errors="ignore").values.astype(float)
    D_struct = squareform(pdist(X, metric="cosine"))

    y = df_ok["objective"].values.astype(float)
    D_perf = np.abs(y.reshape(-1, 1) - y.reshape(1, -1))

    # Save distance matrices
    np.save(os.path.join(OUTPUT_DIR, "distance_struct.npy"), D_struct)
    np.save(os.path.join(OUTPUT_DIR, "distance_perf.npy"), D_perf)

    # --- Pearson (upper triangle) ---
    v1 = upper_triangle_values(D_struct)
    v2 = upper_triangle_values(D_perf)
    pear_r, pear_p = pearsonr(v1, v2)

    # --- Mantel ---
    mantel_r, mantel_p = mantel_test(D_struct, D_perf, permutations=MANTEL_PERMUTATIONS, seed=RANDOM_SEED)

    # --- overlap@10 ---
    ov10 = overlap_at_k(D_struct, D_perf, k=OVERLAP_K)

    metrics = {
        "task": "SlidingPuzzle",
        "n_total_rows_in_file": n_total,
        "n_parse_ok": n_parse_ok,
        "n_parse_fail": n_parse_fail,
        "struct_distance": "cosine(feature_vector)",
        "perf_distance": "abs(objective_i - objective_j)",
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "mantel_r": float(mantel_r),
        "mantel_p": float(mantel_p),
        "overlap_at_10": float(ov10),
        "mantel_permutations": int(MANTEL_PERMUTATIONS),
        "random_seed": int(RANDOM_SEED),
    }

    out_metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics:", out_metrics_path)
    print("Pearson r:", pear_r, "p:", pear_p)
    print("Mantel  r:", mantel_r, "p:", mantel_p)
    print("Overlap@10:", ov10)


if __name__ == "__main__":
    main()
