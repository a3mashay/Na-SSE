#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust statistical outlier detection for sodium solid-state electrolytes.

See README.md for the method. Run with --help for inputs and parameters.
"""

from __future__ import annotations

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Configuration as reported in the manuscript
# --------------------------------------------------------------------------
RANDOM_STATE = 17          # seed shared with the UMAP embedding
PCA_KEEP_VAR = 0.95        # retain 95% of descriptor variance (80 PCs in the paper run)
MIN_PC_VAR = 1e-10         # drop principal components with numerically null variance
SUPPORT_FRACTION = 0.75    # MCD subset size; the sklearn default (~0.51) is rank-deficient
EIG_FLOOR = 1e-8           # discard eigen-directions below EIG_FLOOR * lambda_max
N_REPEATS = 10             # independent random states for the global screen
CONSENSUS_FRAC = 0.8       # report compositions flagged in >= 80% of repeats
LOF_NEIGHBORS = 35         # k for the Local Outlier Factor
Q_GLOBAL = 0.99            # top 1% by robust Mahalanobis^2
Q_LOF = 0.99               # top 1% by LOF score
UMAP_LABELS = ("z1", "z2")  # UMAP coordinate columns carried in the summary CSV

# Columns treated as metadata and excluded from the descriptor matrix
META_PATTERNS = [
    r"^formula_pretty($|_)",
    r"^family($|_)",
    r"^band_gap($|_)",
    r"^Ehull_meV_atom($|_)",
    r"^Na_ratio($|_)",
    r"^cluster($|_)",
    r"^membership_prob($|_)",
    r"^cluster_promising_share($|_)",
    r"^promising_flag($|_)",
    r"^promising_family($|_)",
    r"^composition($|_)",
    r"^composition_obj($|_)",
    rf"^{UMAP_LABELS[0]}($|_)",
    rf"^{UMAP_LABELS[1]}($|_)",
]


def is_meta_col(name: str) -> bool:
    """True if a column is pipeline metadata rather than a chemical descriptor."""
    return any(re.match(p, name, flags=re.IGNORECASE) for p in META_PATTERNS)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_and_merge(features_csv: str, summary_csv: str) -> pd.DataFrame:
    """Merge the descriptor table with the per-compound screening summary.

    The summary may contain several rows per reduced formula; the row with the
    highest HDBSCAN membership probability is kept so that the merge is 1:1.
    """
    feat = pd.read_csv(features_csv, low_memory=False)
    summ = pd.read_csv(summary_csv, low_memory=False)

    for name, d in (("features", feat), ("summary", summ)):
        if "formula_pretty" not in d.columns:
            raise ValueError(f"{name} table is missing the 'formula_pretty' column")

    if "membership_prob" in summ.columns:
        summ = (
            summ.sort_values(["formula_pretty", "membership_prob"], ascending=[True, False])
            .drop_duplicates(subset=["formula_pretty"], keep="first")
            .reset_index(drop=True)
        )

    meta_candidates = [
        "formula_pretty", "family", "band_gap", "Ehull_meV_atom", "Na_ratio",
        "cluster", "membership_prob", UMAP_LABELS[0], UMAP_LABELS[1],
    ]
    keep_meta = list(dict.fromkeys(c for c in meta_candidates if c in summ.columns))

    # Drop metadata already present in the feature table so the merge does not
    # produce _x/_y suffix pairs.
    drop_from_feat = [c for c in keep_meta if c in feat.columns and c != "formula_pretty"]
    feat_clean = feat.drop(columns=drop_from_feat, errors="ignore")

    df = feat_clean.merge(summ[keep_meta], on="formula_pretty", how="inner")
    print(f"Merged rows: {len(df)} | columns: {df.shape[1]}")
    return df


def build_descriptor_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric descriptor matrix: metadata removed, NaNs median-filled, constants dropped."""
    candidate_cols = [c for c in df.columns if not is_meta_col(c)]
    num_cols = [c for c in candidate_cols if is_numeric_dtype(df[c])]
    X = df[num_cols].copy()

    X = X.drop(columns=X.columns[X.isna().all()])
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.drop(columns=X.columns[X.isna().all()])

    X = X.fillna(X.median(numeric_only=True))

    keep = X.var(axis=0).to_numpy() > 0
    X = X.loc[:, keep]
    print(f"Descriptors kept after removing all-NaN and constant columns: {X.shape[1]}")
    return X


# --------------------------------------------------------------------------
# Global score: robust Mahalanobis in PCA space
# --------------------------------------------------------------------------
def robust_cov_mahal(X_cov: np.ndarray,
                     random_state: int = RANDOM_STATE,
                     support_fraction: float = SUPPORT_FRACTION,
                     eig_floor: float = EIG_FLOOR,
                     verbose: bool = True):
    """Squared Mahalanobis distance under a well-conditioned robust covariance.

    Do not revert to a bare MinCovDet fit with a pseudo-inverse: the default
    support fraction is rank-deficient at ~80 PCs, 1e-17 eigenvalues become
    1e15 distances, and the flagged set stops being reproducible between runs.
    """
    mcd = MinCovDet(random_state=random_state,
                    support_fraction=support_fraction,
                    assume_centered=False).fit(X_cov)
    mean = mcd.location_
    inliers = X_cov[mcd.support_] - mean
    cov = LedoitWolf(assume_centered=True).fit(inliers).covariance_

    w, V = np.linalg.eigh(cov)
    keep = w > eig_floor * w.max()
    inv_cov = (V[:, keep] / w[keep]) @ V[:, keep].T

    diff = X_cov - mean
    mahal2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)

    if verbose:
        print(f"  support {int(mcd.support_.sum())}/{len(X_cov)} | "
              f"cond {w.max() / w.min():.4g} | "
              f"directions {int(keep.sum())}/{len(w)} | "
              f"max mahal^2 {mahal2.max():.4g}")
    if w.min() <= 0 or w.max() / w.min() > 1e8:
        warnings.warn(
            "Covariance is still poorly conditioned. Raise --support-fraction or "
            "lower --pca-var before trusting the ranking."
        )
    return mahal2, cov, mean


def repeat_global_screen(X_pc: np.ndarray, n_repeats: int, quantile: float):
    """Run the global screen `n_repeats` times and return per-compound statistics.

    Returns
    -------
    m2_all : (n_repeats, n_samples) array of squared Mahalanobis distances
    sel    : (n_samples,) count of repeats in which the row was flagged
    """
    n = len(X_pc)
    k_flag = int(round((1.0 - quantile) * n))
    m2_all = np.zeros((n_repeats, n))
    sel = np.zeros(n, dtype=int)

    for r in range(n_repeats):
        print(f"  repeat {r + 1}/{n_repeats} (random_state={r})")
        m2, _, _ = robust_cov_mahal(X_pc, random_state=r)
        m2_all[r] = m2
        sel[np.argsort(-m2)[:k_flag]] += 1

    return m2_all, sel, k_flag


def jaccard_matrix(m2_all: np.ndarray, k_flag: int) -> np.ndarray:
    """Pairwise Jaccard overlap between the flagged sets of each repeat."""
    sets = [set(np.argsort(-m2)[:k_flag]) for m2 in m2_all]
    n = len(sets)
    J = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
    return J


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--features", default="composition_features.csv",
                    help="Matminer composition descriptor table")
    ap.add_argument("--summary", default="summary_screened_candidates.csv",
                    help="Per-compound screening summary with cluster labels and UMAP coordinates")
    ap.add_argument("--outdir", default="analysis_ext", help="Output directory")
    ap.add_argument("--pca-var", type=float, default=PCA_KEEP_VAR,
                    help="Fraction of descriptor variance retained by PCA")
    ap.add_argument("--support-fraction", type=float, default=SUPPORT_FRACTION,
                    help="MCD support fraction (the sklearn default is rank-deficient here)")
    ap.add_argument("--repeats", type=int, default=N_REPEATS,
                    help="Independent random states for the global screen")
    ap.add_argument("--consensus", type=float, default=CONSENSUS_FRAC,
                    help="Selection frequency required to report a global outlier")
    ap.add_argument("--lof-neighbors", type=int, default=LOF_NEIGHBORS,
                    help="k for the Local Outlier Factor")
    ap.add_argument("--quantile", type=float, default=Q_GLOBAL,
                    help="Flag quantile applied to both scores (0.99 = top 1%%)")
    ap.add_argument("--top-n", type=int, default=40,
                    help="Rows written to the top-global and top-local tables")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_and_merge(args.features, args.summary)

    if not set(UMAP_LABELS).issubset(df.columns):
        raise KeyError(
            f"UMAP coordinates {UMAP_LABELS} are absent. The local score is computed in the "
            "same 2D embedding used for clustering, so these columns must be carried over "
            "from the screening summary."
        )

    X = build_descriptor_matrix(df)

    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=args.pca_var, random_state=RANDOM_STATE)
    X_pc = pca.fit_transform(X_std)

    keep_pc = pca.explained_variance_ > MIN_PC_VAR * pca.explained_variance_[0]
    X_pc = X_pc[:, keep_pc]
    print(f"PCA retained {X_pc.shape[1]} components "
          f"({pca.explained_variance_ratio_[keep_pc].sum():.4f} of the variance)")

    # ---- global score, repeated ----
    print(f"\nGlobal screen over {args.repeats} repeats:")
    m2_all, sel, k_flag = repeat_global_screen(X_pc, args.repeats, args.quantile)

    stats = pd.DataFrame({
        "mahal2": m2_all.mean(axis=0),
        "mahal2_sd": m2_all.std(axis=0, ddof=1) if args.repeats > 1 else 0.0,
        "mahal2_mean_rank": (-m2_all).argsort(axis=1).argsort(axis=1).mean(axis=0) + 1,
        "selection_freq": sel / args.repeats,
    }, index=df.index)
    stats["outlier_mahal_flag"] = stats["selection_freq"] >= args.consensus
    df = pd.concat([df, stats], axis=1)

    # ---- local score, computed once ----
    X_umap = df[list(UMAP_LABELS)].to_numpy(dtype=float)
    lof = LocalOutlierFactor(n_neighbors=args.lof_neighbors, metric="euclidean", n_jobs=-1)
    lof.fit_predict(X_umap)
    # sklearn returns the negated factor; flip the sign so larger means more outlying
    lof_score = -lof.negative_outlier_factor_
    df = pd.concat([df, pd.DataFrame({
        "lof_score": lof_score,
        "outlier_lof_flag": lof_score >= np.quantile(lof_score, args.quantile),
    }, index=df.index)], axis=1)

    main_path = os.path.join(args.outdir, "outliers_pca_robustcov_summary.csv")
    df.to_csv(main_path, index=False)

    report_cols = [c for c in [
        "formula_pretty", "family", "cluster", "membership_prob",
        "Ehull_meV_atom", "band_gap", "Na_ratio",
        "mahal2", "mahal2_sd", "mahal2_mean_rank", "selection_freq",
        "lof_score", "outlier_mahal_flag", "outlier_lof_flag",
    ] if c in df.columns]

    top_global = (
        df.sort_values(["selection_freq", "mahal2"], ascending=False)
        .head(args.top_n)[report_cols]
    )
    top_local = df.sort_values("lof_score", ascending=False).head(args.top_n)[report_cols]
    top_global.to_csv(os.path.join(args.outdir, "outliers_top_global.csv"), index=False)
    top_local.to_csv(os.path.join(args.outdir, "outliers_top_local.csv"), index=False)

    for key, fname in (("family", "outliers_by_family.csv"),
                       ("cluster", "outliers_by_cluster.csv")):
        if key not in df.columns:
            continue
        tab = (
            df.groupby(key)
            .agg(n=("formula_pretty", "count"),
                 n_global=("outlier_mahal_flag", "sum"),
                 n_local=("outlier_lof_flag", "sum"))
            .reset_index()
        )
        tab["global_share"] = tab["n_global"] / tab["n"]
        tab["local_share"] = tab["n_local"] / tab["n"]
        tab.to_csv(os.path.join(args.outdir, fname), index=False)

    # ---- stability diagnostics ----
    if args.repeats > 1:
        J = jaccard_matrix(m2_all, k_flag)
        off = J[np.triu_indices(args.repeats, k=1)]
        pd.DataFrame(J,
                     index=[f"repeat_{i}" for i in range(args.repeats)],
                     columns=[f"repeat_{i}" for i in range(args.repeats)]
                     ).to_csv(os.path.join(args.outdir, "outlier_repeat_stability.csv"))
    else:
        off = np.array([1.0])

    n_global = int(df["outlier_mahal_flag"].sum())
    n_local = int(df["outlier_lof_flag"].sum())
    overlap = df.loc[df["outlier_mahal_flag"] & df["outlier_lof_flag"], "formula_pretty"].tolist()

    print("\n=== Summary ===")
    print(f"Rows: {len(df)} | principal components: {X_pc.shape[1]}")
    print(f"Each repeat flags the top {(1 - args.quantile) * 100:.0f}%: {k_flag} compositions")
    print(f"Pairwise Jaccard between repeats: mean {off.mean():.3f}, min {off.min():.3f}")
    print(f"Global outliers (selection frequency >= {args.consensus:.0%}): {n_global}")
    print(f"Local outliers  (LOF, top {(1 - args.quantile) * 100:.0f}%):      {n_local}")
    print(f"Flagged by both: {len(overlap)}" + (f" -> {', '.join(overlap)}" if overlap else ""))
    print("Saved:", main_path)


if __name__ == "__main__":
    main()
