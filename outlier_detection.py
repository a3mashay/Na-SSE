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

RANDOM_STATE = 17
PCA_KEEP_VAR = 0.95
LOF_NEIGHBORS = 35
Q_GLOBAL = 0.99
Q_LOF = 0.99
UMAP_LABELS = ("z1", "z2")

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
    return any(re.match(p, name, flags=re.IGNORECASE) for p in META_PATTERNS)

def load_and_merge(features_csv: str, summary_csv: str) -> pd.DataFrame:
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

    drop_from_feat = [c for c in keep_meta if c in feat.columns and c != "formula_pretty"]
    feat_clean = feat.drop(columns=drop_from_feat, errors="ignore")

    df = feat_clean.merge(summ[keep_meta], on="formula_pretty", how="inner")
    print(f"Merged rows: {len(df)} | columns: {df.shape[1]}")
    return df

def build_descriptor_matrix(df: pd.DataFrame) -> pd.DataFrame:
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

def robust_cov_mahal(X_cov: np.ndarray, random_state: int = RANDOM_STATE):
    try:
        mcd = MinCovDet(random_state=random_state, assume_centered=False).fit(X_cov)
        cov, mean = mcd.covariance_, mcd.location_
    except Exception as exc:
        warnings.warn(f"MinCovDet failed ({exc}); falling back to Ledoit-Wolf.")
        lw = LedoitWolf().fit(X_cov)
        cov, mean = lw.covariance_, X_cov.mean(axis=0)

    inv_cov = np.linalg.pinv(cov)
    diff = X_cov - mean
    mahal2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return mahal2, cov, mean

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Robust statistical outlier detection for sodium solid-state electrolytes."
    )
    ap.add_argument("--features", default="composition_features.csv",
                    help="Matminer composition descriptor table")
    ap.add_argument("--summary", default="summary_screened_candidates.csv",
                    help="Per-compound screening summary with cluster labels and UMAP coordinates")
    ap.add_argument("--outdir", default="analysis_ext", help="Output directory")
    ap.add_argument("--pca-var", type=float, default=PCA_KEEP_VAR,
                    help="Fraction of descriptor variance retained by PCA")
    ap.add_argument("--lof-neighbors", type=int, default=LOF_NEIGHBORS,
                    help="k for the Local Outlier Factor")
    ap.add_argument("--quantile", type=float, default=Q_GLOBAL,
                    help="Flag quantile applied to both scores (0.99 = top 1%%)")
    ap.add_argument("--top-n", type=int, default=40,
                    help="Rows written to the top-global and top-local tables")
    args = ap.parse_args()

    warnings.filterwarnings(
        "ignore", message="The covariance matrix associated to your dataset is not full rank"
    )
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
    print(f"PCA retained {X_pc.shape[1]} components "
          f"({pca.explained_variance_ratio_.sum():.4f} of the variance)")

    print("Fitting MinCovDet for the robust Mahalanobis distance...")
    mahal2, _, _ = robust_cov_mahal(X_pc, random_state=RANDOM_STATE)
    df["mahal2"] = mahal2

    X_umap = df[list(UMAP_LABELS)].to_numpy(dtype=float)
    lof = LocalOutlierFactor(n_neighbors=args.lof_neighbors, metric="euclidean", n_jobs=-1)
    lof.fit_predict(X_umap)
    df["lof_score"] = -lof.negative_outlier_factor_

    thr_mahal = np.quantile(df["mahal2"], args.quantile)
    thr_lof = np.quantile(df["lof_score"], args.quantile)
    df["outlier_mahal_flag"] = df["mahal2"] >= thr_mahal
    df["outlier_lof_flag"] = df["lof_score"] >= thr_lof

    main_path = os.path.join(args.outdir, "outliers_pca_robustcov_summary.csv")
    df.to_csv(main_path, index=False)

    report_cols = [c for c in [
        "formula_pretty", "family", "cluster", "membership_prob",
        "Ehull_meV_atom", "band_gap", "Na_ratio", "mahal2", "lof_score",
        "outlier_mahal_flag", "outlier_lof_flag",
    ] if c in df.columns]

    top_global = df.sort_values("mahal2", ascending=False).head(args.top_n)[report_cols]
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

    n_global = int(df["outlier_mahal_flag"].sum())
    n_local = int(df["outlier_lof_flag"].sum())
    overlap = df.loc[df["outlier_mahal_flag"] & df["outlier_lof_flag"], "formula_pretty"].tolist()

    print("\n=== Summary ===")
    print(f"Rows: {len(df)} | principal components: {X_pc.shape[1]}")
    print(f"Global outliers (Mahalanobis, top {(1 - args.quantile) * 100:.0f}%): {n_global}")
    print(f"Local outliers  (LOF,         top {(1 - args.quantile) * 100:.0f}%): {n_local}")
    print(f"Flagged by both: {len(overlap)}" + (f" -> {', '.join(overlap)}" if overlap else ""))
    print("Saved:", main_path)

if __name__ == "__main__":
    main()
