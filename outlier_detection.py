"""
Na-SSE Outlier Detection
========================
Robust statistical outlier detection for sodium solid-state electrolyte (Na-SSE)
candidates, as used in the manuscript.

Method:
  * Standardize composition descriptors and reduce with PCA (95% variance retained).
  * Global outliers: robust Mahalanobis^2 distance using a Minimum Covariance
    Determinant (MinCovDet) estimator, with a Ledoit-Wolf fallback.
  * Local outliers: Local Outlier Factor (LOF) computed in the same PCA space.
  * Compounds in the top 1% of either score are flagged; per-material "why"
    reports are written for the most extreme global outliers.

Inputs (CSV):
  --features  composition feature table (one row per material; must contain
              'formula_pretty' and numeric descriptor columns)
  --summary   screened summary table (provides family / cluster / Ehull / band_gap)

Outputs (written to --outdir):
  outliers_pca_robustcov_summary.csv, per-family and per-cluster counts,
  per-material WHY reports, and diagnostic plots.

Usage:
  python outlier_detection.py --features data/composition_features.csv \
                              --summary  data/summary_screened_candidates.csv \
                              --outdir   outputs/outlier_analysis

If you use this code, please cite the associated manuscript.
"""

# ============================================
# Na-SSE Outlier Analysis (better math version)
# - PCA (variance-preserving) before covariance
# - Robust covariance (MinCovDet) with Ledoit–Wolf fallback
# - Mahalanobis^2 with pinv (stable even if not full rank)
# - LOF computed in PCA space (less noisy)
# - Duplicate-safe merge of features + summary
# - Per-material WHY reports (global + local)
# ============================================

import os, warnings, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet, LedoitWolf
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import spearmanr

# ----------------- Silence noisy warnings -----------------
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jupyter_client.session")
warnings.filterwarnings("ignore", message="The covariance matrix associated to your dataset is not full rank")

# ----------------- Paths (configurable) -----------------
import argparse
_ap = argparse.ArgumentParser(description="Na-SSE outlier detection (robust Mahalanobis + LOF in PCA space)")
_ap.add_argument("--features", default="data/composition_features.csv",
                 help="composition feature table (CSV)")
_ap.add_argument("--summary",  default="data/summary_screened_candidates.csv",
                 help="screened summary table (CSV)")
_ap.add_argument("--outdir",   default="outputs/outlier_analysis",
                 help="output directory")
_args, _ = _ap.parse_known_args()

FEATURES_CSV = _args.features
SUMMARY_CSV  = _args.summary
OUT_DIR      = _args.outdir
WHY_DIR      = os.path.join(OUT_DIR, "outlier_explanations")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(WHY_DIR, exist_ok=True)

# ----------------- Config -----------------
RANDOM_STATE    = 17
PCA_KEEP_VAR    = 0.95   # keep 95% variance (set to an int for fixed # of comps)
LOF_NEIGHBORS   = 35
Q_GLOBAL        = 0.99   # top 1% by Mahalanobis^2
Q_LOF           = 0.99   # top 1% by LOF score
TOPK_WHY        = 25     # create WHY reports for top-K global outliers
N_TOP_FEATURES  = 12     # top features to print in WHY
UMAP_LABELS     = ("z1", "z2")  # columns in summary CSV for 2D plots, if present

rng = np.random.default_rng(RANDOM_STATE)

# ----------------- Load & merge -----------------
feat = pd.read_csv(FEATURES_CSV, low_memory=False)
summ = pd.read_csv(SUMMARY_CSV, low_memory=False)

# choose one row per formula (highest membership_prob)
if "membership_prob" in summ.columns:
    summ = (summ.sort_values(["formula_pretty", "membership_prob"], ascending=[True, False])
                 .drop_duplicates(subset=["formula_pretty"], keep="first")
                 .reset_index(drop=True))

# Make sure formula_pretty exists in both
for dfname, d in [("features", feat), ("summary", summ)]:
    if "formula_pretty" not in d.columns:
        raise ValueError(f"{dfname} is missing 'formula_pretty'")

# Build a clean meta list from summary (avoid duplicates)
meta_candidates = ["formula_pretty","family","band_gap","Ehull_meV_atom","Na_ratio",
                   "cluster","membership_prob",UMAP_LABELS[0],UMAP_LABELS[1]]
keep_meta = [c for c in meta_candidates if c in summ.columns]
# remove dups just in case
keep_meta = list(dict.fromkeys(keep_meta))

# Drop potential meta from features before merge to avoid duplicate columns
drop_if_in_feat = [c for c in keep_meta if c in feat.columns and c != "formula_pretty"]
feat_clean = feat.drop(columns=drop_if_in_feat, errors="ignore")

# Merge (safe)
df = feat_clean.merge(summ[keep_meta].copy(), on="formula_pretty", how="inner", suffixes=("","_meta"))
print(f"Merged rows: {len(df)} | cols: {df.shape[1]}")

# ----------------- Build numeric matrix (exclude meta) -----------------
META_PATTERNS = [
    r"^formula_pretty($|_)",
    r"^family($|_)",
    r"^band_gap($|_)",
    r"^Ehull_meV_atom($|_)",
    r"^Na_ratio($|_)",
    r"^cluster($|_)",
    r"^membership_prob($|_)",
    r"^composition($|_)",
    r"^composition_obj($|_)",
    rf"^{UMAP_LABELS[0]}($|_)", rf"^{UMAP_LABELS[1]}($|_)",
]

def is_meta_col(name: str) -> bool:
    return any(re.match(p, name, flags=re.IGNORECASE) for p in META_PATTERNS)

candidate_cols = [c for c in df.columns if not is_meta_col(c)]
num_cols = [c for c in candidate_cols if is_numeric_dtype(df[c])]
X = df[num_cols].copy()

# Drop all-NaN
all_nan = X.columns[X.isna().all()]
if len(all_nan) > 0:
    X = X.drop(columns=all_nan)

# Coerce numeric & drop all-NaN again
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")
all_nan = X.columns[X.isna().all()]
if len(all_nan) > 0:
    X = X.drop(columns=all_nan)

# Fill missing with column medians (features CSV should already be imputed, but safe)
X = X.fillna(X.median(numeric_only=True))

# Drop-constant
var = X.var(axis=0).to_numpy()
keep = var > 0
X = X.loc[:, keep]
feature_cols = X.columns.tolist()
print(f"Features kept after drop-constant: {X.shape[1]}")

# ----------------- Standardize & PCA -----------------
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# PCA: keep variance (or set an integer)
pca = PCA(n_components=PCA_KEEP_VAR, random_state=RANDOM_STATE)
X_pc = pca.fit_transform(X_std)
pc_names = [f"PC{i+1}" for i in range(X_pc.shape[1])]

# ----------------- Robust / Shrinkage covariance & Mahalanobis -----------------
def robust_cov_mahal(X_cov, random_state=RANDOM_STATE):
    """
    Returns mahal2, cov, mean for given matrix X_cov using MinCovDet with
    Ledoit–Wolf fallback. Uses np.linalg.pinv for stability.
    """
    try:
        mcd = MinCovDet(random_state=random_state, assume_centered=False).fit(X_cov)
        cov  = mcd.covariance_
        mean = mcd.location_
    except Exception:
        lw = LedoitWolf().fit(X_cov)
        cov  = lw.covariance_
        mean = X_cov.mean(axis=0)

    inv_cov = np.linalg.pinv(cov)
    diff = X_cov - mean
    mahal2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return mahal2, cov, mean

print("Fitting MinCovDet for robust Mahalanobis…")
mahal2, cov_pc, mean_pc = robust_cov_mahal(X_pc, random_state=RANDOM_STATE)
df["mahal2"] = mahal2
"""
# ----------------- LOF in PCA space (local outliers) -----------------
lof = LocalOutlierFactor(n_neighbors=LOF_NEIGHBORS, metric="euclidean", n_jobs=-1)
_ = lof.fit_predict(X_pc)
# higher = more outlying; LOF returns negative values where more negative = more outlier
df["lof_score"] = -lof.negative_outlier_factor_
"""
# ----------------- LOF in 2D UMAP space (local outliers) -----------------
if not {UMAP_LABELS[0], UMAP_LABELS[1]}.issubset(df.columns):
    raise KeyError(
        f"UMAP coordinates {UMAP_LABELS} not in df; LOF needs the 2D UMAP "
        "embedding (z1, z2) carried over from the summary CSV."
    )
X_umap = df[[UMAP_LABELS[0], UMAP_LABELS[1]]].to_numpy(dtype=float)
lof = LocalOutlierFactor(n_neighbors=LOF_NEIGHBORS, metric="euclidean", n_jobs=-1)
_ = lof.fit_predict(X_umap)
# higher = more outlying; LOF returns negative values where more negative = more outlier
df["lof_score"] = -lof.negative_outlier_factor_
# ----------------- Flags by quantile -----------------
thr_mahal = np.quantile(df["mahal2"], Q_GLOBAL)
thr_lof   = np.quantile(df["lof_score"], Q_LOF)
df["outlier_mahal_flag"] = df["mahal2"] >= thr_mahal
df["outlier_lof_flag"]   = df["lof_score"] >= thr_lof

# Save main table
out_path = os.path.join(OUT_DIR, "outliers_pca_robustcov_summary.csv")
df_out = df.copy()
df_out.to_csv(out_path, index=False)

print("\n=== Summary ===")
print(f"Rows: {len(df)} | PCs: {X_pc.shape[1]}")
print(f"Global outliers (Mahalanobis top {int((1-Q_GLOBAL)*100)}%): {int(df['outlier_mahal_flag'].sum())}")
print(f"Local outliers  (LOF       top {int((1-Q_LOF)*100)}%): {int(df['outlier_lof_flag'].sum())}")
print("Saved:", out_path)

# ----------------- WHY: per-material explanations -----------------
# Global contributions (original feature space): diff * inv_cov * diff -> per-feature term ~ diff_i * (inv_cov * diff)_i
# We'll compute in PCA space (diagonalizable), then map rough contributions back to original features by loading magnitudes.

inv_cov_pc = np.linalg.pinv(cov_pc)
diff_pc = X_pc - mean_pc  # (n_samples, n_pc)

# Per-PC contribution: (diff_pc @ inv_cov_pc) * diff_pc, keeping diagonal terms
contrib_pc = diff_pc @ inv_cov_pc
contrib_pc *= diff_pc  # elementwise -> per-sample, per-PC contributions; sum over PCs = mahal2

# Map PC contributions back to original features (approx):
# contribution(feature j) ~ sum_i |loading_{j,i}| * contrib_pc_i
# Use absolute loadings normalized per PC to distribute fairly.
loadings = pca.components_.T  # shape (n_features, n_pc)
abs_load = np.abs(loadings)
abs_load = abs_load / (abs_load.sum(axis=0, keepdims=True) + 1e-12)  # normalize over features per PC
# Now compute per-feature contributions
contrib_feat = abs_load @ contrib_pc.T    # shape (n_features, n_samples)
contrib_feat = contrib_feat.T             # (n_samples, n_features)

# Local neighbor deviation (z vs kNN mean/std in PCA space)
from sklearn.neighbors import NearestNeighbors
K_LOCAL = 25
nbrs = NearestNeighbors(n_neighbors=min(K_LOCAL+1, len(df)), metric="euclidean").fit(X_pc)
dist, idx = nbrs.kneighbors(X_pc, return_distance=True)

# Compute local z for each PC per sample
local_z_pc = np.zeros_like(X_pc)
for i in range(len(df)):
    neigh = idx[i, 1:]  # exclude self
    mu = X_pc[neigh].mean(axis=0)
    sd = X_pc[neigh].std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    local_z_pc[i] = (X_pc[i] - mu) / sd

# Map local z in PCs back to features (magnitude via absolute loadings)
local_z_feat = np.abs(local_z_pc) @ abs_load.T  # (n_samples, n_features)

# ---------- Prepare WHY reports for top-K global outliers ----------
order = np.argsort(-df["mahal2"].to_numpy())[:TOPK_WHY]
meta_cols_show = [c for c in ["family","cluster","membership_prob","Ehull_meV_atom","band_gap"] if c in df.columns]
zcols = [c for c in UMAP_LABELS if c in df.columns]

def topk_pairs(names, vals, k=N_TOP_FEATURES, signed=False):
    arr = np.array(vals)
    idx = np.argsort(-np.abs(arr))[:k]
    rows = []
    for j in idx:
        v = arr[j] if signed else np.abs(arr[j])
        rows.append((names[j], float(v)))
    return rows

why_rows = []
for i in order:
    row = df.iloc[i]
    fml = row["formula_pretty"]
    fam = row.get("family","?")
    clu = row.get("cluster","?")
    mp  = row.get("membership_prob", np.nan)
    ehl = row.get("Ehull_meV_atom", np.nan)
    bg  = row.get("band_gap", np.nan)
    m2  = row["mahal2"]
    lof = row["lof_score"]

    # global contributions (approx to original features)
    g_pairs = topk_pairs(feature_cols, contrib_feat[i], k=N_TOP_FEATURES, signed=True)
    # local neighbor deviation (features)
    l_pairs = topk_pairs(feature_cols, local_z_feat[i], k=N_TOP_FEATURES, signed=False)

    # Text report
    report = []
    report.append(f"=== WHY report: {fml} ===")
    report.append(f"family={fam}, cluster={clu}, memb_prob={mp}, Ehull={ehl:.2f} meV/atom, band_gap={bg:.2f} eV")
    report.append(f"LOF score (local) ~ {lof:.3f}   |   Mahalanobis^2 (global) ~ {m2:,.2f}")
    report.append("\nTop global contributors (approx, original features):")
    rep_g = pd.DataFrame(g_pairs, columns=["feature","global_contrib"])
    report.append(rep_g.to_string(index=False))
    report.append("\nTop local (neighbor deviation) contributors:")
    rep_l = pd.DataFrame(l_pairs, columns=["feature","local_z"])
    report.append(rep_l.to_string(index=False))

    txt = "\n".join(report)
    with open(os.path.join(WHY_DIR, f"{i:04d}_{re.sub('[^A-Za-z0-9_]+','_', fml)}.txt"), "w") as fh:
        fh.write(txt)

    # collect a summary row
    why_rows.append({
        "rank": int(np.where(order == i)[0][0]) + 1,
        "formula_pretty": fml,
        "family": fam, "cluster": clu, "membership_prob": mp,
        "Ehull_meV_atom": ehl, "band_gap": bg,
        "mahal2": m2, "lof_score": lof,
        "WHY_file": os.path.join(WHY_DIR, f"{i:04d}_{re.sub('[^A-Za-z0-9_]+','_', fml)}.txt")
    })

why_csv = os.path.join(WHY_DIR, "outlier_explanations_summary.csv")
pd.DataFrame(why_rows).to_csv(why_csv, index=False)

print("\n=== Outlier WHY explainer finished ===")
print("Saved per-material reports to:", WHY_DIR)
print("Summary CSV:", why_csv)
print(f"Rows analysed: {len(df)} | Features used: {len(feature_cols)} | PCs: {X_pc.shape[1]}")

# ----------------- Plots -----------------
plt.figure(figsize=(6,5))
if all(c in df.columns for c in UMAP_LABELS):
    plt.scatter(df[UMAP_LABELS[0]], df[UMAP_LABELS[1]],
                c=df["mahal2"], s=8, alpha=0.9)
    plt.colorbar(label="Mahalanobis$^2$ (global)")
    plt.title("UMAP colored by Mahalanobis$^2$")
    plt.xlabel(UMAP_LABELS[0]); plt.ylabel(UMAP_LABELS[1])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "umap_mahal2.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(6,5))
    plt.scatter(df[UMAP_LABELS[0]], df[UMAP_LABELS[1]],
                c=df["lof_score"], s=8, alpha=0.9)
    plt.colorbar(label="LOF score (local)")
    plt.title("UMAP colored by LOF score")
    plt.xlabel(UMAP_LABELS[0]); plt.ylabel(UMAP_LABELS[1])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "umap_lof.png"), dpi=180)
    plt.close()

# Correlation heatmap (selected variables if present)
corr_vars = [c for c in ["Ehull_meV_atom","band_gap","Na_ratio","mahal2","lof_score"] if c in df.columns]
if "S_conf_dimless" in df.columns: corr_vars.append("S_conf_dimless")
if "n_elements" in df.columns: corr_vars.append("n_elements")

if len(corr_vars) >= 3:
    M = df[corr_vars].copy().dropna()
    rho = M.corr(method="spearman")
    plt.figure(figsize=(6,5))
    im = plt.imshow(rho, vmin=-1, vmax=1)
    plt.colorbar(im, label="Spearman ρ")
    plt.xticks(range(len(corr_vars)), corr_vars, rotation=45, ha="right", fontsize=9)
    plt.yticks(range(len(corr_vars)), corr_vars, fontsize=9)
    plt.title("Spearman correlations")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "spearman_corr.png"), dpi=180)
    plt.close()

print("All plots/CSVs saved in:", OUT_DIR)

# (End of core outlier analysis. All results written to OUT_DIR.)
