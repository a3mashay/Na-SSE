# Na-SSE

Code released with:

> **Correlation and Outlier Analysis of Sodium Solid-State Electrolytes Using High-Throughput Data Mining**

> Alireza Mashayekhi, Sepehr Khazraei, Jack Bekou
> Correspondence: amashayekhi@flexngate.com

This repository hosts the two core methodological components:

| File | Component |
| --- | --- |
| `outlier_detection.py` | Robust statistical outlier detection: PCA-space Minimum-Covariance-Determinant Mahalanobis distance (global) and Local Outlier Factor on the UMAP embedding (local) |
| `wgan_gp_generative.py` | WGAN-GP generative pipeline with Bayesian optimization and weighted nearest-neighbor matching |

---

## Installation

```bash
git clone https://github.com/a3mashay/Na-SSE.git
cd Na-SSE
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9 or newer. `outlier_detection.py` needs only NumPy, pandas, SciPy and
scikit-learn; TensorFlow and `bayesian-optimization` are required by
`wgan_gp_generative.py` alone. A GPU is not required; the generative run trains in a few
minutes on CPU at the dataset size used in the paper.

---

## 1. `outlier_detection.py`

### Method

### Key parameters

| Parameter | Value | Flag |
| --- | --- | --- |
| Random seed | 17 (shared with the UMAP embedding) | — |
| PCA variance retained | 0.95 | `--pca-var` |
| LOF neighbors (k) | 35 | `--lof-neighbors` |
| Flag quantile | 0.99 on each score | `--quantile` |

### Outputs

| File | Contents |
| --- | --- |
| `outliers_pca_robustcov_summary.csv` | Every row with `mahal2`, `lof_score`, and both flags |
| `outliers_top_global.csv` | Top global outliers — supports **Table S1** |
| `outliers_top_local.csv` | Top local outliers — supports **Table S2** |
| `outliers_by_family.csv` | Outlier counts and shares by chemical family — supports **Table 1** |
| `outliers_by_cluster.csv` | Outlier counts and shares by cluster — supports **Table S3** |

---

## 2. `wgan_gp_generative.py`

### Method

### Key parameters

| Parameter | Value |
| --- | --- |
| Random seed | 72 |
| Latent dimension | 16 |
| Generator / critic widths | 128-256-256 / 256-256-128 |
| Gradient penalty λ | 10 |
| Critic steps per generator step | 5 |
| Instance noise σ | 0.02 |
| Optimizers | Adam, lr 4×10⁻⁵ (generator) and 3×10⁻⁴ (critic), β₁ = 0, β₂ = 0.9 |
| Batch size / epochs | 64 / 250 |
| Synthetic pool | 40,000 samples |
| Screen | E_g > 2 eV, 0 ≤ E_hull ≤ 20 meV atom⁻¹, Na_ratio > 0.05 |
| BO budget | 10 initial points + 35 iterations, 6,000 generator draws per query, top-25 averaged |
| BO distance weights | Na_ratio 50, E_hull 0.8, band gap 10, mean EN 5 |
| NN search | k = 100, weight boosts 6 / 2 / 4 / 3 on the four design descriptors |

Useful flags: `--epochs`, `--n-synthetic`, `--seed`.

### Outputs

| File | Contents |
| --- | --- |
| `wgan_training_history.csv` | Epoch-averaged critic loss, generator loss and gradient penalty — supports **Figure 4b,c** |
| `generator_wgan_gp_na_sse.keras` | Trained generator |
| `synthetic_na_sse_all.csv` | Full synthetic pool — supports **Figure 3** and the t-SNE overlay in **Figure 4a** |
| `synthetic_na_sse_screened.csv` | Pool after the physical screen |
| `bo_optimum_target.csv` | The optimized coordinate in descriptor space |
| `bo_best_synthetic_candidate.csv` | Screened sample closest to that coordinate |
| `nn_matches_to_bo_best.csv` | The 100 nearest real compounds, with weighted distances |
| `synthetic_screened_top100.csv` | Top-100 screened candidates by low E_hull, wide gap, higher Na content |

---

## Reproducibility

Both scripts are seeded (17 and 72). outlier_detection.py is deterministic — the same inputs give the same outputs.

wgan_gp_generative.py may not reproduce the paper's numbers exactly. GAN training varies with TensorFlow version and hardware, and both the sampling and the Bayesian-optimization objective are stochastic, so the optimized point and the neighbor ordering can shift between runs. The overall design region is reproducible; individual descriptor values should be regenerated rather than quoted.

Figures are produced from the CSV outputs above; the plotting scripts are not part of this release.

