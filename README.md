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

Both scripts are standalone and are run from the command line. Each takes the tabular
outputs of the upstream screening pipeline as input; that upstream pipeline (Materials
Project retrieval, family assignment, Matminer featurization, UMAP embedding, HDBSCAN
clustering) is described in full in Section 2 of the paper and in the Supporting
Information, and is not part of this release. The input schema is documented below so the
two components can be reproduced or applied to an independently prepared dataset.

## Scope of this release

The paper's data and code availability statement covers these two components only. The
upstream screening pipeline (Materials Project retrieval, family assignment, Matminer
featurization, UMAP embedding, HDBSCAN clustering), the supervised random-forest cluster
classifier and its descriptor-attribution analysis, the cross-file benchmarking searches,
and the plotting scripts are documented in the paper and Supporting Information but are
not released here. No datasets are deposited with this repository.

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

## Input data

Both scripts read CSV tables produced by the upstream screening pipeline. Neither script
retrieves data from the Materials Project; the API key and retrieval step remain with the
upstream pipeline.

### `composition_features.csv`

One row per compound, keyed on `formula_pretty`, holding the Matminer composition
descriptors together with the tabulated electronic and thermodynamic properties. In the
published run this table held 3,873 pre-filtered Na compositions and 328 retained
descriptors.

`outlier_detection.py` uses every numeric non-metadata column in this file.
`wgan_gp_generative.py` uses only the 27 descriptors listed in `FEATURE_COLS`, which must
be present under exactly these names:

| Group | Columns |
| --- | --- |
| Core screening properties (3) | `band_gap`, `Ehull_meV_atom`, `Na_ratio` |
| Norm-based descriptors (6) | `0-norm`, `2-norm`, `3-norm`, `5-norm`, `7-norm`, `10-norm` |
| Averaged / range elemental (9) | `mean AtomicWeight`, `mean Column`, `mean Row`, `range Number`, `mean Number`, `range AtomicRadius`, `mean AtomicRadius`, `range Electronegativity`, `mean Electronegativity` |
| Magpie aggregates (5) | `MagpieData mean AtomicWeight`, `MagpieData mean Electronegativity`, `MagpieData mean GSvolume_pa`, `MagpieData mean GSbandgap`, `MagpieData mean GSmagmom` |
| Element fractions (4) | `Na fraction`, `O fraction`, `P fraction`, `S fraction` |

`mean AtomicWeight` / `MagpieData mean AtomicWeight` and `mean Electronegativity` /
`MagpieData mean Electronegativity` are deliberately kept as separate features: they are
computed from different elemental reference tables (direct stoichiometric average versus
the Magpie preset).

Units: `band_gap` in eV, `Ehull_meV_atom` in meV atom⁻¹, `Na_ratio` dimensionless,
electronegativity on the Pauling scale.

### `summary_screened_candidates.csv`

Required by `outlier_detection.py` only. One row per compound with the pipeline metadata:

| Column | Meaning |
| --- | --- |
| `formula_pretty` | Reduced formula, the merge key against the feature table |
| `family` | `NASICON`, `β-alumina-like`, `Halide`, `Chalcogenide`, `Borohydride`, or `Other` |
| `band_gap`, `Ehull_meV_atom`, `Na_ratio` | Screening properties |
| `cluster` | HDBSCAN label; `-1` denotes unclustered noise |
| `membership_prob` | HDBSCAN membership probability |
| `z1`, `z2` | 2D UMAP coordinates |

If several rows share a `formula_pretty`, the one with the highest `membership_prob` is
kept so the merge stays one-to-one.

---

## 1. `outlier_detection.py`

```bash
python outlier_detection.py \
    --features composition_features.csv \
    --summary  summary_screened_candidates.csv \
    --outdir   analysis_ext
```

### Method

Two scores are computed once on the final working dataset and are deliberately evaluated
in different spaces, so they capture complementary notions of rarity.

**Global.** Descriptors are standardized and reduced by PCA retaining 95% of the variance
(80 components in the published run). Covariance is estimated in that space with a
Minimum Covariance Determinant estimator, with a Ledoit-Wolf shrinkage fallback if MCD
fails to converge, and inverted with a Moore-Penrose pseudo-inverse so the squared
Mahalanobis distance stays well defined when the covariance is rank-deficient. This score
measures rarity relative to the full descriptor distribution.

**Local.** Local Outlier Factor with k = 35 and a Euclidean metric, applied to the
two-dimensional UMAP embedding, i.e. the same low-dimensional map used for HDBSCAN
clustering. This score measures crowding within a compound's immediate neighborhood.
Running LOF in the clustering map rather than in PCA space is intentional and is why the
`z1`/`z2` columns are a hard requirement.

Compounds in the top 1% of either score are flagged. In the published run each criterion
flagged 39 compositions and the two overlapped in only two cases.

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

```bash
python wgan_gp_generative.py \
    --data   composition_features.csv \
    --outdir na_sse_generative_run
```

### Method

**Preprocessing.** Na-containing rows are selected and represented by the 27 descriptors
above. Features are clipped with a 3×IQR rule to reduce the influence of extreme values,
then scaled to [−1, 1] to match the generator's tanh output. A 10% split is held out. The
empirical 1st–99th percentile range of each feature in the real Na dataset is retained and
used later to clip generated samples back onto the data manifold.

**WGAN-GP.** The generator maps 16-dimensional latent vectors through fully connected
layers of 128, 256 and 256 units with LeakyReLU activations to a tanh output in the same
27-dimensional space. The critic is a 256-256-128 fully connected network with LeakyReLU
activations producing a scalar Wasserstein score. Training uses the gradient-penalty
formulation (λ = 10) to enforce the Lipschitz constraint, five critic updates per
generator update, small Gaussian instance noise added to real and generated samples, and
separate Adam optimizers for the two networks, over 250 epochs.

**Sampling and screening.** Generated vectors are lightly jittered, clipped away from the
tanh saturation edges, inverse-transformed to physical units, and clipped to the empirical
1–99% bounds. A first screen keeps band gap > 2 eV (electronic insulation), energy above
hull within 0–20 meV atom⁻¹ (thermodynamic proximity to stable phases), and Na_ratio >
0.05 (sodium remains a significant component).

**Bayesian optimization.** The search runs over four physical descriptors — `Na_ratio`,
`Ehull_meV_atom`, `band_gap` and `mean Electronegativity` — with bounds drawn from the
empirical distributions of the real Na set. Each query samples the generator, computes a
weighted Euclidean distance to the trial target, and averages the 25 smallest distances.
Because the distance is evaluated in physical units, the weights also offset the differing
numeric ranges of the four descriptors. The reward penalises energies above hull beyond
about 5 meV atom⁻¹ and saturates the band-gap term at 6 eV, so band gaps above that limit
are not rewarded further — worth bearing in mind when reading the returned optimum.

**Nearest-neighbor matching.** The optimum is a coordinate in descriptor space, not a
composition. To recover interpretable chemistry, the screened synthetic sample closest to
that coordinate is mapped onto the real Na dataset through a weighted k-nearest-neighbor
search (k = 100) over z-scored descriptors, again emphasizing the four design descriptors.

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

Both scripts are seeded (17 and 72 respectively). `outlier_detection.py` is deterministic
for a given input table. The generative pipeline is not bit-for-bit reproducible across
different TensorFlow builds, thread counts or GPU kernels, which is normal for adversarial
training: re-running can shift the optimized coordinate and reorder the nearest-neighbor
list. The qualitative outcome reported in the paper — convergence toward sodium-dilute,
near-ground-state, wide-gap, fluorine-rich frameworks — is stable across runs, but exact
descriptor values should be regenerated rather than assumed.

Figures in the paper are produced from the CSV outputs above; the plotting scripts are not
part of this release.

## Citation

```bibtex
@article{mashayekhi_na_sse,
  title   = {Correlation and Outlier Analysis of Sodium Solid-State Electrolytes
             Using High-Throughput Data Mining},
  author  = {Mashayekhi, Alireza and Khazraei, Sepehr and Bekou, Jack},
  note    = {Manuscript},
  year    = {2025}
}
```

Please also cite the underlying tools: Materials Project, pymatgen, Matminer, UMAP,
HDBSCAN, scikit-learn, TensorFlow, and `bayesian-optimization`.

## License

MIT. See `LICENSE`.
