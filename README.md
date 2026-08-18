# Na-SSE

This repository hosts the two core methodological components named in the paper's data and code availability statement:

| File | Component |
| --- | --- |
| `outlier_detection.py` | Robust statistical outlier detection: PCA-space robust Mahalanobis distance over repeated fits (global) and Local Outlier Factor on the UMAP embedding (local) |
| `wgan_gp_generative.py` | WGAN-GP generative pipeline with Bayesian optimization and weighted nearest-neighbor matching |

Both scripts are standalone and are run from the command line. Each takes the tabular
outputs of the upstream screening pipeline as input; that upstream pipeline (Materials
Project retrieval, family assignment, Matminer featurization, UMAP embedding, HDBSCAN
clustering) is described in full in Section 2 of the paper and in the Supporting
Information, and is not part of this release. The input schema is documented below so the
two components can be reproduced or applied to an independently prepared dataset.

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

```bash
python outlier_detection.py \
    --features composition_features.csv \
    --summary  summary_screened_candidates.csv \
    --outdir   analysis_ext \
    --repeats  10
```

### Method

The two scores are evaluated in deliberately different spaces, so they capture
complementary notions of rarity.

**Global.** Descriptors are standardized and reduced by PCA retaining 95% of the variance
(80 components in the published run). In that space, a Minimum Covariance Determinant
(MCD) fit with the support fraction fixed at 0.75 supplies the robust location and the
inlying subset; the covariance is re-estimated on that subset with Ledoit-Wolf shrinkage,
and inverted after discarding eigen-directions below 1e-8 of the largest eigenvalue. This
score measures rarity relative to the full descriptor distribution.

Estimating the covariance with MCD alone is not viable here. At ~80 principal components
the default MCD subset holds only about half the rows, the scatter matrix is numerically
rank-deficient, and a pseudo-inverse turns eigenvalues of order 1e-17 into squared
distances of order 1e15. The flagged set is then not reproducible: repeated fits of the
raw estimator agree on roughly a fifth of it. With the shrinkage estimator above, repeats
agree at a Jaccard of about 0.95.

The global screen is therefore run `--repeats` times (default 10) with independent random
states. For every composition the script records the fraction of repeats in which it was
flagged (`selection_freq`), the mean and standard deviation of its squared Mahalanobis
distance, and its mean rank. Compositions flagged in at least `--consensus` of the repeats
(default 80%) are reported as global outliers.

**Local.** Local Outlier Factor with k = 35 and a Euclidean metric, applied to the
two-dimensional UMAP embedding, i.e. the same low-dimensional map used for HDBSCAN
clustering. This score measures crowding within a compound's immediate neighborhood.
Running LOF in the clustering map rather than in PCA space is intentional and is why the
`z1`/`z2` columns are a hard requirement. LOF operates on fixed coordinates and is
deterministic, so it is computed once rather than repeated.

Compounds in the top 1% of either score are flagged. In the published run each repeat of
the global screen flagged 39 compositions.


## 2. `wgan_gp_generative.py`

```bash
python wgan_gp_generative.py \
    --data   composition_features.csv \
    --outdir na_sse_generative_run \
    --runs   10
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

**Repeated runs.** Seeding alone does not make adversarial training reproducible: the
reduction kernels used for gradient accumulation on graphics hardware are
non-deterministic by default, so two runs with the same seed match on the first epoch to
three decimal places and then drift apart, which is enough to reorder the
nearest-neighbour list. `set_seeds()` enables deterministic kernels and routes every latent
draw through explicitly seeded generators. Beyond that, `--runs N` repeats the whole
pipeline with seeds `seed, seed+1, ...`, archives each run's artifacts under a `__seedNN`
suffix, and writes ensemble summaries. The run whose optimum lies closest to the ensemble
mean is copied back to the canonical filenames and is the one to show wherever a single
representative run is needed.

## Reproducibility

Both scripts are seeded (17 and 72 respectively), but neither is reproducible by seeding
alone, and both handle this by repetition rather than by asserting determinism.

`outlier_detection.py` is deterministic for a given input table once the covariance is
well conditioned, and the repeat protocol makes that verifiable: `outlier_repeat_stability.csv`
reports the pairwise Jaccard overlap between repeats, which should sit near 0.95. A value
far below that means the covariance is still ill-conditioned — raise `--support-fraction`
or lower `--pca-var` before using the ranking.

The generative pipeline is not bit-for-bit reproducible across TensorFlow builds, thread
counts or GPU kernels, which is normal for adversarial training. Deterministic kernels
make a single seed repeatable on fixed hardware; across hardware, use `--runs` and report
the ensemble. The qualitative outcome reported in the paper is stable across
runs, but individual descriptor values and neighbour orderings should be regenerated
rather than assumed.

