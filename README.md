# Na-SSE

This repository hosts the two core methodological components named in the paper's data and
code availability statement:

| File | Component |
| --- | --- |
| `outlier_detection.py` | Robust statistical outlier detection: PCA-space robust Mahalanobis distance over repeated fits (global) and Local Outlier Factor on the UMAP embedding (local) |
| `wgan_gp_generative.py` | WGAN-GP generative pipeline with Bayesian optimization and weighted nearest-neighbor matching |

Both scripts are standalone and are run from the command line. Each takes the tabular
outputs of the upstream screening pipeline as input; that upstream pipeline (Materials
Project retrieval, family assignment, Matminer featurization, UMAP embedding, HDBSCAN
clustering) is described in full in Section 2 of the paper and in the Supporting
Information, and is not part of this release.

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
---
## 2. `wgan_gp_generative.py`

```bash
python wgan_gp_generative.py \
    --data   composition_features.csv \
    --outdir na_sse_generative_run \
    --runs   10
```
---

## Reproducibility

Both scripts are seeded (17 and 72 respectively), but neither is reproducible by seeding
alone, and both handle this by repetition rather than by asserting determinism.

`outlier_detection.py` is deterministic for a given input table once the covariance is
well conditioned, and the repeat protocol makes that verifiable:
`outlier_repeat_stability.csv` reports the pairwise Jaccard overlap between repeats, which
should sit near 0.95. A value far below that means the covariance is still
ill-conditioned — raise `--support-fraction` or lower `--pca-var` before using the ranking.

The generative pipeline is not bit-for-bit reproducible across TensorFlow builds, thread
counts or GPU kernels, which is normal for adversarial training. Deterministic kernels
make a single seed repeatable on fixed hardware; across hardware, use `--runs` and report
the ensemble. The qualitative outcome reported in the paper is stable across runs, but
individual descriptor values and neighbour orderings should be regenerated rather than
assumed.
