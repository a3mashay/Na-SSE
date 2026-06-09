# Na-SSE: outlier detection and generative design

Code accompanying the manuscript on data-driven screening and generative
exploration of **sodium solid-state electrolytes (Na-SSEs)**.

This repository contains the two methodological components used in the paper:

|Script|What it does|
|-|-|
|`outlier\_detection.py`|Robust statistical outlier detection — PCA (95% variance) + Minimum-Covariance-Determinant Mahalanobis distance + Local Outlier Factor in the same PCA space; writes outlier tables, per-family/per-cluster counts, and per-material "why" reports. Reproduces the outlier analysis (Tables 1–4).|
|`wgan\_gp\_generative.py`|Generative pipeline — Wasserstein GAN with gradient penalty (WGAN-GP) trained on Na-containing composition descriptors, screening of the synthetic pool, Bayesian optimization toward target descriptors, and weighted k-nearest-neighbor matching back to known chemistries. Reproduces the generative results (Figs 3–4).|

## Citation

If you use this code, please cite the associated manuscript.

## Requirements
numpy; pandas; scipy; scikit-learn; matplotlib; tensorflow>=2.12; bayesian-optimization
