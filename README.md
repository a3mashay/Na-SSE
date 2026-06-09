# Na-SSE: Outlier Detection and Generative Design

Code accompanying the manuscript on data-driven screening and generative
exploration of **sodium solid-state electrolytes (Na-SSEs)**.

This repository contains the two methodological components used in the paper:

|Script|What it does|
|-|-|
|`outlier_detection.py`|Identifies statistically anomalous Na-SSE candidates using a robust, two-pronged approach. Composition descriptors are first reduced by PCA (retaining 95% of the variance); global outliers are then flagged by a Minimum-Covariance-Determinant Mahalanobis distance, and local outliers by a Local Outlier Factor computed in the same PCA space. Outputs include ranked outlier tables, per-family and per-cluster enrichment counts, and per-material "why" reports explaining each flag. Reproduces the outlier analysis (Tables 1–4).|
|`wgan_gp_generative.py`|Explores composition space beyond the screened dataset. A Wasserstein GAN with gradient penalty (WGAN-GP) is trained on Na-containing composition descriptors to generate synthetic candidates; the synthetic pool is screened against basic Na-SSE criteria, steered by Bayesian optimization toward target properties (near-ground-state, wide-gap, moderately Na-rich), and mapped back to real chemistries through weighted k-nearest-neighbor matching. Reproduces the generative results (Figs 3–4).|

## Citation

If you use this code, please cite the associated manuscript.

## Requirements
numpy; pandas; scipy; scikit-learn; matplotlib; tensorflow>=2.12; bayesian-optimization
