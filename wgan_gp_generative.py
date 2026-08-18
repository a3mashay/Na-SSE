#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WGAN-GP generative pipeline with Bayesian optimization and nearest-neighbor matching.

See README.md for the method. Run with --help for inputs and parameters.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from bayes_opt import BayesianOptimization
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import Model, layers

# --------------------------------------------------------------------------
# Configuration as reported in the manuscript
# --------------------------------------------------------------------------
SEED = 72

# WGAN-GP
LATENT_DIM = 16
G_H = [128, 256, 256]      # generator hidden widths
D_H = [256, 256, 128]      # critic hidden widths
GP_LAMBDA = 10.0           # gradient-penalty coefficient
BATCH_SIZE = 64
EPOCHS = 250
N_CRITIC = 5               # critic updates per generator update
INST_NOISE = 0.02          # Gaussian instance noise std
LR_G = 4e-5
LR_D = 3e-4
ADAM_BETA_1 = 0.0
ADAM_BETA_2 = 0.9

# Sampling and screening
N_SYNTHETIC = 40_000
CLIP_FRACTION = 0.98       # keep samples away from the tanh saturation edges
JITTER_STD = 0.005
BAND_GAP_MIN = 2.0         # eV
EHULL_MAX_STABLE = 20.0    # meV/atom
NA_RATIO_MIN = 0.05

# Bayesian optimization
BO_INIT_POINTS = 10
BO_N_ITER = 35
BO_DRAWS = 6000            # generator samples evaluated per BO query
BO_TOP_K = 25              # nearest samples averaged into the objective
BO_WEIGHTS = np.array([50.0, 0.8, 10.0, 5.0], dtype=np.float32)
EHULL_PENALTY_KNEE = 5.0   # meV/atom above which Ehull is penalised
BAND_GAP_REWARD_CAP = 6.0  # eV beyond which the band-gap reward saturates
BAND_GAP_REWARD_SCALE = 8.0
EHULL_PENALTY_SCALE = 0.05

# Nearest-neighbour matching
NN_K = 100
NN_BOOST = np.array([6.0, 2.0, 4.0, 3.0], dtype=np.float64)

COL_EN = "mean Electronegativity"
DESIGN_COLS = ["Na_ratio", "Ehull_meV_atom", "band_gap", COL_EN]

# The 27-dimensional descriptor set. Column names must match the header of
# composition_features.csv exactly.
FEATURE_COLS = [
    # core screening properties
    "band_gap",
    "Ehull_meV_atom",
    "Na_ratio",
    # norm-based composition descriptors
    "0-norm", "2-norm", "3-norm", "5-norm", "7-norm", "10-norm",
    # averaged and range-based elemental descriptors
    "mean AtomicWeight",
    "mean Column",
    "mean Row",
    "range Number",
    "mean Number",
    "range AtomicRadius",
    "mean AtomicRadius",
    "range Electronegativity",
    "mean Electronegativity",
    # Magpie aggregate descriptors
    "MagpieData mean AtomicWeight",
    "MagpieData mean Electronegativity",
    "MagpieData mean GSvolume_pa",
    "MagpieData mean GSbandgap",
    "MagpieData mean GSmagmom",
    # selected element fractions
    "Na fraction",
    "O fraction",
    "P fraction",
    "S fraction",
]


# Explicit generators, so latent draws do not depend on global op-counter state.
TFG: tf.random.Generator | None = None
RNG: np.random.Generator | None = None


def set_seeds(seed: int = SEED, deterministic: bool = True) -> None:
    """Seed every source of randomness and, by default, enable deterministic ops."""
    global TFG, RNG
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)   # seeds python, numpy and tensorflow
    if deterministic and not os.environ.get("TF_DISABLE_OP_DETERMINISM"):
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:           # pragma: no cover - build dependent
            warnings.warn(f"Deterministic kernels unavailable ({exc}); continuing without.")
    TFG = tf.random.Generator.from_seed(seed)
    RNG = np.random.default_rng(seed)


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
def load_na_dataset(path: str):
    """Load the composition-feature table and keep Na-containing entries."""
    df = pd.read_csv(path, low_memory=False)

    formula_col = next(
        (c for c in ("formula_pretty", "pretty_formula", "formula") if c in df.columns), None
    )
    if formula_col is None:
        raise RuntimeError(
            "No formula column found (looked for 'formula_pretty', 'pretty_formula', 'formula')."
        )

    df = df.dropna(subset=[formula_col]).copy()
    df[formula_col] = df[formula_col].astype(str)
    df = df[df[formula_col].str.contains("Na")].reset_index(drop=True)

    if len(df) < 500:
        raise RuntimeError(
            f"Only {len(df)} Na-containing rows after filtering. Check composition_features.csv."
        )
    print(f"Na-containing rows kept: {len(df)}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required descriptor columns: {missing}")

    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"Rows after numeric cleaning: {len(df)}")

    return df, formula_col


def prepare_features(df: pd.DataFrame):
    """3xIQR clipping, MinMax scaling to [-1, 1], and empirical 1-99% bounds."""
    X = df[FEATURE_COLS].copy()

    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    X_clip = X.clip(lower=q1 - 3 * iqr, upper=q3 + 3 * iqr, axis=1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X_clip).astype(np.float32)

    X_train, X_val = train_test_split(X_scaled, test_size=0.10, random_state=SEED, shuffle=True)

    p1, p99 = X.quantile(0.01), X.quantile(0.99)
    emp_bounds = {c: (float(p1[c]), float(p99[c])) for c in FEATURE_COLS}

    print(f"Feature dimension: {X_train.shape[1]} | train {len(X_train)} | held out {len(X_val)}")
    return X_train, X_val, scaler, emp_bounds


# --------------------------------------------------------------------------
# WGAN-GP
# --------------------------------------------------------------------------
def build_generator(latent_dim: int, out_dim: int) -> Model:
    z = layers.Input(shape=(latent_dim,))
    x = z
    for width in G_H:
        x = layers.Dense(width)(x)
        x = layers.LeakyReLU(0.15)(x)
    out = layers.Dense(out_dim, activation="tanh")(x)
    return Model(z, out, name="generator")


def build_critic(in_dim: int) -> Model:
    x_in = layers.Input(shape=(in_dim,))
    x = x_in
    for width in D_H:
        x = layers.Dense(width)(x)
        x = layers.LeakyReLU(0.2)(x)
    score = layers.Dense(1)(x)  # linear Wasserstein score, no sigmoid
    return Model(x_in, score, name="critic")


def gradient_penalty(critic: Model, real: tf.Tensor, fake: tf.Tensor) -> tf.Tensor:
    """Two-sided gradient penalty on interpolates, enforcing the 1-Lipschitz constraint."""
    alpha = TFG.uniform([tf.shape(real)[0], 1], 0.0, 1.0)
    inter = alpha * real + (1.0 - alpha) * fake
    with tf.GradientTape() as tape:
        tape.watch(inter)
        pred = critic(inter)
    grads = tape.gradient(pred, inter)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    return tf.reduce_mean((grad_norm - 1.0) ** 2)


def train_wgan_gp(X_train: np.ndarray, outdir: str, epochs: int = EPOCHS):
    """Train the WGAN-GP and return the generator plus the epoch-averaged history."""
    feat_dim = X_train.shape[1]
    generator = build_generator(LATENT_DIM, feat_dim)
    critic = build_critic(feat_dim)

    g_opt = tf.keras.optimizers.Adam(LR_G, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2)
    d_opt = tf.keras.optimizers.Adam(LR_D, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2)

    train_ds = (
        tf.data.Dataset.from_tensor_slices(X_train)
        .shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=True)
    )

    @tf.function
    def critic_step(real_batch):
        real_noisy = real_batch + TFG.normal(tf.shape(real_batch), stddev=INST_NOISE)
        z = TFG.normal((tf.shape(real_batch)[0], LATENT_DIM))
        with tf.GradientTape() as tape:
            fake = generator(z, training=True)
            fake_noisy = fake + TFG.normal(tf.shape(fake), stddev=INST_NOISE)
            d_loss = tf.reduce_mean(critic(fake_noisy, training=True)) - tf.reduce_mean(
                critic(real_noisy, training=True)
            )
            gp = gradient_penalty(critic, real_noisy, fake_noisy)
            total = d_loss + GP_LAMBDA * gp
        grads = tape.gradient(total, critic.trainable_variables)
        d_opt.apply_gradients(zip(grads, critic.trainable_variables))
        return d_loss, gp

    @tf.function
    def generator_step(batch_size):
        z = TFG.normal((batch_size, LATENT_DIM))
        with tf.GradientTape() as tape:
            g_loss = -tf.reduce_mean(critic(generator(z, training=True), training=True))
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_opt.apply_gradients(zip(grads, generator.trainable_variables))
        return g_loss

    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
    history = {"epoch": [], "d_loss": [], "g_loss": [], "gp": []}

    for epoch in range(1, epochs + 1):
        d_losses, gps, g_losses = [], [], []
        for real_batch in train_ds.take(steps_per_epoch):
            for _ in range(N_CRITIC):
                d_loss, gp = critic_step(real_batch)
                d_losses.append(float(d_loss.numpy()))
                gps.append(float(gp.numpy()))
            g_losses.append(float(generator_step(tf.shape(real_batch)[0]).numpy()))

        history["epoch"].append(epoch)
        history["d_loss"].append(float(np.mean(d_losses)))
        history["g_loss"].append(float(np.mean(g_losses)))
        history["gp"].append(float(np.mean(gps)))

        if epoch == 1 or epoch % 25 == 0:
            print(f"Epoch {epoch:4d} | D loss {history['d_loss'][-1]:.4f} "
                  f"(GP {history['gp'][-1]:.3f}) | G loss {history['g_loss'][-1]:.4f}")

    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(outdir, "wgan_training_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"Training history saved -> {hist_path}")

    gen_path = os.path.join(outdir, "generator_wgan_gp_na_sse.keras")
    generator.save(gen_path)
    print(f"Generator saved -> {gen_path}")

    return generator, hist_df


# --------------------------------------------------------------------------
# Sampling and screening
# --------------------------------------------------------------------------
def sample_synthetic(generator, scaler, emp_bounds, n=N_SYNTHETIC,
                     clip_fraction=CLIP_FRACTION) -> pd.DataFrame:
    """Draw latent vectors, decode, and return samples in physical units."""
    z = RNG.normal(0, 1, size=(n, LATENT_DIM)).astype(np.float32)
    syn_scaled = generator.predict(z, verbose=0)

    syn_scaled += RNG.normal(0, JITTER_STD, size=syn_scaled.shape).astype(np.float32)
    syn_scaled = np.clip(syn_scaled, -clip_fraction, clip_fraction)

    syn = pd.DataFrame(scaler.inverse_transform(syn_scaled), columns=FEATURE_COLS)
    for c in FEATURE_COLS:
        lo, hi = emp_bounds[c]
        syn[c] = syn[c].clip(lo, hi)
    return syn


def post_screen(syn: pd.DataFrame) -> pd.DataFrame:
    """Keep insulating, near-ground-state, Na-bearing synthetic compositions."""
    mask = (
        (syn["band_gap"] > BAND_GAP_MIN)
        & syn["Ehull_meV_atom"].between(0.0, EHULL_MAX_STABLE, inclusive="both")
        & (syn["Na_ratio"] > NA_RATIO_MIN)
    )
    return syn.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------
# Bayesian optimization
# --------------------------------------------------------------------------
def make_bounds(df: pd.DataFrame) -> dict:
    """Search bounds taken from the empirical distribution of the real Na dataset."""
    return {
        "Na_ratio": (
            float(max(0.01, df["Na_ratio"].quantile(0.01))),
            float(min(1.0, df["Na_ratio"].quantile(0.99))),
        ),
        "Ehull_meV_atom": (0.0, float(min(50.0, df["Ehull_meV_atom"].quantile(0.99)))),
        "band_gap": (
            float(max(1.5, df["band_gap"].quantile(0.01))),
            float(min(8.0, df["band_gap"].quantile(0.99))),
        ),
        "mean_EG": (float(df[COL_EN].quantile(0.01)), float(df[COL_EN].quantile(0.99))),
    }


def run_bayesian_optimization(generator, scaler, emp_bounds, df: pd.DataFrame) -> dict:
    """Maximise proximity of generated samples to a trial design point."""

    def objective(Na_ratio, Ehull_meV_atom, band_gap, mean_EG):
        z = RNG.normal(0, 1, size=(BO_DRAWS, LATENT_DIM)).astype(np.float32)
        syn = pd.DataFrame(
            scaler.inverse_transform(generator.predict(z, verbose=0).astype(np.float32)),
            columns=FEATURE_COLS,
        )
        for c in FEATURE_COLS:
            lo, hi = emp_bounds[c]
            syn[c] = syn[c].clip(lo, hi)

        vecs = syn[DESIGN_COLS].to_numpy()
        target = np.array([Na_ratio, Ehull_meV_atom, band_gap, mean_EG], dtype=np.float32)

        # Distance is evaluated in physical units, so the weights also offset the
        # differing numeric ranges of the four descriptors.
        dists = norm((vecs - target) * BO_WEIGHTS, axis=1)
        top = np.partition(dists, BO_TOP_K)[:BO_TOP_K]

        bonus = (
            -EHULL_PENALTY_SCALE * max(0.0, Ehull_meV_atom - EHULL_PENALTY_KNEE)
            + BAND_GAP_REWARD_SCALE * min(band_gap, BAND_GAP_REWARD_CAP)
        )
        return float(bonus - np.mean(top))

    bo = BayesianOptimization(
        f=objective, pbounds=make_bounds(df), random_state=SEED, verbose=2
    )
    bo.maximize(init_points=BO_INIT_POINTS, n_iter=BO_N_ITER)
    print("BO best parameters:", bo.max["params"])
    return bo.max["params"]


def pick_to_target(df_syn: pd.DataFrame, target: dict) -> pd.DataFrame:
    """Screened synthetic sample closest to the BO optimum under the same metric."""
    V = df_syn[DESIGN_COLS].to_numpy()
    t = np.array(
        [target["Na_ratio"], target["Ehull_meV_atom"], target["band_gap"], target["mean_EG"]],
        dtype=np.float32,
    )
    dist = norm((V - t) * BO_WEIGHTS, axis=1)
    return df_syn.iloc[[int(np.argmin(dist))]].copy()


# --------------------------------------------------------------------------
# Nearest-neighbour matching
# --------------------------------------------------------------------------
def nearest_real_neighbors(df: pd.DataFrame, best_syn: pd.DataFrame, k=NN_K) -> pd.DataFrame:
    """Map the optimised feature vector onto known Na compounds.

    Descriptors are z-scored on the real Na dataset, the four design descriptors
    are up-weighted, and the search runs in that weighted space.
    """
    std = StandardScaler().fit(df[FEATURE_COLS])
    real_std = std.transform(df[FEATURE_COLS])

    weights = np.ones(real_std.shape[1], dtype=np.float64)
    weights[[FEATURE_COLS.index(c) for c in DESIGN_COLS]] = NN_BOOST

    real_w = real_std * weights
    best_w = std.transform(best_syn[FEATURE_COLS].to_numpy()) * weights

    nn = NearestNeighbors(n_neighbors=min(k, len(df))).fit(real_w)
    dists, idxs = nn.kneighbors(best_w)

    matches = df.iloc[idxs[0]].copy()
    matches["nn_distance_weighted"] = dists[0]
    return matches.reset_index(drop=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
ARTIFACTS = [
    "synthetic_na_sse_all.csv",
    "synthetic_na_sse_screened.csv",
    "wgan_training_history.csv",
    "generator_wgan_gp_na_sse.keras",
    "bo_optimum_target.csv",
    "bo_best_synthetic_candidate.csv",
    "nn_matches_to_bo_best.csv",
    "synthetic_screened_top100.csv",
]


def archive_run(outdir: str, seed: int) -> None:
    """Copy this run's artifacts to __seedNN names before the next run overwrites them."""
    for fn in ARTIFACTS:
        src = os.path.join(outdir, fn)
        if os.path.exists(src):
            stem, ext = os.path.splitext(fn)
            shutil.copy(src, os.path.join(outdir, f"{stem}__seed{seed}{ext}"))


def restore_run(outdir: str, seed: int) -> None:
    """Copy one archived run back to the canonical filenames."""
    for fn in ARTIFACTS:
        stem, ext = os.path.splitext(fn)
        src = os.path.join(outdir, f"{stem}__seed{seed}{ext}")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(outdir, fn))


def run_pipeline(args, seed: int) -> dict:
    """One complete pass: train, sample, screen, optimise, match. Returns a summary dict."""
    tf.keras.backend.clear_session()
    set_seeds(seed, deterministic=not args.no_determinism)

    df, formula_col = load_na_dataset(args.data)
    X_train, _, scaler, emp_bounds = prepare_features(df)

    generator, _ = train_wgan_gp(X_train, args.outdir, epochs=args.epochs)

    synthetic = sample_synthetic(generator, scaler, emp_bounds, n=args.n_synthetic)
    synthetic.to_csv(os.path.join(args.outdir, "synthetic_na_sse_all.csv"), index=False)
    print(f"Synthetic pool: n = {len(synthetic)}")

    screened = post_screen(synthetic)
    screened.to_csv(os.path.join(args.outdir, "synthetic_na_sse_screened.csv"), index=False)
    print(f"Screened pool:  n = {len(screened)}")
    if screened.empty:
        raise RuntimeError("The physical screen removed every sample; check the training data.")

    best_params = run_bayesian_optimization(generator, scaler, emp_bounds, df)
    pd.DataFrame([best_params]).to_csv(
        os.path.join(args.outdir, "bo_optimum_target.csv"), index=False
    )

    best_syn = pick_to_target(screened, best_params)
    best_syn.to_csv(os.path.join(args.outdir, "bo_best_synthetic_candidate.csv"), index=False)

    nn_matches = nearest_real_neighbors(df, best_syn)
    nn_matches.to_csv(os.path.join(args.outdir, "nn_matches_to_bo_best.csv"), index=False)

    top_screened = screened.sort_values(
        ["Ehull_meV_atom", "band_gap", "Na_ratio"], ascending=[True, False, False]
    ).head(100)
    top_screened.to_csv(os.path.join(args.outdir, "synthetic_screened_top100.csv"), index=False)

    print("\n=== BO-best synthetic candidate (key descriptors) ===")
    print(best_syn[DESIGN_COLS + ["Na fraction", "O fraction", "P fraction", "S fraction"]]
          .to_string(index=False))

    print("\n=== Ten nearest real compounds ===")
    print(nn_matches[[formula_col, "nn_distance_weighted"] + DESIGN_COLS]
          .head(10).to_string(index=False))

    print(f"\nOutputs written to: {args.outdir}")

    return {
        "seed": seed,
        "best_params": best_params,
        "best_syn": best_syn.iloc[0],
        "n_synthetic": len(synthetic),
        "n_screened": len(screened),
        "nn_top10": nn_matches[formula_col].head(10).tolist(),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="composition_features.csv",
                    help="Matminer composition descriptor table")
    ap.add_argument("--outdir", default="na_sse_generative_run", help="Output directory")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="WGAN-GP training epochs")
    ap.add_argument("--n-synthetic", type=int, default=N_SYNTHETIC,
                    help="Size of the synthetic pool")
    ap.add_argument("--seed", type=int, default=SEED, help="Base random seed")
    ap.add_argument("--runs", type=int, default=1,
                    help="Independent repetitions of the whole pipeline (seed, seed+1, ...)")
    ap.add_argument("--no-determinism", action="store_true",
                    help="Skip deterministic kernels (faster; use only within an ensemble)")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    os.makedirs(args.outdir, exist_ok=True)

    results = []
    for i in range(args.runs):
        seed = args.seed + i
        if args.runs > 1:
            print(f"\n{'=' * 60}\n  run {i + 1}/{args.runs}  (seed {seed})\n{'=' * 60}")
        results.append(run_pipeline(args, seed))
        if args.runs > 1:
            archive_run(args.outdir, seed)

    if args.runs == 1:
        print(f"\nAll outputs written to: {args.outdir}")
        return

    summarise_ensemble(results, args.outdir)


def summarise_ensemble(results: list, outdir: str) -> None:
    """Write ensemble means, spreads and neighbour frequencies; restore the medoid run."""
    from collections import Counter

    n = len(results)
    bo_df = pd.DataFrame([{"seed": r["seed"], **r["best_params"]} for r in results])
    syn_df = pd.DataFrame([{
        "seed": r["seed"],
        "n_synthetic": r["n_synthetic"],
        "n_screened": r["n_screened"],
        **{c: float(r["best_syn"][c]) for c in DESIGN_COLS},
    } for r in results])

    hits, ranks = Counter(), {}
    for r in results:
        hits.update(r["nn_top10"])
        for rank, f in enumerate(r["nn_top10"], 1):
            ranks.setdefault(f, []).append(rank)

    nn_df = pd.DataFrame([
        {"formula": f, "runs_in_top10": c, "freq": c / n,
         "mean_rank": float(np.mean(ranks[f])), "best_rank": int(min(ranks[f]))}
        for f, c in hits.items()
    ]).sort_values(["runs_in_top10", "mean_rank"], ascending=[False, True])

    bo_df.to_csv(os.path.join(outdir, "ensemble_bo_params.csv"), index=False)
    syn_df.to_csv(os.path.join(outdir, "ensemble_best_synthetic.csv"), index=False)
    nn_df.to_csv(os.path.join(outdir, "ensemble_nn_frequency.csv"), index=False)

    print(f"\n{'=' * 60}\n  ENSEMBLE OVER {n} RUNS\n{'=' * 60}")
    print("\n--- Bayesian-optimization target ---")
    for c in bo_df.columns.drop("seed"):
        print(f"  {c:18s} {bo_df[c].mean():9.4f} +/- {bo_df[c].std(ddof=1):.4f}")
    print("\n--- best synthetic candidate and pool sizes ---")
    for c in syn_df.columns.drop("seed"):
        print(f"  {c:18s} {syn_df[c].mean():9.4f} +/- {syn_df[c].std(ddof=1):.4f}")
    print(f"\n--- nearest neighbours ({len(nn_df)} distinct phases entered a top-10) ---")
    print(nn_df.head(20).to_string(index=False))

    # representative run: the one whose optimum is closest to the ensemble mean
    key = [c for c in bo_df.columns if c != "seed"]
    sd = bo_df[key].std(ddof=1).replace(0, 1)
    z = (bo_df[key] - bo_df[key].mean()) / sd
    medoid = int(bo_df.loc[(z ** 2).sum(axis=1).idxmin(), "seed"])
    restore_run(outdir, medoid)
    print(f"\nRepresentative run: seed {medoid} (restored to the canonical filenames)")
    print(f"All outputs written to: {outdir}")


if __name__ == "__main__":
    main()
