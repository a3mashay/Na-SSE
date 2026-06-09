"""
Na-SSE Generative Pipeline (WGAN-GP + Bayesian optimization + nearest-neighbor matching)
========================================================================================
Generative exploration of sodium solid-state electrolyte (Na-SSE) composition space,
as used in the manuscript.

Method:
  * Represent each Na-containing composition with a compact descriptor vector,
    scale to [-1, 1], and train a Wasserstein GAN with gradient penalty (WGAN-GP).
  * Sample a large synthetic pool, screen it for basic SSE requirements
    (band gap, energy above hull, Na content), and steer the generator with
    Bayesian optimization toward moderately Na-rich, near-ground-state, wide-gap targets.
  * Map the optimized point back to known chemistries with a weighted
    k-nearest-neighbor search.

Inputs (CSV):
  --data   composition feature table; Na-containing rows are selected automatically
           (must contain a formula column and the descriptor columns used below)

Outputs (written to --outdir):
  trained-model artifacts, synthetic candidate tables, the Bayesian-optimization
  best point, and the nearest-neighbor matches to that point.

Usage:
  python wgan_gp_generative.py --data data/composition_features.csv \
                               --outdir outputs/generative_run

Requires TensorFlow, scikit-learn, and bayesian-optimization (bayes_opt).
If you use this code, please cite the associated manuscript.
"""

# ============================================
# Na-SSE Generative Pipeline (Colab-ready)
# WGAN-GP + BO + NN matching
# Using composition_features.csv schema
# ============================================

# %% ---- Imports & setup
import os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# TF / Keras
import tensorflow as tf
from tensorflow.keras import layers, Model

# Sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Bayesian Optimization
from bayes_opt import BayesianOptimization
from numpy.linalg import norm

# Reproducibility
SEED = 72
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ---- Paths (configurable)
import argparse
_ap = argparse.ArgumentParser(description="Na-SSE WGAN-GP + Bayesian optimization + NN matching")
_ap.add_argument("--data",   default="data/composition_features.csv",
                 help="composition feature table containing Na entries (CSV)")
_ap.add_argument("--outdir", default="outputs/generative_run",
                 help="output directory")
_args, _ = _ap.parse_known_args()

PATH_DATA = _args.data
OUT_DIR   = _args.outdir
os.makedirs(OUT_DIR, exist_ok=True)

# %% ---- Load dataset
df = pd.read_csv(PATH_DATA, low_memory=False)

# ---- Identify formula column ----
formula_col = None
for cand in ["formula_pretty", "pretty_formula", "formula"]:
    if cand in df.columns:
        formula_col = cand
        break

if formula_col is None:
    raise RuntimeError(
        "No formula-like column found (tried 'formula_pretty', 'pretty_formula', 'formula')."
    )

df = df.dropna(subset=[formula_col]).copy()
df[formula_col] = df[formula_col].astype(str)

# ---- Filter Na-containing entries ----
df = df[df[formula_col].str.contains("Na")]
df = df.reset_index(drop=True)

if len(df) < 500:
    raise RuntimeError(
        f"Too few Na entries after Na filtering: {len(df)}. "
        "Broaden filter or check composition_features.csv."
    )

print(f"Na-containing rows kept: {len(df)}")

# %% ---- Feature set (must match your header exactly)
# Core physics + Na content + Magpie stats
feature_cols = [
    # core target properties
    "band_gap",
    "Ehull_meV_atom",
    "Na_ratio",

    # norm-based composition descriptors
    "0-norm",
    "2-norm",
    "3-norm",
    "5-norm",
    "7-norm",
    "10-norm",

    # basic averaged elemental properties
    "mean AtomicWeight",
    "mean Column",
    "mean Row",
    "range Number",
    "mean Number",
    "range AtomicRadius",
    "mean AtomicRadius",
    "range Electronegativity",
    "mean Electronegativity",

    # Magpie-based aggregate descriptors
    "MagpieData mean AtomicWeight",
    "MagpieData mean Electronegativity",
    "MagpieData mean GSvolume_pa",
    "MagpieData mean GSbandgap",
    "MagpieData mean GSmagmom",

    # key element fractions (Na-centered)
    "Na fraction",
    "O fraction",
    "P fraction",
    "S fraction",
]

# Ensure all required columns exist
for c in feature_cols:
    if c not in df.columns:
        raise RuntimeError(f"Required feature column '{c}' not found in composition_features.csv")

# Coerce numerics
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=feature_cols).reset_index(drop=True)
print(f"Rows after numeric cleaning: {len(df)}")

# %% ---- Train/test split & scaling
X = df[feature_cols].copy()

# Robust clipping via IQR to soften strong outliers before MinMax scaling
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
X_clip = X.clip(lower=(Q1 - 3 * IQR), upper=(Q3 + 3 * IQR), axis=1)

# Scale to [-1, 1] to match tanh output of generator
mm = MinMaxScaler(feature_range=(-1, 1))
X_scaled = mm.fit_transform(X_clip).astype(np.float32)

X_train, X_val = train_test_split(
    X_scaled, test_size=0.10, random_state=SEED, shuffle=True
)

# Empirical 1–99% bounds in original units for safety clipping later
p1 = X.quantile(0.01)
p99 = X.quantile(0.99)
emp_bounds = {c: (p1[c], p99[c]) for c in feature_cols}

feat_dim = X_train.shape[1]
print(f"Feature dimension: {feat_dim}")

# %% ---- WGAN-GP configuration
LATENT_DIM = 16
G_H = [128, 256, 256]   # generator hidden sizes
D_H = [256, 256, 128]   # critic hidden sizes
GP_LAMBDA = 10.0
BATCH_SIZE = 64
EPOCHS = 250
N_CRITIC = 5
INST_NOISE = 0.02

# --- Generator ---
def build_generator(latent_dim=LATENT_DIM, out_dim=feat_dim):
    z = layers.Input(shape=(latent_dim,))
    x = layers.Dense(G_H[0])(z)
    x = layers.LeakyReLU(0.15)(x)
    x = layers.Dense(G_H[1])(x)
    x = layers.LeakyReLU(0.15)(x)
    x = layers.Dense(G_H[2])(x)
    x = layers.LeakyReLU(0.15)(x)
    out = layers.Dense(out_dim, activation="tanh")(x)
    return Model(z, out, name="generator")

# --- Critic (WGAN discriminator) ---
def build_critic(in_dim=feat_dim):
    x_in = layers.Input(shape=(in_dim,))
    x = layers.Dense(D_H[0])(x_in)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(D_H[1])(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(D_H[2])(x)
    x = layers.LeakyReLU(0.2)(x)
    score = layers.Dense(1)(x)  # linear WGAN score
    return Model(x_in, score, name="critic")

generator = build_generator()
critic = build_critic()

# slightly smaller critic LR to soften the first steps
g_opt = tf.keras.optimizers.Adam(learning_rate=4e-5, beta_1=0.0, beta_2=0.9)
d_opt = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.0, beta_2=0.9)

# --- Gradient penalty ---
def gradient_penalty(f, real, fake):
    alpha = tf.random.uniform([real.shape[0], 1], 0.0, 1.0)
    inter = alpha * real + (1.0 - alpha) * fake
    with tf.GradientTape() as t:
        t.watch(inter)
        pred = f(inter)
    grads = t.gradient(pred, inter)
    norm_val = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    gp = tf.reduce_mean((norm_val - 1.0) ** 2)
    return gp

# Dataset pipeline
train_ds = (
    tf.data.Dataset.from_tensor_slices(X_train)
    .shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=True)
)

# --- One critic step ---
@tf.function
def train_critic_step(real_batch):
    real_noisy = real_batch + tf.random.normal(tf.shape(real_batch), stddev=INST_NOISE)
    z = tf.random.normal((tf.shape(real_batch)[0], LATENT_DIM))

    with tf.GradientTape() as tape:
        fake = generator(z, training=True)
        fake_noisy = fake + tf.random.normal(tf.shape(fake), stddev=INST_NOISE)

        d_real = critic(real_noisy, training=True)
        d_fake = critic(fake_noisy, training=True)

        d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
        gp = gradient_penalty(critic, real_noisy, fake_noisy)
        d_loss_total = d_loss + GP_LAMBDA * gp

    grads = tape.gradient(d_loss_total, critic.trainable_variables)
    d_opt.apply_gradients(zip(grads, critic.trainable_variables))
    return d_loss, gp

# --- One generator step ---
@tf.function
def train_generator_step(batch_size):
    z = tf.random.normal((batch_size, LATENT_DIM))
    with tf.GradientTape() as tape:
        fake = generator(z, training=True)
        d_fake = critic(fake, training=True)
        g_loss = -tf.reduce_mean(d_fake)
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(grads, generator.trainable_variables))
    return g_loss

# %% ---- Train WGAN-GP with *epoch-averaged* history logging
d_loss_history = []
g_loss_history = []
gp_history     = []
epoch_history  = []

steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)

for epoch in range(1, EPOCHS + 1):
    d_losses_epoch = []
    gp_epoch       = []
    g_losses_epoch = []

    for real_batch in train_ds.take(steps_per_epoch):
        # critic N_CRITIC steps
        for _ in range(N_CRITIC):
            d_loss, gp = train_critic_step(real_batch)
            d_losses_epoch.append(float(d_loss.numpy()))
            gp_epoch.append(float(gp.numpy()))

        # generator step
        g_loss = train_generator_step(tf.shape(real_batch)[0])
        g_losses_epoch.append(float(g_loss.numpy()))

    # epoch-wise averages
    d_loss_mean = float(np.mean(d_losses_epoch))
    g_loss_mean = float(np.mean(g_losses_epoch))
    gp_mean     = float(np.mean(gp_epoch))

    d_loss_history.append(d_loss_mean)
    g_loss_history.append(g_loss_mean)
    gp_history.append(gp_mean)
    epoch_history.append(epoch)

    if epoch % 25 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:4d} | "
            f"D loss: {d_loss_mean:.4f} (GP {gp_mean:.3f}) | "
            f"G loss: {g_loss_mean:.4f}"
        )

# save history for plotting later (Figure G3)
hist_df = pd.DataFrame({
    "epoch": epoch_history,
    "d_loss": d_loss_history,
    "g_loss": g_loss_history,
    "gp": gp_history,
})
HIST_PATH = os.path.join(OUT_DIR, "wgan_training_history.csv")
hist_df.to_csv(HIST_PATH, index=False)
print(f"📁 WGAN training history saved → {HIST_PATH}")

# Save generator
GEN_PATH = os.path.join(OUT_DIR, "generator_wgan_gp_na_sse.keras")
generator.save(GEN_PATH)
print(f"✅ Saved generator → {GEN_PATH}")

# %% ---- Sampling + inverse scaling
def sample_synthetic(n=20000, clip_fraction=0.98):
    z = np.random.normal(0, 1, size=(n, LATENT_DIM)).astype(np.float32)
    syn_scaled = generator.predict(z, verbose=0)

    # small jitter + clip away from tanh edges
    syn_scaled += np.random.normal(0, 0.005, size=syn_scaled.shape).astype(np.float32)
    clip_val = float(clip_fraction)
    syn_scaled = np.clip(syn_scaled, -clip_val, clip_val)

    syn = mm.inverse_transform(syn_scaled)
    syn = pd.DataFrame(syn, columns=feature_cols)

    # final safety clip inside 1–99% empirical bounds
    for c in feature_cols:
        lo, hi = emp_bounds[c]
        syn[c] = syn[c].clip(lo, hi)

    return syn

synthetic_df = sample_synthetic(n=40000)
SYN_PATH = os.path.join(OUT_DIR, "synthetic_na_sse_all.csv")
synthetic_df.to_csv(SYN_PATH, index=False)
print(f"📁 Synthetic set saved → {SYN_PATH} (n={len(synthetic_df)})")

# %% ---- Post-screen for Na-SSE-like behavior
EHULL_MAX_STABLE = 20.0  # meV/atom, adjust if needed

def post_screen(df_in):
    df_out = df_in.copy()

    conds = []
    # Band gap > 2 eV (insulating window)
    conds.append(df_out["band_gap"] > 2.0)
    # Ehull close to zero, in meV/atom
    conds.append(df_out["Ehull_meV_atom"].between(0.0, EHULL_MAX_STABLE, inclusive="both"))
    # Some Na content
    conds.append(df_out["Na_ratio"] > 0.05)

    mask = np.logical_and.reduce(conds)
    return df_out.loc[mask].reset_index(drop=True)

synthetic_screened = post_screen(synthetic_df)
SCREEN_PATH = os.path.join(OUT_DIR, "synthetic_na_sse_screened.csv")
synthetic_screened.to_csv(SCREEN_PATH, index=False)
print(f"📁 Screened synthetic set saved → {SCREEN_PATH} (n={len(synthetic_screened)})")

# %% ---- Bayesian Optimization on (Na_ratio, Ehull, Eg, mean Electronegativity)

col_EN = "mean Electronegativity"

pbounds = {
    "Na_ratio": (
        float(max(0.01, df["Na_ratio"].quantile(0.01))),
        float(min(1.0, df["Na_ratio"].quantile(0.99))),
    ),
    "Ehull_meV_atom": (
        0.0,
        float(min(50.0, df["Ehull_meV_atom"].quantile(0.99))),
    ),
    "band_gap": (
        float(max(1.5, df["band_gap"].quantile(0.01))),
        float(min(8.0, df["band_gap"].quantile(0.99))),
    ),
    "mean_EG": (
        float(df[col_EN].quantile(0.01)),
        float(df[col_EN].quantile(0.99)),
    ),
}

def bo_objective(Na_ratio, Ehull_meV_atom, band_gap, mean_EG,
                 draws=6000, top_k=25):
    z = np.random.normal(0, 1, size=(draws, LATENT_DIM)).astype(np.float32)
    syn_scaled = generator.predict(z, verbose=0).astype(np.float32)
    syn = mm.inverse_transform(syn_scaled)
    syn = pd.DataFrame(syn, columns=feature_cols)

    # empirical clipping
    for c in feature_cols:
        lo, hi = emp_bounds[c]
        syn[c] = syn[c].clip(lo, hi)

    vecs = syn[["Na_ratio", "Ehull_meV_atom", "band_gap", col_EN]].values
    target = np.array([Na_ratio, Ehull_meV_atom, band_gap, mean_EG], dtype=np.float32)

    # weights: push Ehull down, push Eg moderately high, enforce reasonable Na_ratio and EN
    w = np.array([50.0, 0.8, 10.0, 5.0], dtype=np.float32)
    dists = norm((vecs - target) * w, axis=1)
    top = np.partition(dists, top_k)[:top_k]

    # reward shaping – favour low Ehull and wide band gap
    bonus = (
        -0.05 * max(0.0, Ehull_meV_atom - 5.0) +   # penalize Ehull above ~5 meV/atom
         8.0 * min(band_gap, 6.0)                  # diminishing returns after ~6 eV
    )
    return float(bonus - np.mean(top))

bo = BayesianOptimization(
    f=bo_objective,
    pbounds=pbounds,
    random_state=SEED,
    verbose=2
)

bo.maximize(init_points=10, n_iter=35)
best = bo.max["params"]
print("🌟 BO best parameters:", best)

def pick_to_target(df_syn, target_dict):
    V = df_syn[["Na_ratio", "Ehull_meV_atom", "band_gap", col_EN]].values
    target = np.array([
        target_dict["Na_ratio"],
        target_dict["Ehull_meV_atom"],
        target_dict["band_gap"],
        target_dict["mean_EG"],
    ], dtype=np.float32)
    w = np.array([50.0, 0.8, 10.0, 5.0], dtype=np.float32)
    dist = norm((V - target) * w, axis=1)
    idx = int(np.argmin(dist))
    return df_syn.iloc[[idx]].copy()

best_syn = pick_to_target(synthetic_screened, best)
BEST_SYN_PATH = os.path.join(OUT_DIR, "bo_best_synthetic_candidate.csv")
best_syn.to_csv(BEST_SYN_PATH, index=False)
print(f"📁 BO-best synthetic candidate saved → {BEST_SYN_PATH}")

# %% ---- Nearest Neighbor matching to real Na entries

std = StandardScaler().fit(df[feature_cols])
real_std = std.transform(df[feature_cols])

nn = NearestNeighbors(n_neighbors=100).fit(real_std)

# weight key features more: Na_ratio, Ehull, Eg, mean EN
boost_cols = ["Na_ratio", "Ehull_meV_atom", "band_gap", col_EN]
boost_idx = [feature_cols.index(c) for c in boost_cols]

weights = np.ones((real_std.shape[1],), dtype=np.float32)
weights[boost_idx] = np.array([6.0, 2.0, 4.0, 3.0], dtype=np.float32)

def weighted_transform(M, w):
    return M * w

best_vec = best_syn[feature_cols].values
best_vec_std = std.transform(best_vec)

real_w = weighted_transform(real_std, weights)
best_w = weighted_transform(best_vec_std, weights)

dists, idxs = nn.kneighbors(best_w)

rows = []
for d, i in zip(dists[0], idxs[0]):
    row = df.iloc[i].copy()
    row["nn_distance_weighted"] = d
    rows.append(row)

nn_matches = pd.DataFrame(rows)

NN_PATH = os.path.join(OUT_DIR, "nn_matches_to_bo_best.csv")
nn_matches.to_csv(NN_PATH, index=False)
print(f"📁 NN matches to BO-best saved → {NN_PATH}")

# %% ---- Export Top-100 screened candidates
top_screened = synthetic_screened.sort_values(
    ["Ehull_meV_atom", "band_gap", "Na_ratio"],
    ascending=[True, False, False]
).head(100)

TOP100_PATH = os.path.join(OUT_DIR, "synthetic_screened_top100.csv")
top_screened.to_csv(TOP100_PATH, index=False)
print(f"📁 Top-100 screened saved → {TOP100_PATH}")

# %% ---- Quick console preview
print("\n=== BO-best synthetic (key features) ===")
print(best_syn[[
    "Na_ratio",
    "Ehull_meV_atom",
    "band_gap",
    col_EN,
    "Na fraction",
    "O fraction",
    "P fraction",
    "S fraction",
]])

print("\n=== Top 10 NN matches (formula, distance, Ehull, Eg) ===")
cols_show = [
    formula_col,
    "nn_distance_weighted",
    "Ehull_meV_atom",
    "band_gap",
    "Na_ratio",
    col_EN,
]
print(nn_matches[cols_show].head(10).to_string(index=False))
