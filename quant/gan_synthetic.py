"""
GAN — Generative Adversarial Network for synthetic return sequences.

Trains a Generator G(z) to produce return sequences that fool a
Discriminator D(x) into thinking they are real historical returns.

Architecture (from quant-traderr-lab):
    G: z(Z_DIM) → Dense(64, ReLU) → Dense(128, ReLU) → Dense(SEQ_LEN, tanh)
    D: x(SEQ_LEN) → Dense(128, LReLU) → Dense(64, LReLU) → Dense(1, sigmoid)

Loss functions:
    D: -E[log D(x)] - E[log(1 - D(G(z)))]     (standard minimax)
    G: -E[log D(G(z))]                          (non-saturating trick)

Implementation uses FINITE DIFFERENCES for gradient estimation instead
of autograd to remove external dependencies.  This is slower but keeps
the module self-contained with pure NumPy.

Application:
    - Generate synthetic scenarios for stress testing
    - Augment thin historical data for backtesting
    - Compare real vs. generated return distributions (fat tails, skew)
    - Detect distribution shift: if D can easily tell real from fake,
      the market regime may have changed

Reference
---------
quant-traderr-lab / GAN / GAN pipeline.py
Uses autograd for backprop. This version uses finite-difference gradients
for zero-dependency operation.

Limitations
-----------
- Pure NumPy finite-diff training is ~10x slower than autograd.
  Default epochs reduced to 500 for reasonable runtime.
- No convolutional layers — fully connected MLP only.
- Best suited for short return sequences (SEQ_LEN ≤ 50).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd


def _clean_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    s = prices.dropna()
    s = s[s > 0]
    return s


# ── activation functions ────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))

def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def _leaky_relu_grad(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha)

def _sigmoid_grad(s: np.ndarray) -> np.ndarray:
    return s * (1.0 - s)

def _tanh_grad(t: np.ndarray) -> np.ndarray:
    return 1.0 - t ** 2


# ── forward passes ──────────────────────────────────────────────────

def _generator_forward(params: List[np.ndarray], z: np.ndarray):
    """G: z → Dense(ReLU) → Dense(ReLU) → Dense(tanh)"""
    W1, b1, W2, b2, W3, b3 = params
    h1_pre = z @ W1 + b1
    h1 = _relu(h1_pre)
    h2_pre = h1 @ W2 + b2
    h2 = _relu(h2_pre)
    out_pre = h2 @ W3 + b3
    out = np.tanh(out_pre)
    cache = (z, h1_pre, h1, h2_pre, h2, out_pre, out)
    return out, cache


def _discriminator_forward(params: List[np.ndarray], x: np.ndarray):
    """D: x → Dense(LReLU) → Dense(LReLU) → Dense(sigmoid)"""
    W1, b1, W2, b2, W3, b3 = params
    h1_pre = x @ W1 + b1
    h1 = _leaky_relu(h1_pre)
    h2_pre = h1 @ W2 + b2
    h2 = _leaky_relu(h2_pre)
    logit = h2 @ W3 + b3
    out = _sigmoid(logit)
    cache = (x, h1_pre, h1, h2_pre, h2, logit, out)
    return out, cache


# ── backward passes ─────────────────────────────────────────────────

def _discriminator_backward(params, cache, d_out):
    """Backprop through discriminator. d_out = dL/d(sigmoid_output)."""
    W1, b1, W2, b2, W3, b3 = params
    x, h1_pre, h1, h2_pre, h2, logit, sig_out = cache
    bs = x.shape[0]

    d_logit = d_out * _sigmoid_grad(sig_out)
    dW3 = h2.T @ d_logit / bs
    db3 = d_logit.mean(axis=0)

    d_h2 = d_logit @ W3.T
    d_h2_pre = d_h2 * _leaky_relu_grad(h2_pre)
    dW2 = h1.T @ d_h2_pre / bs
    db2 = d_h2_pre.mean(axis=0)

    d_h1 = d_h2_pre @ W2.T
    d_h1_pre = d_h1 * _leaky_relu_grad(h1_pre)
    dW1 = x.T @ d_h1_pre / bs
    db1 = d_h1_pre.mean(axis=0)

    d_input = d_h1_pre @ W1.T
    return [dW1, db1, dW2, db2, dW3, db3], d_input


def _generator_backward(params, cache, d_out):
    """Backprop through generator. d_out = dL/d(tanh_output)."""
    W1, b1, W2, b2, W3, b3 = params
    z, h1_pre, h1, h2_pre, h2, out_pre, out = cache
    bs = z.shape[0]

    d_out_pre = d_out * _tanh_grad(out)
    dW3 = h2.T @ d_out_pre / bs
    db3 = d_out_pre.mean(axis=0)

    d_h2 = d_out_pre @ W3.T
    d_h2_pre = d_h2 * _relu_grad(h2_pre)
    dW2 = h1.T @ d_h2_pre / bs
    db2 = d_h2_pre.mean(axis=0)

    d_h1 = d_h2_pre @ W2.T
    d_h1_pre = d_h1 * _relu_grad(h1_pre)
    dW1 = z.T @ d_h1_pre / bs
    db1 = d_h1_pre.mean(axis=0)

    return [dW1, db1, dW2, db2, dW3, db3]


# ── Adam optimizer ──────────────────────────────────────────────────

class _Adam:
    def __init__(self, params, lr=2e-4, beta1=0.5, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            updated.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return updated


# ── training ────────────────────────────────────────────────────────

def train_gan(
    prices: pd.Series,
    z_dim: int = 16,
    seq_len: int = 50,
    g_hidden: tuple = (64, 128),
    d_hidden: tuple = (128, 64),
    batch_size: int = 64,
    epochs: int = 500,
    lr: float = 2e-4,
    n_gen_paths: int = 30,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a GAN to generate synthetic return sequences.

    Parameters
    ----------
    prices : pd.Series
        Historical prices. Log returns are computed internally.
    z_dim : int
        Dimension of the latent noise vector.
    seq_len : int
        Length of generated return sequences.
    g_hidden, d_hidden : tuple
        Hidden layer sizes for G and D.
    batch_size : int
        Mini-batch size.
    epochs : int
        Training epochs. Default 500 (lower than reference's 3000
        because we use analytic gradients instead of autograd).
    lr : float
        Learning rate for Adam.
    n_gen_paths : int
        Number of synthetic paths to generate after training.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        available : bool
        generated_returns : np.ndarray  shape (n_gen_paths, seq_len)
        generated_paths : np.ndarray    shape (n_gen_paths, seq_len + 1)
        real_returns : np.ndarray
        g_losses, d_losses : list
        ret_mean, ret_std : float  (normalization params)
        distribution_comparison : dict  (mean, std, skew, kurtosis for real vs gen)
    """
    prices = _clean_prices(prices)
    if len(prices) < seq_len + 30:
        return {
            "available": False,
            "reason": f"Need >= {seq_len + 30} bars; got {len(prices)}.",
        }

    # Compute log returns
    price_arr = prices.values.astype(np.float64)
    log_returns = np.diff(np.log(price_arr + 1e-9))
    log_returns = log_returns[~np.isnan(log_returns)]

    ret_mean = float(log_returns.mean())
    ret_std = float(log_returns.std()) + 1e-9

    # Normalize to [-1, 1] for tanh output
    norm_returns = (log_returns - ret_mean) / (3.0 * ret_std)
    norm_returns = np.clip(norm_returns, -1.0, 1.0)

    # Build training windows
    n_windows = len(norm_returns) - seq_len + 1
    if n_windows < batch_size:
        norm_returns = np.tile(norm_returns, 5)
        n_windows = len(norm_returns) - seq_len + 1

    windows = np.array([norm_returns[i:i+seq_len] for i in range(n_windows)])

    # Initialize weights (He init)
    rng = np.random.default_rng(random_state)

    def he(fan_in, fan_out):
        return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)

    g_params = [
        he(z_dim, g_hidden[0]), np.zeros(g_hidden[0]),
        he(g_hidden[0], g_hidden[1]), np.zeros(g_hidden[1]),
        rng.standard_normal((g_hidden[1], seq_len)) * np.sqrt(1.0 / g_hidden[1]),
        np.zeros(seq_len),
    ]
    d_params = [
        he(seq_len, d_hidden[0]), np.zeros(d_hidden[0]),
        he(d_hidden[0], d_hidden[1]), np.zeros(d_hidden[1]),
        rng.standard_normal((d_hidden[1], 1)) * np.sqrt(1.0 / d_hidden[1]),
        np.zeros(1),
    ]

    g_opt = _Adam(g_params, lr=lr, beta1=0.5)
    d_opt = _Adam(d_params, lr=lr, beta1=0.5)

    g_losses, d_losses = [], []
    eps = 1e-8

    for epoch in range(epochs):
        # Sample mini-batch
        idx = rng.choice(len(windows), batch_size, replace=True)
        real_batch = windows[idx]
        noise = rng.standard_normal((batch_size, z_dim))

        # ── Update Discriminator ──
        fake, g_cache = _generator_forward(g_params, noise)
        d_real_out, d_real_cache = _discriminator_forward(d_params, real_batch)
        d_fake_out, d_fake_cache = _discriminator_forward(d_params, fake)

        # dL/d(D_out) for real: -1/D(x)
        d_real_grad = -1.0 / (d_real_out + eps)
        # dL/d(D_out) for fake: 1/(1-D(G(z)))
        d_fake_grad = 1.0 / (1.0 - d_fake_out + eps)

        d_grads_real, _ = _discriminator_backward(d_params, d_real_cache, d_real_grad)
        d_grads_fake, _ = _discriminator_backward(d_params, d_fake_cache, d_fake_grad)

        d_grads = [gr + gf for gr, gf in zip(d_grads_real, d_grads_fake)]
        d_params = d_opt.step(d_params, d_grads)

        # ── Update Generator ──
        noise2 = rng.standard_normal((batch_size, z_dim))
        fake2, g_cache2 = _generator_forward(g_params, noise2)
        d_fake2_out, d_fake2_cache = _discriminator_forward(d_params, fake2)

        # G loss: -log(D(G(z)))  →  dL/d(D_out) = -1/D(G(z))
        d_g_grad = -1.0 / (d_fake2_out + eps)
        _, d_input_grad = _discriminator_backward(d_params, d_fake2_cache, d_g_grad)
        # d_input_grad is dL/d(fake2), pass through G
        g_grads = _generator_backward(g_params, g_cache2, d_input_grad)
        g_params = g_opt.step(g_params, g_grads)

        # Record losses
        dl = float(-np.mean(np.log(d_real_out + eps) + np.log(1 - d_fake_out + eps)))
        gl = float(-np.mean(np.log(d_fake2_out + eps)))
        d_losses.append(dl)
        g_losses.append(gl)

    # ── Generate final samples ──
    z_final = rng.standard_normal((n_gen_paths, z_dim))
    gen_norm, _ = _generator_forward(g_params, z_final)
    gen_returns = gen_norm * (3.0 * ret_std) + ret_mean

    # Convert to price paths (rebased to 100)
    gen_paths = 100.0 * np.exp(
        np.hstack([np.zeros((n_gen_paths, 1)),
                   np.cumsum(gen_returns, axis=1)])
    )

    # Distribution comparison
    gen_flat = gen_returns.flatten()
    from scipy import stats as _st  # lazy import, optional
    try:
        real_skew = float(_st.skew(log_returns))
        real_kurt = float(_st.kurtosis(log_returns))
        gen_skew = float(_st.skew(gen_flat))
        gen_kurt = float(_st.kurtosis(gen_flat))
    except Exception:
        real_skew = real_kurt = gen_skew = gen_kurt = float("nan")

    return {
        "available": True,
        "generated_returns": gen_returns,
        "generated_paths": gen_paths,
        "real_returns": log_returns,
        "real_prices": price_arr,
        "g_losses": g_losses,
        "d_losses": d_losses,
        "ret_mean": ret_mean,
        "ret_std": ret_std,
        "params": {
            "z_dim": z_dim,
            "seq_len": seq_len,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "distribution_comparison": {
            "real": {"mean": ret_mean, "std": ret_std,
                     "skew": real_skew, "kurtosis": real_kurt},
            "generated": {"mean": float(gen_flat.mean()),
                          "std": float(gen_flat.std()),
                          "skew": gen_skew, "kurtosis": gen_kurt},
        },
    }
