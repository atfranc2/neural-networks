import os
from typing import Iterable, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Compute the normal distribution PDF for given mean and std."""
    sigma = float(sigma)
    if sigma <= 0:
        # Avoid division by zero; return zeros to keep plot stable
        return np.zeros_like(x)
    coeff = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    return coeff * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def save_normal_variants_figure(
    out_path: str = "3_makemore_v3_batchnorm/figures/normal_variants.png",
    shift: float = 5,
    scale: float = 3,
    title: Optional[str] = None,
) -> str:
    """
    Build a single figure with 4 normal distributions derived from the same dataset.
    If `data` is None, a dataset is generated internally (self-contained) using
    a reproducible normal draw.

    The 4 scenarios are:
      - Original (normal fit to data)
      - Shifted (add constant)
      - Scaled (multiply constant)
      - Shifted + Scaled

    Adds legend entries with mean and std for each case and saves to `out_path`.

    Returns the saved file path.
    """
    # Generate dataset internally if none provided (self-contained behavior)
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0, scale=1, size=20_000)
    data = np.asarray(list(data), dtype=float).ravel()

    mu = float(np.mean(data))
    sigma = float(np.std(data))

    # Define transformed means/stds according to operations on data
    # Shift: x + c  => mean+ c, std unchanged
    # Scale: a * x => mean * a, std * a
    mu_shift = mu + shift
    sigma_shift = sigma

    mu_scale = mu * scale
    sigma_scale = sigma * scale

    mu_shift_scale = mu * scale + shift
    sigma_shift_scale = sigma * scale

    # Build x-range that covers all four distributions robustly
    mus = np.array([mu, mu_shift, mu_scale, mu_shift_scale], dtype=float)
    sigmas = np.array([sigma, sigma_shift, sigma_scale, sigma_shift_scale], dtype=float)
    # Use a generous span (±4 sigma of the largest sigma) centered around all means
    sigma_max = float(np.max(sigmas)) if np.all(sigmas > 0) else (sigma if sigma > 0 else 1.0)
    x_min = float(np.min(mus)) - 4.0 * sigma_max
    x_max = float(np.max(mus)) + 4.0 * sigma_max + 5
    x = np.linspace(x_min, x_max, 800)

    curves: list[Tuple[str, str, float, float]] = [
        ("Original", "tab:blue", mu, sigma),
        (f"Shifted (+{shift:g})", "tab:orange", mu_shift, sigma_shift),
        (f"Scaled (×{scale:g})", "tab:green", mu_scale, sigma_scale),
        (f"Shifted+Scaled (+{shift:g}, ×{scale:g})", "tab:red", mu_shift_scale, sigma_shift_scale),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color, m, s in curves:
        y = normal_pdf(x, m, s)
        ax.plot(x, y, color=color, lw=2, label=f"{label}: μ={m:.3f}, σ={s:.3f}")

    ax.set_xlabel("value")
    ax.set_ylabel("density")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Normal distributions derived from the same dataset")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


# python /app/3_makemore_v3_batchnorm/plots.py
save_normal_variants_figure()
