import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rate_per_step(annual_rate: float, steps_per_year: int) -> float:
    return (1.0 + float(annual_rate))**(1.0/steps_per_year) - 1.0

def expit(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def logit(x: float, eps: float = 1e-9) -> float:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))

def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.allclose(x.sum(), 0.0): return np.nan
    x = np.sort(x); n = len(x); cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

def clip01(x: float, eps: float = 0.0) -> float:
    return float(np.clip(x, 0.0 + eps, 1.0 - eps if eps else 1.0))

def mean_or0(seq) -> float:
    return float(np.mean(seq)) if seq else 0.0

def moving_avg(s: pd.Series, w=5) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()

def legend_merge(ax, ax2, loc="upper left", fontsize=8):
    h1, n1 = ax.get_legend_handles_labels()
    h2, n2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, n1 + n2, loc=loc, fontsize=fontsize)

def get_map(maybe_map, key, default):
    return (maybe_map or {}).get(key, default)

def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "axes.titlesize": 12, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 1.75, "figure.autolayout": True,
    })

def soft_min(x: float, y: float, smooth: float = 1.0) -> float:
    """
    Smooth minimum function with overflow protection.
    S(x,y) = (x + y - sqrt((x-y)^2 + eps)) / 2
    """
    diff = x - y
    # Prevent overflow in square if numbers are astronomical
    if abs(diff) > 1e9:
        return min(x, y)
    
    val = 0.5 * (x + y - np.sqrt(diff**2 + smooth))
    # [FIX] Numerical stability: Capability/Constraints cannot be negative
    return max(0.0, float(val))