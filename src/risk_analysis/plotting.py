from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def line_plot(
    data: pd.Series | pd.DataFrame,
    path: Path,
    *,
    title: str,
    ylabel: str = "",
    zero_line: bool = False,
    figsize: tuple[float, float] = (12, 5),
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    frame = data.to_frame() if isinstance(data, pd.Series) else data
    for column in frame.columns:
        ax.plot(frame.index, frame[column], linewidth=1.1, label=str(column))
    if zero_line:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    if frame.shape[1] > 1:
        ax.legend()
    ax.set(title=title, xlabel="Date", ylabel=ylabel)
    ax.grid(alpha=0.3)
    return save_figure(fig, path)


def acf_pacf_plot(
    series: pd.Series, path: Path, *, title: str, lags: int = 30
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    x = series.dropna().to_numpy()
    plot_acf(x, lags=lags, zero=False, ax=axes[0])
    plot_pacf(x, lags=lags, zero=False, method="ywm", ax=axes[1])
    axes[0].set_title(f"ACF - {title}")
    axes[1].set_title(f"PACF - {title}")
    return save_figure(fig, path)
