from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .connectedness import net_pairwise
from .pipeline import SpecificationResult


VOLATILITY_TITLES = {
    "BTC": "BTC-vnd DY Volatility",
    "SJC": "Gold SJC DY Volatility",
    "VNINDEX": "VN-index DY Volatility",
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_volatility(data: pd.DataFrame, directory: Path) -> list[Path]:
    paths = []
    for asset in data.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data.index, data[asset])
        ax.set(title=VOLATILITY_TITLES.get(asset, asset), xlabel="Date")
        ax.grid(alpha=0.3)
        paths.append(_save(fig, directory / f"volatility_{asset.lower()}.png"))
    return paths


def plot_stability(result: SpecificationResult, directory: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        result.stability.index,
        result.stability["spectral_radius"],
        label="Spectral Radius",
    )
    ax.axhline(1, linestyle="--", color="tab:red", label="Stability Boundary")
    ax.set(
        title="TVP-VAR(1) Stability Over Time",
        xlabel="Date",
        ylabel="Spectral Radius",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, directory / "stability_spectral_radius.png")


def plot_tci(result: SpecificationResult, directory: Path) -> Path:
    tci = result.connectedness.tci
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tci.index, tci, linewidth=1.5)
    ax.set(
        title=f"TVP-VAR Dynamic Total Connectedness Index (H = {result.horizon})",
        xlabel="Date",
        ylabel="TCI (%)",
    )
    ax.grid(alpha=0.3)
    return _save(fig, directory / "tci.png")


def plot_net(result: SpecificationResult, directory: Path) -> list[Path]:
    net = result.connectedness.net
    horizon = result.horizon

    fig, ax = plt.subplots(figsize=(12, 6))
    for asset in net.columns:
        ax.plot(net.index, net[asset], linewidth=1.3, label=asset)
    # NET > 0: net transmitter, NET < 0: net receiver
    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.set(
        title=f"TVP-VAR Dynamic Net Connectedness (H = {horizon})",
        xlabel="Date",
        ylabel="NET Connectedness (%)",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    paths = [_save(fig, directory / "net_all.png")]

    for asset in net.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(net.index, net[asset], linewidth=1.4)
        ax.axhline(0, linestyle="--", linewidth=1, color="black")
        ax.set(
            title=f"Dynamic Net Connectedness - {asset} (H = {horizon})",
            xlabel="Date",
            ylabel="NET (%)",
        )
        ax.grid(alpha=0.3)
        paths.append(_save(fig, directory / f"net_{asset.lower()}.png"))
    return paths


def plot_sensitivity(
    results: dict[str, SpecificationResult],
    directory: Path,
    *,
    baseline: str = "baseline_0999",
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for key in ["horizon_05", baseline, "horizon_20"]:
        result = results[key]
        axes[0].plot(result.connectedness.tci, label=f"H={result.horizon}")
    axes[0].set(title="Horizon sensitivity", ylabel="TCI (%)")
    axes[0].legend()

    for key in ["kappa_094", baseline, "kappa_098"]:
        result = results[key]
        axes[1].plot(
            result.connectedness.tci, label=f"κσ={result.model.kappa_sigma}"
        )
    axes[1].set(title="Covariance forgetting-factor sensitivity", ylabel="TCI (%)")
    axes[1].legend()

    reference = results[baseline]
    for variable in reference.model.variables:
        axes[2].plot(reference.connectedness.net[variable], label=variable)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set(title="Baseline net connectedness", ylabel="NET (%)")
    axes[2].legend()
    return _save(fig, directory / "sensitivity.png")


def plot_net_pairwise(result: SpecificationResult, directory: Path) -> Path:
    pairwise = net_pairwise(result.connectedness)
    fig, axes = plt.subplots(len(pairwise.columns), 1, figsize=(12, 9), sharex=True)
    for ax, column in zip(axes, pairwise.columns):
        ax.plot(pairwise.index, pairwise[column], linewidth=1.2)
        ax.axhline(0, linestyle="--", linewidth=1, color="black")
        ax.set(title=f"Net pairwise connectedness {column} (H = {result.horizon})",
               ylabel="NPDC (%)")
        ax.grid(alpha=0.3)
    return _save(fig, directory / "net_pairwise.png")
