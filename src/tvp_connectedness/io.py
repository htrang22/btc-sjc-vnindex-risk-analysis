from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .diagnostics import instability_context
from .pipeline import SpecificationResult, compare_specifications


VARIABLES = ("BTC", "SJC", "VNINDEX")


def load_volatility_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["Date"])
    data = (
        data.sort_values("Date")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .set_index("Date")
    )
    if tuple(data.columns) != VARIABLES:
        raise ValueError(f"expected columns {VARIABLES}, got {tuple(data.columns)}")
    if not data.index.is_monotonic_increasing:
        raise ValueError("dates must be monotonically increasing")
    return data.astype(float)


def save_results(result: SpecificationResult, directory: str | Path) -> None:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    result.connectedness.tci.to_csv(output / "tci.csv", header=True)
    result.connectedness.from_others.to_csv(output / "from.csv")
    result.connectedness.to_others.to_csv(output / "to.csv")
    result.connectedness.net.to_csv(output / "net.csv")
    result.stability.to_csv(output / "stability.csv")
    result.episodes.to_csv(output / "unstable_episodes.csv", index=False)
    instability_context(
        result.stability,
        result.connectedness.tci,
        result.connectedness.net,
    ).to_csv(output / "unstable_observations.csv")
    result.summary().to_frame("value").to_csv(output / "summary.csv")


def save_comparison(
    results: list[SpecificationResult], path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    compare_specifications(results).to_csv(output)
