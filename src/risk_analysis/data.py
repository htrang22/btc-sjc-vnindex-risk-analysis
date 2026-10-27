from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .config import ASSETS, RETURN_FILES, ROOT


def load_asset(asset: str) -> pd.DataFrame:
    """Price and return of one asset on its own trading calendar, indexed by Date."""
    file, price_column, return_column = RETURN_FILES[asset]
    data = pd.read_csv(ROOT / file, parse_dates=["Date"])
    data = data.sort_values("Date").set_index("Date")
    return data[[price_column, return_column]].rename(
        columns={price_column: "price", return_column: "return"}
    )


def load_assets() -> dict[str, pd.DataFrame]:
    return {asset: load_asset(asset) for asset in ASSETS}


def synchronized_returns(assets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Percentage log returns computed on the dates common to all assets."""
    prices = pd.concat(
        {asset: frame["price"] for asset, frame in assets.items()},
        axis=1,
        join="inner",
    ).sort_index()
    returns = 100 * np.log(prices / prices.shift(1))
    return returns.dropna()


def describe_returns(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Paper Table 4.1 layout; kurtosis is excess kurtosis."""
    rows = {}
    for asset, values in series.items():
        x = values.dropna()
        rows[asset] = {
            "N": len(x),
            "Mean": x.mean(),
            "Std": x.std(),
            "Min": x.min(),
            "25%": x.quantile(0.25),
            "Median": x.median(),
            "75%": x.quantile(0.75),
            "Max": x.max(),
            "Skewness": x.skew(),
            "Excess kurtosis": x.kurt(),
        }
    return pd.DataFrame(rows).T


def describe_volatility(data: pd.DataFrame) -> pd.DataFrame:
    """Paper Table 4.2 layout; kurtosis is Pearson (non-excess) kurtosis."""
    table = pd.DataFrame(
        {
            "N": data.count(),
            "Mean": data.mean(),
            "Std": data.std(),
            "Min": data.min(),
            "25%": data.quantile(0.25),
            "Median": data.median(),
            "75%": data.quantile(0.75),
            "Max": data.max(),
            "Skewness": data.apply(lambda x: skew(x.dropna(), bias=False)),
            "Kurtosis": data.apply(
                lambda x: kurtosis(x.dropna(), fisher=False, bias=False)
            ),
        }
    )
    table.index.name = "Asset"
    return table
