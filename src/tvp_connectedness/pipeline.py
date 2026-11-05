from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .connectedness import ConnectednessResult, calculate_connectedness
from .diagnostics import stability_table, unstable_episodes
from .gfevd import generalized_fevd
from .model import TVPVARResult, estimate_tvp_var


@dataclass(frozen=True)
class SpecificationResult:
    name: str
    horizon: int
    model: TVPVARResult
    stability: pd.DataFrame
    episodes: pd.DataFrame
    connectedness: ConnectednessResult

    def summary(self) -> pd.Series:
        result = {
            "lambda_beta": self.model.lambda_beta,
            "kappa_sigma": self.model.kappa_sigma,
            "horizon": self.horizon,
            "stable_share": self.stability["stable"].mean(),
            "unstable_periods": int((~self.stability["stable"]).sum()),
            "max_spectral_radius": self.stability["spectral_radius"].max(),
            "mean_tci": self.connectedness.tci.mean(),
            "final_tci": self.connectedness.tci.iloc[-1],
        }
        result.update(
            {
                f"mean_net_{variable}": value
                for variable, value in self.connectedness.net.mean().items()
            }
        )
        result.update(
            {
                f"final_net_{variable}": value
                for variable, value in self.connectedness.net.iloc[-1].items()
            }
        )
        return pd.Series(result, name=self.name)


def run_specification(
    data: pd.DataFrame,
    *,
    name: str,
    lambda_beta: float,
    kappa_sigma: float = 0.96,
    initial_covariance: float = 0.01,
    horizon: int = 10,
) -> SpecificationResult:
    model = estimate_tvp_var(
        data,
        lambda_beta=lambda_beta,
        kappa_sigma=kappa_sigma,
        initial_covariance=initial_covariance,
    )
    stability = stability_table(model)
    theta = generalized_fevd(model, horizon=horizon)
    connectedness = calculate_connectedness(
        theta, dates=model.dates, variables=model.variables
    )
    return SpecificationResult(
        name=name,
        horizon=horizon,
        model=model,
        stability=stability,
        episodes=unstable_episodes(stability),
        connectedness=connectedness,
    )


def compare_specifications(
    specifications: list[SpecificationResult],
) -> pd.DataFrame:
    if not specifications:
        raise ValueError("at least one specification is required")
    return pd.DataFrame([result.summary() for result in specifications])
