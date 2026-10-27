from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConnectednessResult:
    tci: pd.Series
    from_others: pd.DataFrame
    to_others: pd.DataFrame
    net: pd.DataFrame
    theta: np.ndarray

    @property
    def final_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "FROM": self.from_others.iloc[-1],
                "TO": self.to_others.iloc[-1],
                "NET": self.net.iloc[-1],
            }
        )


def calculate_connectedness(
    theta: np.ndarray,
    *,
    dates: pd.Index,
    variables: tuple[str, ...],
) -> ConnectednessResult:
    if theta.ndim != 3 or theta.shape[1] != theta.shape[2]:
        raise ValueError("theta must have shape (time, variables, variables)")
    if theta.shape[0] != len(dates) or theta.shape[1] != len(variables):
        raise ValueError("theta dimensions do not match dates and variables")
    if not np.isfinite(theta).all():
        raise ValueError("theta contains non-finite values")
    if not np.allclose(theta.sum(axis=2), 1, atol=1e-10):
        raise ValueError("theta rows must sum to one")

    diagonal = np.diagonal(theta, axis1=1, axis2=2)
    from_others = (theta.sum(axis=2) - diagonal) * 100
    to_others = (theta.sum(axis=1) - diagonal) * 100
    net = to_others - from_others
    tci = from_others.sum(axis=1) / theta.shape[1]

    return ConnectednessResult(
        tci=pd.Series(tci, index=dates, name="TCI"),
        from_others=pd.DataFrame(from_others, index=dates, columns=variables),
        to_others=pd.DataFrame(to_others, index=dates, columns=variables),
        net=pd.DataFrame(net, index=dates, columns=variables),
        theta=theta,
    )
