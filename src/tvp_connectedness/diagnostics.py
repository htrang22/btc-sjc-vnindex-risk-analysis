from __future__ import annotations

import numpy as np
import pandas as pd

from .model import TVPVARResult


def stability_table(model: TVPVARResult) -> pd.DataFrame:
    radius = np.array(
        [np.max(np.abs(np.linalg.eigvals(matrix))) for matrix in model.coefficients]
    )
    return pd.DataFrame(
        {"spectral_radius": radius, "stable": radius < 1},
        index=model.dates,
    )


def unstable_episodes(table: pd.DataFrame) -> pd.DataFrame:
    unstable = table.loc[~table["stable"]].copy()
    if unstable.empty:
        return pd.DataFrame(columns=["start", "end", "duration", "max_radius"])

    positions = np.flatnonzero(~table["stable"].to_numpy())
    unstable["position"] = positions
    unstable["episode"] = unstable["position"].diff().ne(1).cumsum()

    records = []
    for _, group in unstable.groupby("episode"):
        records.append(
            {
                "start": group.index[0],
                "end": group.index[-1],
                "duration": len(group),
                "max_radius": group["spectral_radius"].max(),
            }
        )
    return pd.DataFrame(records)


def instability_context(
    table: pd.DataFrame,
    tci: pd.Series,
    net: pd.DataFrame,
) -> pd.DataFrame:
    """Combine unstable dates with connectedness measures for interpretation."""
    if not table.index.equals(tci.index) or not table.index.equals(net.index):
        raise ValueError("stability and connectedness indices must match")
    context = table.loc[~table["stable"]].copy()
    context["TCI"] = tci.loc[context.index]
    for variable in net.columns:
        context[f"NET_{variable}"] = net.loc[context.index, variable]
    return context
