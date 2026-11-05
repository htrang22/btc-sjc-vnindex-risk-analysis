from __future__ import annotations

from pathlib import Path

import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show(table: pd.DataFrame | pd.Series, *, digits: int = 4) -> None:
    print(table.round(digits).to_string())


def save_table(
    table: pd.DataFrame | pd.Series,
    directory: Path,
    name: str,
    *,
    index: bool = True,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    table.to_csv(path, index=index)
    return path
