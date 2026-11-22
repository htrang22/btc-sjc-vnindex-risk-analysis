"""Step 4 - walk-forward volatility forecasts and 1-day VaR/ES backtests.

Uses each asset's own-calendar returns and the Model 1 ARMA-EGARCH-t
specification. Parameters are re-estimated on an expanding window every
``FORECAST_REFIT_EVERY`` observations; every forecast uses data up to the
previous day only.

Part A compares 1-day and 5-day variance forecasts of EGARCH-t with GARCH-t,
GJR-t, EWMA and a rolling variance (QLIKE, MSE, MAE, Mincer-Zarnowitz,
Diebold-Mariano). Part B turns the EGARCH-t forecast into 95% and 99% VaR/ES
(parametric Student-t and filtered historical simulation) and backtests them.

Part C repeats A and B on a true holdout: returns after the original sample end
(2026-08-01), which played no part in choosing the model specification. The
walk-forward simply continues into the new data with the same procedure.
Part C runs only when the holdout-*.csv files exist (see DATA_PROVENANCE.md).

Usage (from the repository root):

    python step4_forecast_var.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from risk_analysis import config  # noqa: E402
from risk_analysis.backtesting import (  # noqa: E402
    diebold_mariano,
    loss_table,
    mincer_zarnowitz,
    var_es_backtest,
)
from risk_analysis.data import holdout_available, load_assets, load_extended_assets  # noqa: E402
from risk_analysis.forecasting import VOL_MODELS, var_es_table, walk_forward  # noqa: E402
from risk_analysis.plotting import save_figure  # noqa: E402
from risk_analysis.reporting import save_table, section, show  # noqa: E402

TABLES = config.OUTPUTS / config.STEP4_DIR
FIGURES = config.FIGURES / config.STEP4_DIR
LABELS = config.ASSET_LABELS
REFERENCE = "EGARCH-t"
H = max(config.FORECAST_HORIZONS)


def plot_forecasts(asset: str, forecasts: pd.DataFrame, figures: Path, note: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecasts.index, np.abs(forecasts["resid"]), color="0.75", linewidth=0.8,
            label="|ARMA residual|")
    for model in ("EGARCH-t", "GJR-t", "EWMA"):
        ax.plot(forecasts.index, np.sqrt(forecasts[f"var1_{model}"]), linewidth=1.1,
                label=f"{model} 1-day forecast")
    ax.set(title=f"{LABELS[asset]} - {note} 1-day volatility forecasts",
           xlabel="Date", ylabel="Volatility (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, figures / f"volatility_forecast_{asset.lower()}.png")


def plot_var(asset: str, table: pd.DataFrame, figures: Path, note: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(table.index, table["loss"], color="0.7", width=1.0, label="Daily loss (-return)")
    styles = {"VaR95_t": ("tab:blue", "95% VaR (t)"), "VaR99_t": ("tab:red", "99% VaR (t)"),
              "VaR99_FHS": ("tab:orange", "99% VaR (FHS)")}
    for column, (color, label) in styles.items():
        ax.plot(table.index, table[column], color=color, linewidth=1.1, label=label)
    for column, color, label in (("VaR95_t", "tab:blue", "95% VaR (t) violation"),
                                 ("VaR99_t", "tab:red", "99% VaR (t) violation")):
        hits = table["loss"] > table[column]
        ax.scatter(table.index[hits], table["loss"][hits], color=color, zorder=3, s=18,
                   label=label)
    ax.set(title=f"{LABELS[asset]} - {note} 1-day VaR backtest", xlabel="Date",
           ylabel="Loss (%)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    save_figure(fig, figures / f"var_backtest_{asset.lower()}.png")


def evaluate(assets: dict[str, pd.Series], *, tables: Path, figures: Path, part: str,
             note: str, after: str | None = None) -> None:
    """Walk-forward forecasts, forecast evaluation and VaR/ES backtests.

    With ``after``, forecasts are produced over the whole series but evaluated
    only on dates after it.
    """
    losses, mz, dm, backtests, forecast_tables, parameters = [], [], [], [], [], []

    for asset in config.ASSETS:
        section(f"STEP 4{part} - {LABELS[asset]}: WALK-FORWARD FORECASTS ({note})")
        result = walk_forward(
            assets[asset],
            asset=asset,
            arma_order=config.MODEL1_ARMA[asset],
            egarch_order=config.MODEL1_EGARCH[asset],
            initial=config.FORECAST_INITIAL_WINDOW,
            refit_every=config.FORECAST_REFIT_EVERY,
            horizon=H,
            simulations=config.FORECAST_SIMULATIONS,
            seed=config.FORECAST_SEED,
            ewma_lambda=config.EWMA_LAMBDA,
            rolling_window=config.ROLLING_WINDOW,
        )
        parameters.append(result.parameters)
        if after is not None:
            result = result.after(after)
        f = result.forecasts
        blocks = f["block"].nunique()
        print(f"Evaluated: {len(f)} days, {f.index[0].date()} -> {f.index[-1].date()} "
              f"({blocks} estimation windows)")

        for horizon in config.FORECAST_HORIZONS:
            losses.append(loss_table(result, horizon, VOL_MODELS))
            mz.append(mincer_zarnowitz(result, horizon, VOL_MODELS))
            for loss in ("QLIKE", "MSE"):
                dm.append(diebold_mariano(result, horizon, VOL_MODELS,
                                          reference=REFERENCE, loss=loss))

        table = var_es_table(result, config.VAR_LEVELS)
        backtests.append(var_es_backtest(result, table, config.VAR_LEVELS,
                                         simulations=config.ES_BACKTEST_SIMULATIONS,
                                         seed=config.FORECAST_SEED))
        # Saved like the committed conditional-volatility files: EGARCH sigma and
        # VaR/ES only. Returns, ARMA means and benchmark variances are left out,
        # because EWMA/rolling forecasts and MA means let the inputs be rebuilt.
        forecast_tables.append(
            table.drop(columns=["loss", "mean", "block"]).assign(Asset=asset)
        )
        plot_forecasts(asset, f, figures, note)
        plot_var(asset, table, figures, note)

    section(f"STEP 4{part} - VOLATILITY FORECAST EVALUATION ({note})")
    parameters = pd.concat(parameters, ignore_index=True)
    print("EGARCH-t parameters across re-estimations (min / max):")
    columns = ["omega", "alpha[1]", "gamma[1]", "beta sum", "nu"]
    show(parameters.groupby("Asset")[columns].agg(["min", "max"]))
    save_table(parameters, tables, "egarch_parameters_by_window", index=False)

    losses = pd.concat(losses, ignore_index=True)
    show(losses.set_index(["Asset", "Horizon", "Model"]))
    save_table(losses, tables, "forecast_losses", index=False)

    mz = pd.concat(mz, ignore_index=True)
    print("\nMincer-Zarnowitz (proxy = a + b * forecast; H0: a = 0, b = 1):")
    show(mz.set_index(["Asset", "Horizon", "Model"]))
    save_table(mz, tables, "mincer_zarnowitz", index=False)

    dm = pd.concat(dm, ignore_index=True)
    print(f"\nDiebold-Mariano vs {REFERENCE} (DM > 0: {REFERENCE} more accurate):")
    show(dm.set_index(["Asset", "Horizon", "Loss", "Benchmark"]).drop(columns="Reference"))
    save_table(dm, tables, "diebold_mariano", index=False)

    section(f"STEP 4{part} - 1-DAY VaR / ES BACKTESTS, EGARCH-t ({note})")
    backtests = pd.concat(backtests, ignore_index=True)
    show(backtests.set_index(["Asset", "Method", "Level"]).T)
    save_table(backtests, tables, "var_es_backtest", index=False)

    forecasts = pd.concat(forecast_tables)
    forecasts.index.name = "Date"
    save_table(forecasts, tables, "daily_var_es")


def main() -> None:
    evaluate(load_assets(), tables=TABLES, figures=FIGURES, part="A/B",
             note="out-of-sample")

    if holdout_available():
        evaluate(load_extended_assets(), tables=TABLES / "holdout",
                 figures=FIGURES / "holdout", part="C", note="holdout",
                 after=config.ORIGINAL_SAMPLE_END)
    else:
        section("STEP 4C - HOLDOUT SKIPPED")
        print("holdout-*.csv not found; see DATA_PROVENANCE.md to build them.")

    section("STEP 4 SAVED")
    print("Tables :", TABLES)
    print("Figures:", FIGURES)


if __name__ == "__main__":
    main()
