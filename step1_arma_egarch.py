"""Step 1 - ARMA-EGARCH volatility of BTC, SJC gold and VN-Index.

Part A (Model 1) estimates each asset on its own trading calendar
(paper sections 4.1 and 4.2.1-4.2.4). Part B re-filters the returns on the
dates common to all three assets; its standardized residuals feed Step 2
and its conditional volatility feeds Step 3.

Usage (from the repository root):

    python step1_arma_egarch.py            # full run with model searches
    python step1_arma_egarch.py --quick    # skip ARMA/EGARCH candidate searches
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from risk_analysis import config  # noqa: E402
from risk_analysis.arma_egarch import (  # noqa: E402
    MarginalModel,
    arch_lm_table,
    arma_grid,
    egarch_candidates,
    fit_marginal,
    ljung_box_table,
    parameter_table,
    standardized_diagnostics,
)
from risk_analysis.data import (  # noqa: E402
    describe_returns,
    describe_volatility,
    load_assets,
    synchronized_returns,
)
from risk_analysis.plotting import acf_pacf_plot, line_plot  # noqa: E402
from risk_analysis.reporting import save_table, section, show  # noqa: E402
from risk_analysis.stationarity import unit_root_table, zivot_andrews_table  # noqa: E402

TABLES = config.OUTPUTS / config.STEP1_DIR
FIGURES = config.FIGURES / config.STEP1_DIR
LABELS = config.ASSET_LABELS


def report_models(
    models: dict[str, MarginalModel], prefix: str, *, with_candidates: bool
) -> None:
    """Save parameters, residual tests and diagnostics of a set of marginal models."""
    if with_candidates:
        candidate_tables = []
        for asset, model in models.items():
            table = egarch_candidates(model.arma_residuals, config.EGARCH_CANDIDATES)
            candidate_tables.append(table.assign(Asset=asset))
        candidates = pd.concat(candidate_tables, ignore_index=True)
        print("\nEGARCH candidates (Pass_All = converged + LB(z) + ARCH-LM):")
        show(candidates[["Asset", "Model", "AIC", "BIC", "Pass_LB_z", "Pass_ARCH", "Pass_All"]])
        save_table(candidates, TABLES, f"{prefix}_egarch_candidates", index=False)

    lb_tables, arch_tables, params, diagnostics = [], [], [], []
    for asset, model in models.items():
        p, q = model.arma_order
        lb_tables.append(
            ljung_box_table(model.arma_residuals, model_df=p + q).assign(Asset=asset)
        )
        arch_tables.append(arch_lm_table(model.arma_residuals).assign(Asset=asset))
        params.append(
            pd.concat(
                {
                    "ARMA": parameter_table(model.arma_fit),
                    "EGARCH": parameter_table(model.egarch_fit),
                }
            ).assign(Asset=asset)
        )
        diagnostics.append(
            {"Asset": asset, "Model": model.label,
             **standardized_diagnostics(model.std_residuals)}
        )
        acf_pacf_plot(
            model.arma_residuals,
            FIGURES / f"{prefix}_acf_pacf_arma_residuals_{asset.lower()}.png",
            title=f"{LABELS[asset]} ARMA{model.arma_order} residuals",
        )

    save_table(pd.concat(lb_tables), TABLES, f"{prefix}_arma_ljung_box")
    arch = pd.concat(arch_tables)
    print("\nARCH-LM on ARMA residuals (H0: no ARCH effect):")
    show(arch)
    save_table(arch, TABLES, f"{prefix}_arch_lm")

    parameters = pd.concat(params)
    parameters.index.names = ["Equation", "Parameter"]
    save_table(parameters, TABLES, f"{prefix}_parameters")

    summary = pd.DataFrame([model.summary() for model in models.values()])
    print("\nFinal models:")
    show(summary.set_index("Asset").T)
    save_table(summary, TABLES, f"{prefix}_model_summary", index=False)

    diagnostics = pd.DataFrame(diagnostics)
    print("\nStandardized-residual diagnostics:")
    show(diagnostics.set_index("Asset").T)
    save_table(diagnostics, TABLES, f"{prefix}_egarch_diagnostics", index=False)


def part_a_own_calendar(assets: dict[str, pd.DataFrame], quick: bool) -> None:
    section("STEP 1A - MODEL 1: OWN-CALENDAR RETURNS")
    returns = {asset: frame["return"].dropna() for asset, frame in assets.items()}

    for asset, frame in assets.items():
        line_plot(frame["price"], FIGURES / f"price_{asset.lower()}.png",
                  title=f"{LABELS[asset]} price")
        line_plot(returns[asset], FIGURES / f"return_{asset.lower()}.png",
                  title=f"{LABELS[asset]} return", ylabel="Return (%)", zero_line=True)
        acf_pacf_plot(returns[asset], FIGURES / f"acf_pacf_return_{asset.lower()}.png",
                      title=f"{LABELS[asset]} return", lags=12)

    descriptive = describe_returns(returns)
    print("Descriptive statistics of returns (Table 4.1):")
    show(descriptive)
    save_table(descriptive, TABLES, "model1_descriptive_returns")

    unit_root = unit_root_table(returns)
    za = zivot_andrews_table(returns)
    print("\nUnit-root tests:")
    show(unit_root.set_index(["Test", "Asset", "Case"]))
    print("\nZivot-Andrews:")
    show(za.set_index(["Asset", "Case"]))
    save_table(unit_root, TABLES, "model1_unit_root_tests", index=False)
    save_table(za, TABLES, "model1_zivot_andrews", index=False)

    if not quick:
        grids = []
        for asset, max_order in config.MODEL1_ARMA_SEARCH.items():
            grid = arma_grid(returns[asset], max_p=max_order, max_q=max_order)
            grids.append(grid.assign(Asset=asset))
            print(f"\n{asset} ARMA candidates passing Ljung-Box (top 5 by BIC):")
            show(grid[grid["Pass_LB"]].sort_values("BIC").head(5).set_index("Model"))
        save_table(pd.concat(grids, ignore_index=True), TABLES,
                   "model1_arma_candidates", index=False)

    models = {
        asset: fit_marginal(
            returns[asset],
            asset=asset,
            arma_order=config.MODEL1_ARMA[asset],
            egarch_order=config.MODEL1_EGARCH[asset],
        )
        for asset in config.ASSETS
    }
    report_models(models, "model1", with_candidates=not quick)

    for asset, model in models.items():
        line_plot(
            model.volatility,
            FIGURES / f"model1_conditional_volatility_{asset.lower()}.png",
            title=f"{LABELS[asset]} conditional volatility\n{model.label}",
            ylabel="Conditional volatility (%)",
        )
    volatility = pd.concat({a: m.volatility for a, m in models.items()}, axis=1)
    save_table(volatility, TABLES, "model1_conditional_volatility")


def part_b_synchronized(assets: dict[str, pd.DataFrame], quick: bool) -> None:
    section("STEP 1B - SYNCHRONIZED SAMPLE (INPUT FOR STEPS 2 AND 3)")
    returns = synchronized_returns(assets)
    print(f"Synchronized returns: {len(returns)} observations, "
          f"{returns.index[0].date()} -> {returns.index[-1].date()}")
    save_table(returns, TABLES, "synchronized_returns")
    series = {asset: returns[asset] for asset in config.ASSETS}

    descriptive = describe_returns(series)
    show(descriptive)
    save_table(descriptive, TABLES, "sync_descriptive_returns")
    unit_root = unit_root_table(series, adf_cases=("c",), pp_cases=("c",))
    save_table(unit_root, TABLES, "sync_unit_root_tests", index=False)

    if not quick:
        grids = []
        for asset in config.ASSETS:
            grid = arma_grid(
                series[asset],
                max_p=config.SYNC_ARMA_SEARCH,
                max_q=config.SYNC_ARMA_SEARCH,
                lb_lags=(15, 20),
            )
            grids.append(grid.assign(Asset=asset))
            print(f"\n{asset} ARMA candidates passing Ljung-Box (top 5 by AIC):")
            show(grid[grid["Pass_LB"]].head(5).set_index("Model"))
        save_table(pd.concat(grids, ignore_index=True), TABLES,
                   "sync_arma_candidates", index=False)

    models = {
        asset: fit_marginal(
            series[asset],
            asset=asset,
            arma_order=config.SYNC_ARMA[asset],
            egarch_order=config.SYNC_EGARCH[asset],
        )
        for asset in config.ASSETS
    }
    report_models(models, "sync", with_candidates=not quick)

    volatility = pd.concat({a: m.volatility for a, m in models.items()}, axis=1)
    std_residuals = pd.concat({a: m.std_residuals for a, m in models.items()}, axis=1)
    volatility.index.name = std_residuals.index.name = "Date"

    descriptive = describe_volatility(volatility)
    print("\nConditional volatility of the synchronized sample (Table 4.2):")
    show(descriptive)
    save_table(descriptive, TABLES, "sync_descriptive_volatility")

    save_table(std_residuals, TABLES, "synchronized_std_residuals")
    volatility.to_csv(config.CONDITIONAL_VOLATILITY_FILE)
    line_plot(volatility, FIGURES / "sync_conditional_volatility.png",
              title="Conditional volatility - synchronized sample",
              ylabel="Conditional volatility (%)")
    print(f"\nSaved Step 2 input: {config.SYNC_STD_RESID_FILE}")
    print(f"Saved Step 3 input: {config.CONDITIONAL_VOLATILITY_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="skip ARMA grid and EGARCH candidate searches")
    args = parser.parse_args()

    assets = load_assets()
    part_a_own_calendar(assets, args.quick)
    part_b_synchronized(assets, args.quick)
    section("STEP 1 SAVED")
    print("Tables :", TABLES)
    print("Figures:", FIGURES)


if __name__ == "__main__":
    main()
