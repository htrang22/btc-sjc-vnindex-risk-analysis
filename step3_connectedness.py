"""Step 3 - TVP-VAR dynamic connectedness of BTC, SJC and VN-Index volatility.

Input: conditional volatility of the synchronized ARMA-EGARCH models (Step 1).
A TVP-VAR(1) with forgetting factors gives time-varying coefficients, from
which the TVP-GFEVD and the TCI_t, FROM_t, TO_t, NET_t and net pairwise
connectedness are computed. Six specifications check robustness to the
forgetting factors and the forecast horizon.

Usage (from the repository root, after step1_arma_egarch.py):

    python step3_connectedness.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning

# Trading-day dates have no fixed frequency; statsmodels warns but ignores it
warnings.filterwarnings("ignore", category=ValueWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from risk_analysis import config  # noqa: E402
from risk_analysis.data import describe_volatility  # noqa: E402
from risk_analysis.reporting import save_table, section, show  # noqa: E402
from risk_analysis.stationarity import unit_root_table, zivot_andrews_table  # noqa: E402
from tvp_connectedness import (  # noqa: E402
    compare_specifications,
    load_volatility_data,
    run_specification,
    save_results,
)
from tvp_connectedness.connectedness import gfevd_long, net_pairwise  # noqa: E402
from tvp_connectedness.lag_selection import select_lag_order  # noqa: E402
from tvp_connectedness.plots import (  # noqa: E402
    plot_net,
    plot_net_pairwise,
    plot_sensitivity,
    plot_stability,
    plot_tci,
    plot_volatility,
)

TABLES = config.OUTPUTS / config.STEP3_DIR
FIGURES = config.FIGURES / config.STEP3_DIR


def main() -> None:
    data = load_volatility_data(config.CONDITIONAL_VOLATILITY_FILE)
    if not (data > 0).all().all():
        raise ValueError("conditional volatility must be strictly positive")

    section("STEP 3 - INPUT: CONDITIONAL VOLATILITY")
    print(f"Observations: {len(data)}, "
          f"{data.index[0].date()} -> {data.index[-1].date()}")
    descriptive = describe_volatility(data)
    show(descriptive)
    save_table(descriptive, TABLES, "descriptive_volatility")
    plot_volatility(data, FIGURES)

    # Volatility is positive and does not fluctuate around zero: tests use a drift
    section("STATIONARITY OF CONDITIONAL VOLATILITY")
    series = {asset: data[asset] for asset in data.columns}
    unit_root = unit_root_table(series, adf_cases=("c",), pp_cases=("c",))
    show(unit_root.set_index(["Test", "Asset", "Case"]))
    save_table(unit_root, TABLES, "unit_root_tests", index=False)
    za = zivot_andrews_table({"SJC": data["SJC"]}, cases=("ct",))
    print("\nZivot-Andrews (SJC):")
    show(za.set_index(["Asset", "Case"]))
    save_table(za, TABLES, "zivot_andrews_sjc", index=False)

    section("LAG ORDER (BIC and HQIC select p = 1 -> TVP-VAR(1))")
    lags = select_lag_order(data)
    show(lags)
    save_table(lags, TABLES, "lag_selection")

    section("TVP-VAR(1) SPECIFICATIONS")
    results = {
        spec["name"]: run_specification(data, initial_covariance=0.01, **spec)
        for spec in config.TVP_SPECIFICATIONS
    }
    for result in results.values():
        save_results(result, TABLES / "specifications" / result.name)
    comparison = compare_specifications(list(results.values()))
    show(comparison.T)
    save_table(comparison, TABLES, "specification_comparison")

    baseline = results[config.TVP_BASELINE]
    connectedness = baseline.connectedness
    variables = list(baseline.model.variables)

    gfevd_final = pd.DataFrame(
        connectedness.theta[-1] * 100, index=variables, columns=variables
    )
    gfevd_mean = pd.DataFrame(
        connectedness.theta.mean(axis=0) * 100, index=variables, columns=variables
    )
    gfevd_final.index.name = gfevd_mean.index.name = "Receiver \\ Shock"
    mean_table = pd.DataFrame(
        {
            "FROM": connectedness.from_others.mean(),
            "TO": connectedness.to_others.mean(),
            "NET": connectedness.net.mean(),
        }
    )
    mean_table["Role"] = mean_table["NET"].map(
        lambda net: "Net transmitter" if net > 0 else "Net receiver"
    )
    final_table = connectedness.final_table
    final_table["Role"] = final_table["NET"].map(
        lambda net: "Net transmitter" if net > 0 else "Net receiver"
    )
    pairwise = net_pairwise(connectedness)
    pairwise.index.name = "Date"

    section(f"BASELINE {config.TVP_BASELINE}: TVP-GFEVD AND CONNECTEDNESS")
    print("Final-period normalized GFEVD (%):")
    show(gfevd_final)
    print("\nSample-average normalized GFEVD (%):")
    show(gfevd_mean)
    print(f"\nTCI: final {connectedness.tci.iloc[-1]:.4f}%, "
          f"mean {connectedness.tci.mean():.4f}%, "
          f"max {connectedness.tci.max():.4f}% on {connectedness.tci.idxmax().date()}")
    print("\nFinal-period connectedness (%):")
    show(final_table)
    print("\nSample-average connectedness (%):")
    show(mean_table)
    print("\nMean net pairwise connectedness (%):")
    show(pairwise.mean())
    print("\nLocally unstable episodes (spectral radius >= 1):")
    print(baseline.episodes.to_string(index=False))

    baseline_dir = TABLES / "baseline"
    save_table(gfevd_final, baseline_dir, "gfevd_final")
    save_table(gfevd_mean, baseline_dir, "gfevd_mean")
    save_table(gfevd_long(connectedness), baseline_dir, "gfevd_time_varying")
    save_table(final_table, baseline_dir, "connectedness_final")
    save_table(mean_table, baseline_dir, "connectedness_mean")
    save_table(pairwise, baseline_dir, "net_pairwise")

    plot_stability(baseline, FIGURES)
    plot_tci(baseline, FIGURES)
    plot_net(baseline, FIGURES)
    plot_net_pairwise(baseline, FIGURES)
    plot_sensitivity(results, FIGURES, baseline=config.TVP_BASELINE)

    section("STEP 3 SAVED")
    print("Tables :", TABLES)
    print("Figures:", FIGURES)


if __name__ == "__main__":
    main()
