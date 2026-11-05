"""Step 2 - CCC/DCC-GARCH conditional correlations between BTC, SJC and VN-Index.

Input: standardized residuals of the synchronized ARMA-EGARCH models (Step 1).
Each pair is estimated as a bivariate DCC(1,1) by multi-start SLSQP; a pair
whose shock coefficient a collapses to zero is reported as CCC.

Usage (from the repository root, after step1_arma_egarch.py):

    python step2_dcc_garch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from risk_analysis import config  # noqa: E402
from risk_analysis.dcc import (  # noqa: E402
    correlation_episodes,
    correlation_strength,
    engle_sheppard_table,
    estimate_pair,
    matrix_validity,
)
from risk_analysis.plotting import plt, save_figure  # noqa: E402
from risk_analysis.reporting import save_table, section, show  # noqa: E402

TABLES = config.OUTPUTS / config.STEP2_DIR
FIGURES = config.FIGURES / config.STEP2_DIR


def plot_dcc_path(path: pd.Series, file: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(path.index, path, linewidth=1.2)
    ax.axhline(path.mean(), linestyle="--", label=f"Mean = {path.mean():.4f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title=f"Dynamic conditional correlation: {path.name}",
           xlabel="Date", ylabel="Conditional correlation")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, file)


def plot_all_correlations(correlations: pd.DataFrame, labels: dict[str, str], file: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    for column in correlations.columns:
        ax.plot(correlations.index, correlations[column], linewidth=1.6, label=labels[column])
    ax.axhline(0, linestyle="--", linewidth=1, color="black")

    for column, label in labels.items():
        if not label.endswith("(DCC)"):
            continue
        path = correlations[column]
        for when, offset in ((path.idxmin(), (30, 10)), (path.idxmax(), (-110, -10))):
            ax.scatter(when, path[when], s=55, zorder=5)
            ax.annotate(
                f"{path[when]:.3f}\n{when:%Y-%m-%d}",
                xy=(when, path[when]),
                xytext=offset,
                textcoords="offset points",
                arrowprops={"arrowstyle": "->"},
            )
    ax.set(title="Conditional correlations among BTC, SJC gold and VN-Index",
           xlabel="Date", ylabel="Conditional correlation")
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(fig, file)


def main() -> None:
    residuals = pd.read_csv(config.SYNC_STD_RESID_FILE, parse_dates=["Date"]).set_index("Date")
    section("STEP 2 - CCC/DCC-GARCH ON STANDARDIZED RESIDUALS")
    print(f"Observations: {len(residuals)}, "
          f"{residuals.index[0].date()} -> {residuals.index[-1].date()}")
    print("\nUnconditional correlation of standardized residuals:")
    show(residuals.corr())

    results = [
        estimate_pair(residuals, pair, ccc_threshold=config.CCC_A_THRESHOLD)
        for pair in config.DCC_PAIRS
    ]

    for result in results:
        print(f"\n{result.name}: best multi-start solution "
              f"a={result.a:.6f}, b={result.b:.6f}, a+b={result.a + result.b:.6f} "
              f"-> {result.model} "
              f"({int(result.multistart['Converged'].sum())}/{len(result.multistart)} starts converged)")
        save_table(result.multistart, TABLES,
                   f"multistart_{result.name.lower().replace('-', '_')}", index=False)

    specification = pd.DataFrame(
        {
            "Pair": [r.name for r in results],
            "Model": [r.model for r in results],
            "a": [r.a for r in results],
            "b": [r.b for r in results],
            "a+b": [r.a + r.b for r in results],
            "CCC correlation": [r.ccc_correlation for r in results],
        }
    )
    section("FINAL SPECIFICATION")
    show(specification.set_index("Pair"), digits=6)
    save_table(specification, TABLES, "specification", index=False)

    tests = engle_sheppard_table(results)
    section("ENGLE-SHEPPARD CONSTANT-CORRELATION TEST (H0: CCC)")
    show(tests.set_index(["Pair", "Lag"]))
    save_table(tests, TABLES, "engle_sheppard_test", index=False)

    validity = pd.DataFrame([matrix_validity(r) for r in results])
    save_table(validity, TABLES, "correlation_matrix_validity", index=False)

    correlations = pd.concat([r.final_path for r in results], axis=1)
    correlations.index.name = "Date"
    save_table(correlations, TABLES, "conditional_correlations")

    rows = []
    for result in results:
        path = result.final_path
        rows.append(
            {
                "Pair": result.name,
                "Model": result.model,
                "Mean": path.mean(),
                "Std": path.std(),
                "Median": path.median(),
                "Min": path.min(),
                "Min date": path.idxmin().date() if result.model != "CCC" else None,
                "Max": path.max(),
                "Max date": path.idxmax().date() if result.model != "CCC" else None,
                "Strength": correlation_strength(path.mean()),
            }
        )
    summary = pd.DataFrame(rows)
    section("CORRELATION SUMMARY")
    show(summary.set_index("Pair"), digits=6)
    save_table(summary, TABLES, "correlation_summary", index=False)

    labels = {}
    for result in results:
        labels[result.name] = f"{result.name} ({'CCC' if result.model == 'CCC' else 'DCC'})"
        if result.model == "CCC":
            continue
        path = result.dcc_path
        plot_dcc_path(path, FIGURES / f"dcc_{result.name.lower().replace('-', '_')}.png")
        episodes = correlation_episodes(path)
        print(f"\n{result.name} sign reversals: {len(episodes['sign_reversals'])}")
        for name, table in episodes.items():
            save_table(table, TABLES / "episodes", f"{result.name.lower().replace('-', '_')}_{name}")

    plot_all_correlations(correlations, labels, FIGURES / "conditional_correlations.png")

    section("STEP 2 SAVED")
    print("Tables :", TABLES)
    print("Figures:", FIGURES)


if __name__ == "__main__":
    main()
