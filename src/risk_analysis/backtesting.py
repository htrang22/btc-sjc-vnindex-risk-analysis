"""Step 4: forecast evaluation (loss functions, Mincer-Zarnowitz, Diebold-Mariano)
and VaR/ES backtests (Kupiec, Christoffersen, Basel traffic light, McNeil-Frey,
Acerbi-Szekely Z2).

Daily data has no intraday realized variance, so the volatility proxy is the
squared ARMA residual (summed over h days for the h-day horizon). MSE on the
variance and QLIKE are robust to this noisy proxy (Patton 2011): they rank
models the same way as the true variance would. MAE on volatility is not
robust and is reported only for reference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .forecasting import WalkForward

# ---------------------------------------------------------------------------
# Volatility forecast evaluation
# ---------------------------------------------------------------------------


def _hac_lags(n: int, horizon: int) -> int:
    """Newey-West rule of thumb, at least h-1 for overlapping h-day targets."""
    return max(horizon - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def losses(proxy: np.ndarray, forecast: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "QLIKE": np.log(forecast) + proxy / forecast,
        "MSE": (proxy - forecast) ** 2,
        "MAE_vol": np.abs(np.sqrt(proxy) - np.sqrt(forecast)),
    }


def _paired(result: WalkForward, horizon: int, models: tuple[str, ...]) -> pd.DataFrame:
    f = result.forecasts
    columns = [f"realized{horizon}"] + [f"var{horizon}_{m}" for m in models]
    return f[columns].dropna()


def loss_table(result: WalkForward, horizon: int, models: tuple[str, ...]) -> pd.DataFrame:
    data = _paired(result, horizon, models)
    proxy = data[f"realized{horizon}"].to_numpy()
    rows = []
    for model in models:
        values = losses(proxy, data[f"var{horizon}_{model}"].to_numpy())
        rows.append({"Asset": result.asset, "Horizon": horizon, "Model": model,
                     "N": len(proxy), **{k: v.mean() for k, v in values.items()}})
    table = pd.DataFrame(rows)
    for column in ("QLIKE", "MSE", "MAE_vol"):
        table[f"Rank_{column}"] = table[column].rank().astype(int)
    return table


def mincer_zarnowitz(result: WalkForward, horizon: int, models: tuple[str, ...]) -> pd.DataFrame:
    """proxy = a + b * forecast + u; H0: a = 0, b = 1 (HAC Wald test)."""
    data = _paired(result, horizon, models)
    proxy = data[f"realized{horizon}"].to_numpy()
    lags = _hac_lags(len(proxy), horizon)
    rows = []
    for model in models:
        x = sm.add_constant(data[f"var{horizon}_{model}"].to_numpy())
        fit = sm.OLS(proxy, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        wald = fit.wald_test((np.eye(2), np.array([0.0, 1.0])), scalar=True)
        rows.append({
            "Asset": result.asset, "Horizon": horizon, "Model": model,
            "a": fit.params[0], "b": fit.params[1],
            "b p-value (H0 b=1)": 2 * stats.norm.sf(abs((fit.params[1] - 1) / fit.bse[1])),
            "Wald p-value (a=0, b=1)": float(wald.pvalue),
            "R2": fit.rsquared,
        })
    return pd.DataFrame(rows)


def diebold_mariano(
    result: WalkForward, horizon: int, models: tuple[str, ...], *, reference: str,
    loss: str = "QLIKE",
) -> pd.DataFrame:
    """d = L(benchmark) - L(reference). DM > 0 means the reference is more accurate.

    Uses the Harvey-Leybourne-Newbold small-sample correction and a Student-t
    reference distribution with n-1 degrees of freedom.
    """
    data = _paired(result, horizon, models)
    proxy = data[f"realized{horizon}"].to_numpy()
    base = losses(proxy, data[f"var{horizon}_{reference}"].to_numpy())[loss]
    n = len(proxy)
    lags = _hac_lags(n, horizon)
    rows = []
    for model in models:
        if model == reference:
            continue
        d = losses(proxy, data[f"var{horizon}_{model}"].to_numpy())[loss] - base
        fit = sm.OLS(d, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        dm = fit.tvalues[0]
        hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        stat = dm * hln
        rows.append({
            "Asset": result.asset, "Horizon": horizon, "Loss": loss,
            "Reference": reference, "Benchmark": model,
            "Mean loss diff": d.mean(), "DM (HLN)": stat,
            "p-value two-sided": 2 * stats.t.sf(abs(stat), n - 1),
            "Better": reference if stat > 0 else model,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# VaR backtests
# ---------------------------------------------------------------------------


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def kupiec(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    n, x = len(hits), int(hits.sum())
    pi = x / n
    lr = -2 * (_xlogy(n - x, 1 - alpha) + _xlogy(x, alpha)
               - _xlogy(n - x, 1 - pi) - _xlogy(x, pi))
    return lr, stats.chi2.sf(lr, 1)


def christoffersen(hits: np.ndarray, alpha: float) -> dict[str, float]:
    h = hits.astype(int)
    prev, curr = h[:-1], h[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    log_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    log_alt = (_xlogy(n00, 1 - pi01) + _xlogy(n01, pi01)
               + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11))
    lr_ind = -2 * (log_null - log_alt)
    lr_uc, _ = kupiec(hits, alpha)
    lr_cc = lr_uc + lr_ind
    return {
        "LR_ind": lr_ind, "p_ind": stats.chi2.sf(lr_ind, 1),
        "LR_cc": lr_cc, "p_cc": stats.chi2.sf(lr_cc, 2),
        "Consecutive hits": n11,
    }


def traffic_light(hits: np.ndarray, alpha: float) -> str:
    """Basel zones from the binomial CDF: green < 95%, yellow < 99.99%, red otherwise.

    On 250 days at 99% this gives the Basel bands green 0-4, yellow 5-9, red 10+.
    """
    cdf = stats.binom.cdf(int(hits.sum()), len(hits), alpha)
    if cdf < 0.95:
        return "Green"
    return "Yellow" if cdf < 0.9999 else "Red"


# ---------------------------------------------------------------------------
# ES backtests
# ---------------------------------------------------------------------------


def mcneil_frey(loss: np.ndarray, es: np.ndarray, sigma: np.ndarray, hits: np.ndarray,
                *, rng: np.random.Generator, n_boot: int = 5000) -> tuple[float, float]:
    """Exceedance residuals (L - ES) / sigma on VaR-violation days have mean 0 under H0.

    One-sided bootstrap p-value against mean > 0 (ES underestimates the tail loss).
    """
    r = ((loss - es) / sigma)[hits]
    if len(r) < 2:
        return np.nan, np.nan
    t_obs = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    centered = r - r.mean()
    draws = rng.choice(centered, size=(n_boot, len(r)), replace=True)
    sd = draws.std(axis=1, ddof=1)
    t_boot = np.divide(draws.mean(axis=1), sd / np.sqrt(len(r)),
                       out=np.zeros(n_boot), where=sd > 0)
    return r.mean(), float((t_boot >= t_obs).mean())


def acerbi_szekely_z2(loss: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> float:
    """Z2 = 1 - sum(L * 1{L > VaR} / ES) / (T * alpha). E[Z2] = 0 under H0, < 0 if ES too low."""
    hits = loss > var
    return 1 - np.sum(loss * hits / es) / (len(loss) * alpha)


def z2_pvalue(observed: float, draws_z: np.ndarray, mean: np.ndarray, sigma: np.ndarray,
              var: np.ndarray, es: np.ndarray, alpha: float) -> float:
    """One-sided p-value P(Z2 <= observed) with losses simulated from the forecast model.

    ``draws_z`` has shape (simulations, T): standardized shocks drawn from the
    model's own predictive distribution for each day.
    """
    losses_sim = -(mean + sigma * draws_z)
    hits = losses_sim > var
    z2 = 1 - np.sum(losses_sim * hits / es, axis=1) / (len(mean) * alpha)
    return float((z2 <= observed).mean())


def simulate_shocks(result: WalkForward, table: pd.DataFrame, method: str,
                    *, simulations: int, rng: np.random.Generator) -> np.ndarray:
    n = len(table)
    if method == "t":
        nu = table["nu"].to_numpy()
        draws = rng.standard_t(nu, size=(simulations, n))
        return draws * np.sqrt((nu - 2) / nu)
    draws = np.empty((simulations, n))
    blocks = table["block"].to_numpy()
    for block in np.unique(blocks):
        z = result.z_by_block[block]
        z = z[np.isfinite(z)]
        columns = np.flatnonzero(blocks == block)
        draws[:, columns] = rng.choice(z, size=(simulations, len(columns)))
    return draws


def var_es_backtest(result: WalkForward, table: pd.DataFrame, levels: tuple[float, ...],
                    *, simulations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    loss = table["loss"].to_numpy()
    sigma = table["sigma"].to_numpy()
    mean = table["mean"].to_numpy()
    rows = []
    for method in ("t", "FHS"):
        shocks = simulate_shocks(result, table, method, simulations=simulations, rng=rng)
        for alpha in levels:
            level = round((1 - alpha) * 100)
            var = table[f"VaR{level}_{method}"].to_numpy()
            es = table[f"ES{level}_{method}"].to_numpy()
            hits = loss > var
            lr_uc, p_uc = kupiec(hits, alpha)
            mf_mean, mf_p = mcneil_frey(loss, es, sigma, hits, rng=rng)
            z2 = acerbi_szekely_z2(loss, var, es, alpha)
            last = hits[-250:]
            rows.append({
                "Asset": result.asset,
                "Method": "Student-t" if method == "t" else "FHS",
                "Level": f"{level}%",
                "N": len(hits),
                "Expected hits": len(hits) * alpha,
                "Hits": int(hits.sum()),
                "Hit rate": hits.mean(),
                "LR_uc": lr_uc, "p_uc": p_uc,
                **christoffersen(hits, alpha),
                # The Basel traffic light is defined for 99% VaR only.
                "Traffic light (all)": traffic_light(hits, alpha) if alpha == 0.01 else "",
                "Hits last 250": int(last.sum()),
                "Traffic light (last 250)": (traffic_light(last, alpha)
                                             if alpha == 0.01 else ""),
                "Mean VaR": var.mean(),
                "Mean ES": es.mean(),
                "MF mean exceedance": mf_mean,
                "MF p-value": mf_p,
                "Z2": z2,
                "Z2 p-value": z2_pvalue(z2, shocks, mean, sigma, var, es, alpha),
            })
    return pd.DataFrame(rows)
