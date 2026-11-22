"""Step 4: walk-forward (expanding window) volatility forecasts.

Every ``refit_every`` observations the ARMA mean and the volatility models are
re-estimated on all data observed so far. Between refits the parameters stay
fixed and the filters are updated with each new observation, so every forecast
for day t uses information up to day t-1 only.

All volatility models are fitted to the same ARMA residuals, so the comparison
isolates the variance equation:

* EGARCH-t : the Model 1 specification (the model under evaluation)
* GARCH-t  : GARCH(1,1) with Student-t errors
* GJR-t    : GJR-GARCH(1,1,1) with Student-t errors
* EWMA     : RiskMetrics, sigma2[t+1] = lambda * sigma2[t] + (1 - lambda) * e[t]^2
* Rolling  : mean of e^2 over the previous ``rolling_window`` observations

The h-day forecast is the cumulative variance of days t..t+h-1 from origin t-1;
EWMA and Rolling are flat, so theirs is h times the 1-day forecast. GARCH and
GJR have analytic multi-step forecasts. EGARCH does not, and with Student-t
errors the multi-step conditional variance is not even finite: E[exp(a|z|)]
does not exist for a t-distributed z. Simulating from the fitted t therefore
gives unstable averages, so EGARCH multi-step forecasts are bootstrapped from
the in-sample standardized residuals of each estimation window instead.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from .arma_egarch import Order2, Order3, _fit_arima

VOL_MODELS = ("EGARCH-t", "GARCH-t", "GJR-t", "EWMA", "Rolling")


@dataclass(frozen=True)
class WalkForward:
    """Daily out-of-sample forecasts and the in-sample EGARCH z of each refit block."""

    asset: str
    forecasts: pd.DataFrame
    z_by_block: dict[int, np.ndarray]
    parameters: pd.DataFrame          # EGARCH-t estimates of every refit block

    def after(self, cutoff: str) -> "WalkForward":
        """Only the forecasts for dates after ``cutoff`` (e.g. a holdout period)."""
        keep = self.forecasts.index > pd.Timestamp(cutoff)
        return WalkForward(self.asset, self.forecasts.loc[keep], self.z_by_block,
                           self.parameters)


def _arch_spec(name: str, egarch_order: Order3) -> dict:
    if name == "EGARCH-t":
        p, o, q = egarch_order
        return {"vol": "EGARCH", "p": p, "o": o, "q": q}
    if name == "GARCH-t":
        return {"vol": "GARCH", "p": 1, "o": 0, "q": 1}
    if name == "GJR-t":
        return {"vol": "GARCH", "p": 1, "o": 1, "q": 1}
    raise ValueError(name)


def _generic_starts(name: str, egarch_order: Order3, variance: float) -> list[np.ndarray]:
    """A few economically sensible starting points (no look-ahead information)."""
    starts = []
    for nu in (5.0, 3.0):
        if name == "EGARCH-t":
            p, o, q = egarch_order
            for persistence in (0.95, 0.85):
                betas = [persistence] if q == 1 else [persistence * 0.55, persistence * 0.45]
                betas += [0.0] * (q - len(betas))
                omega = (1 - persistence) * np.log(variance)
                starts.append(np.array([omega] + [0.15] * p + [-0.05] * o + betas + [nu]))
        elif name == "GARCH-t":
            starts.append(np.array([0.05 * variance, 0.08, 0.90, nu]))
        else:
            starts.append(np.array([0.05 * variance, 0.06, 0.05, 0.88, nu]))
    return starts


def _robust_fit(model, *, last_obs: int, starts: list[np.ndarray | None]):
    """Fit from several starting values and keep the highest log-likelihood.

    The default starting values of ``arch`` occasionally land the EGARCH
    optimizer in a degenerate optimum on the shorter early windows, which makes
    the variance filter explode out of sample.
    """
    best = None
    for start in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = model.fit(last_obs=last_obs, disp="off", show_warning=False,
                                starting_values=start, options={"maxiter": 1000})
            except Exception:
                continue
        if not np.all(np.isfinite(fit.params)) or not np.isfinite(fit.loglikelihood):
            continue
        key = (fit.convergence_flag == 0, fit.loglikelihood)
        if best is None or key > best[0]:
            best = (key, fit)
    if best is None:
        raise RuntimeError("all starting values failed")
    return best[1]


def _cumulative(variance: pd.DataFrame, horizon: int) -> np.ndarray:
    return variance.iloc[:, :horizon].sum(axis=1).to_numpy()


def walk_forward(
    y: pd.Series,
    *,
    asset: str,
    arma_order: Order2,
    egarch_order: Order3,
    initial: int,
    refit_every: int,
    horizon: int,
    simulations: int,
    seed: int,
    ewma_lambda: float,
    rolling_window: int,
) -> WalkForward:
    y = y.dropna().astype(float)
    n = len(y)
    values = y.to_numpy()
    if initial >= n:
        raise ValueError(f"{asset}: initial window {initial} >= sample size {n}")

    blocks: list[pd.DataFrame] = []
    z_by_block: dict[int, np.ndarray] = {}
    previous: dict[str, np.ndarray] = {}         # last block's estimates as a start
    parameters: list[dict] = []
    for block, t0 in enumerate(range(initial, n, refit_every)):
        end = min(t0 + refit_every, n)           # one-step targets t0 .. end-1
        last = min(end - 1 + horizon, n)         # data needed for h-day realized
        targets = np.arange(t0, end)

        arma = _fit_arima(y.iloc[:t0], arma_order)
        filtered = arma.apply(pd.Series(values[:last]))
        e = np.asarray(filtered.resid, dtype=float)
        mean = values[:last] - e
        resid = pd.Series(e)

        out = {
            "Date": y.index[targets],
            "block": block,
            "return": values[targets],
            "mean": mean[targets],
            "resid": e[targets],
        }

        for name in ("EGARCH-t", "GARCH-t", "GJR-t"):
            model = arch_model(resid, mean="Zero", dist="StudentsT", rescale=False,
                               **_arch_spec(name, egarch_order))
            starts = [None, previous.get(name)] + _generic_starts(
                name, egarch_order, float(np.mean(e[:t0] ** 2)))
            fit = _robust_fit(model, last_obs=t0, starts=starts)
            previous[name] = fit.params.to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                one_step = fit.forecast(horizon=1, start=t0 - 1, reindex=False).variance
                if name == "EGARCH-t":
                    multi = fit.forecast(
                        horizon=horizon, start=t0 - 1, reindex=False,
                        method="bootstrap", simulations=simulations,
                        random_state=np.random.RandomState(seed + block),
                    ).variance
                else:
                    multi = fit.forecast(horizon=horizon, start=t0 - 1, reindex=False).variance
            k = len(targets)
            out[f"var1_{name}"] = one_step.iloc[:k, 0].to_numpy()
            out[f"var{horizon}_{name}"] = _cumulative(multi.iloc[:k], horizon)
            if name == "EGARCH-t":
                out["nu"] = fit.params["nu"]
                sigma_in = fit.conditional_volatility[:t0]
                z_by_block[block] = (e[:t0] / np.asarray(sigma_in))
                parameters.append({
                    "Asset": asset, "Block": block, "Estimation end": y.index[t0 - 1],
                    "N": t0, "Converged": fit.convergence_flag == 0,
                    "LogLik": fit.loglikelihood, **fit.params.to_dict(),
                    "beta sum": sum(v for k, v in fit.params.items() if k.startswith("beta")),
                })

        e2 = e**2
        ewma = np.empty(last + 1)
        ewma[0] = e2[: min(30, t0)].mean()
        for t in range(last):
            ewma[t + 1] = ewma_lambda * ewma[t] + (1 - ewma_lambda) * e2[t]
        out["var1_EWMA"] = ewma[targets]
        out[f"var{horizon}_EWMA"] = horizon * ewma[targets]

        rolling = np.array([e2[t - rolling_window:t].mean() for t in targets])
        out["var1_Rolling"] = rolling
        out[f"var{horizon}_Rolling"] = horizon * rolling

        realized = np.full(len(targets), np.nan)
        for i, t in enumerate(targets):
            if t + horizon <= last:
                realized[i] = e2[t:t + horizon].sum()
        out[f"realized{horizon}"] = realized
        out["realized1"] = e2[targets]
        blocks.append(pd.DataFrame(out))

    forecasts = pd.concat(blocks, ignore_index=True).set_index("Date")
    return WalkForward(asset=asset, forecasts=forecasts, z_by_block=z_by_block,
                       parameters=pd.DataFrame(parameters))


# ---------------------------------------------------------------------------
# 1-day VaR and ES (reported as positive losses, loss = -return)
# ---------------------------------------------------------------------------

def standardized_t_tail(alpha: float, nu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower-tail quantile and expected shortfall of a unit-variance Student-t."""
    nu = np.asarray(nu, dtype=float)
    scale = np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(alpha, nu)
    tail_mean = -stats.t.pdf(q, nu) / alpha * (nu + q**2) / (nu - 1)
    return q * scale, tail_mean * scale


def empirical_tail(z: np.ndarray, alpha: float) -> tuple[float, float]:
    z = z[np.isfinite(z)]
    q = np.quantile(z, alpha)
    return q, z[z <= q].mean()


def var_es_table(result: WalkForward, levels: tuple[float, ...]) -> pd.DataFrame:
    """VaR and ES from the EGARCH-t forecast: parametric Student-t and FHS."""
    f = result.forecasts
    sigma = np.sqrt(f["var1_EGARCH-t"].to_numpy())
    mean = f["mean"].to_numpy()
    table = pd.DataFrame(index=f.index)
    table["loss"] = -f["return"]
    table["mean"] = mean
    table["sigma"] = sigma
    table["nu"] = f["nu"]
    table["block"] = f["block"]
    for alpha in levels:
        level = round((1 - alpha) * 100)
        q, es = standardized_t_tail(alpha, f["nu"].to_numpy())
        table[f"VaR{level}_t"] = -(mean + sigma * q)
        table[f"ES{level}_t"] = -(mean + sigma * es)
        tails = {b: empirical_tail(z, alpha) for b, z in result.z_by_block.items()}
        q_fhs = f["block"].map(lambda b: tails[b][0]).to_numpy()
        es_fhs = f["block"].map(lambda b: tails[b][1]).to_numpy()
        table[f"VaR{level}_FHS"] = -(mean + sigma * q_fhs)
        table[f"ES{level}_FHS"] = -(mean + sigma * es_fhs)
    return table
