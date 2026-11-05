"""Step 2: bivariate CCC/DCC(1,1) on standardized EGARCH residuals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import chi2

# Same 18 starting values (a, b) as the original multi-start SLSQP search
STARTING_VALUES = (
    (0.001, 0.95), (0.005, 0.95), (0.010, 0.95), (0.020, 0.95),
    (0.010, 0.90), (0.020, 0.90), (0.030, 0.90), (0.050, 0.90),
    (0.020, 0.80), (0.050, 0.80), (0.100, 0.80),
    (0.050, 0.60), (0.100, 0.60), (0.200, 0.50),
    (0.0001, 0.10), (0.0001, 0.50),
    (0.001, 0.01), (0.010, 0.10),
)
PENALTY = 1e12


def long_run_q(z: np.ndarray) -> np.ndarray:
    """Q_bar = (1/T) sum z_t z_t' on the original (not re-standardized) residuals."""
    return z.T @ z / len(z)


def dcc_q_path(z: np.ndarray, q_bar: np.ndarray, a: float, b: float) -> np.ndarray:
    """Q_0 = Q_bar; Q_t = (1-a-b) Q_bar + a z_{t-1} z_{t-1}' + b Q_{t-1}."""
    n_obs, n_vars = z.shape
    outer = np.einsum("ti,tj->tij", z[:-1], z[:-1])
    drive = (1.0 - a - b) * q_bar + a * outer
    q = np.empty((n_obs, n_vars, n_vars))
    q[0] = q_bar
    for i in range(n_vars):
        for j in range(n_vars):
            q[1:, i, j], _ = lfilter(
                [1.0], [1.0, -b], drive[:, i, j], zi=[b * q_bar[i, j]]
            )
    return q


def correlation_path(q: np.ndarray) -> np.ndarray:
    scale = 1.0 / np.sqrt(np.diagonal(q, axis1=1, axis2=2))
    return q * scale[:, :, None] * scale[:, None, :]


def dcc_neg_loglik(params: np.ndarray, z: np.ndarray, q_bar: np.ndarray) -> float:
    """Correlation part of the DCC likelihood, summed over t = 1..T-1."""
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.999999:
        return PENALTY
    q = dcc_q_path(z, q_bar, a, b)[1:]
    if np.any(np.diagonal(q, axis1=1, axis2=2) <= 1e-12):
        return PENALTY
    r = correlation_path(q)
    sign, logdet = np.linalg.slogdet(r)
    if np.any(sign <= 0):
        return PENALTY
    z_t = z[1:]
    correlated = np.einsum("ti,ti->t", z_t, np.linalg.solve(r, z_t[..., None])[..., 0])
    independent = np.einsum("ti,ti->t", z_t, z_t)
    return float(0.5 * np.sum(logdet + correlated - independent))


@dataclass(frozen=True)
class PairResult:
    pair: tuple[str, str]
    dates: pd.Index
    z: np.ndarray
    q_bar: np.ndarray
    multistart: pd.DataFrame
    a: float
    b: float
    model: str  # "CCC" or "DCC(1,1)"

    @property
    def name(self) -> str:
        return "-".join(self.pair)

    @property
    def ccc_correlation(self) -> float:
        d = 1.0 / np.sqrt(np.diag(self.q_bar))
        return float(self.q_bar[0, 1] * d[0] * d[1])

    @property
    def dcc_path(self) -> pd.Series:
        r = correlation_path(dcc_q_path(self.z, self.q_bar, self.a, self.b))
        return pd.Series(r[:, 0, 1], index=self.dates, name=self.name)

    @property
    def final_path(self) -> pd.Series:
        """CCC pairs are constant; DCC pairs follow the estimated path."""
        if self.model == "CCC":
            return pd.Series(self.ccc_correlation, index=self.dates, name=self.name)
        return self.dcc_path


def estimate_pair(
    residuals: pd.DataFrame,
    pair: tuple[str, str],
    *,
    ccc_threshold: float,
) -> PairResult:
    frame = residuals[list(pair)].replace([np.inf, -np.inf], np.nan).dropna()
    z = frame.to_numpy()
    q_bar = long_run_q(z)

    rows = []
    for start_a, start_b in STARTING_VALUES:
        opt = minimize(
            dcc_neg_loglik,
            x0=np.array([start_a, start_b]),
            args=(z, q_bar),
            method="SLSQP",
            bounds=[(1e-10, 0.999), (1e-10, 0.999)],
            constraints=[{"type": "ineq", "fun": lambda x: 0.999999 - x[0] - x[1]}],
            options={"maxiter": 5000, "ftol": 1e-12},
        )
        rows.append(
            {
                "start_a": start_a,
                "start_b": start_b,
                "a": opt.x[0],
                "b": opt.x[1],
                "a+b": opt.x.sum(),
                "NegLogLik": opt.fun,
                "Converged": opt.success,
                "Iterations": opt.nit,
            }
        )
    multistart = pd.DataFrame(rows).sort_values("NegLogLik").reset_index(drop=True)
    converged = multistart[multistart["Converged"]]
    if converged.empty:
        raise RuntimeError(f"DCC multi-start did not converge for {pair}")
    best = converged.iloc[0]
    model = "CCC" if best["a"] < ccc_threshold else "DCC(1,1)"
    return PairResult(
        pair=pair,
        dates=frame.index,
        z=z,
        q_bar=q_bar,
        multistart=multistart,
        a=float(best["a"]),
        b=float(best["b"]),
        model=model,
    )


def engle_sheppard_test(z: np.ndarray, lag: int) -> tuple[float, int, float]:
    """Constant-correlation LM test on centred vech(z_t z_t') (H0: CCC)."""
    n_obs, n_vars = z.shape
    rows, cols = np.tril_indices(n_vars)
    w = np.einsum("ti,tj->tij", z, z)[:, rows, cols]
    w = w - w.mean(axis=0)
    m = w.shape[1]

    y = w[lag:]
    x = np.hstack([w[lag - l: n_obs - l] for l in range(1, lag + 1)])
    x = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    ssr = ((y - x @ beta) ** 2).sum(axis=0)
    tss = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(tss > 1e-14, np.maximum(1.0 - ssr / tss, 0.0), 0.0)

    statistic = len(y) * r2.sum()
    df = m * m * lag
    return float(statistic), df, float(chi2.sf(statistic, df))


def engle_sheppard_table(
    results: list[PairResult], lags: tuple[int, ...] = (1, 2, 5, 10)
) -> pd.DataFrame:
    rows = []
    for result in results:
        for lag in lags:
            statistic, df, p_value = engle_sheppard_test(result.z, lag)
            rows.append(
                {
                    "Pair": result.name,
                    "Lag": lag,
                    "Statistic": statistic,
                    "df": df,
                    "p-value": p_value,
                    "Decision 5%": "Reject CCC" if p_value < 0.05 else "Do not reject CCC",
                }
            )
    return pd.DataFrame(rows)


def matrix_validity(result: PairResult) -> dict[str, float | bool | str]:
    path = result.final_path.to_numpy()
    r = np.empty((len(path), 2, 2))
    r[:, 0, 0] = r[:, 1, 1] = 1.0
    r[:, 0, 1] = r[:, 1, 0] = path
    eigenvalues = np.linalg.eigvalsh(r)
    return {
        "Pair": f"{result.name} ({result.model})",
        "Min eigenvalue": eigenvalues.min(),
        "Min correlation": path.min(),
        "Max correlation": path.max(),
        "Positive definite": bool(eigenvalues.min() > 0),
        "Within [-1, 1]": bool(np.all(np.abs(path) <= 1)),
    }


def correlation_strength(rho: float) -> str:
    magnitude = abs(rho)
    if magnitude < 0.10:
        return "Very weak"
    if magnitude < 0.30:
        return "Weak"
    if magnitude < 0.50:
        return "Moderate"
    return "Strong"


def correlation_episodes(path: pd.Series, *, top: int = 10) -> dict[str, pd.DataFrame]:
    """Largest 1-, 5- and 20-observation moves and sign reversals of a DCC path."""
    frame = path.rename("rho").to_frame()
    episodes = {}
    for window, n in ((1, top), (5, 5), (20, 5)):
        change = frame["rho"].diff(window).rename(f"delta_{window}")
        joined = frame.join(change)
        episodes[f"top_increases_{window}"] = joined.nlargest(n, change.name)
        episodes[f"top_decreases_{window}"] = joined.nsmallest(n, change.name)

    previous = frame["rho"].shift(1)
    reversal = np.select(
        [(previous >= 0) & (frame["rho"] < 0), (previous <= 0) & (frame["rho"] > 0)],
        ["Positive -> Negative", "Negative -> Positive"],
        default="",
    )
    frame = frame.assign(previous=previous, reversal=reversal)
    episodes["sign_reversals"] = frame[frame["reversal"] != ""]
    return episodes
