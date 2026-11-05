"""Step 1: two-step ARMA filtering followed by zero-mean EGARCH-t estimation."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

Order2 = tuple[int, int]
Order3 = tuple[int, int, int]


def _fit_arima(y: pd.Series, order: Order2) -> ARIMAResults:
    p, q = order
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(
            y.reset_index(drop=True),
            order=(p, 0, q),
            trend="c",
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit()


def arma_grid(
    y: pd.Series,
    *,
    max_p: int,
    max_q: int,
    lb_lags: tuple[int, ...] = (10, 15, 20),
) -> pd.DataFrame:
    """Fit every ARMA(p,q) and report AIC, BIC and Ljung-Box p-values of residuals.

    A model passes when every Ljung-Box lag with positive degrees of freedom
    (lag > p + q) has p-value > 0.05.
    """
    y = y.dropna().astype(float)
    rows = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                fit = _fit_arima(y, (p, q))
            except Exception:
                continue
            if not fit.mle_retvals.get("converged", True):
                continue
            lb = acorr_ljungbox(fit.resid, lags=list(lb_lags), model_df=p + q)
            row = {"Model": f"ARMA({p},{q})", "p": p, "q": q, "AIC": fit.aic, "BIC": fit.bic}
            valid = []
            for lag in lb_lags:
                p_value = lb.loc[lag, "lb_pvalue"] if lag > p + q else np.nan
                row[f"LB_p{lag}"] = p_value
                if not np.isnan(p_value):
                    valid.append(p_value > 0.05)
            row["Pass_LB"] = bool(valid) and all(valid)
            rows.append(row)
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def ljung_box_table(
    residuals: pd.Series, *, lags: tuple[int, ...] = (5, 10, 15, 20), model_df: int = 0
) -> pd.DataFrame:
    table = acorr_ljungbox(residuals.dropna(), lags=list(lags), model_df=model_df)
    table.index.name = "Lag"
    return table


def arch_lm_table(
    residuals: pd.Series, *, lags: tuple[int, ...] = (5, 10, 15, 20)
) -> pd.DataFrame:
    rows = []
    for lag in lags:
        lm_stat, lm_p, f_stat, f_p = het_arch(residuals.dropna(), nlags=lag)
        rows.append(
            {
                "Lag": lag,
                "LM statistic": lm_stat,
                "LM p-value": lm_p,
                "F statistic": f_stat,
                "F p-value": f_p,
                "ARCH effect": lm_p < 0.05,
            }
        )
    return pd.DataFrame(rows).set_index("Lag")


def fit_egarch(residuals: pd.Series, order: Order3) -> ARCHModelResult:
    p, o, q = order
    model = arch_model(
        residuals.dropna(),
        mean="Zero",
        vol="EGARCH",
        p=p,
        o=o,
        q=q,
        dist="StudentsT",
        rescale=False,
    )
    return model.fit(disp="off", show_warning=False)


def standardized_diagnostics(
    z: pd.Series, *, lags: tuple[int, ...] = (10, 15, 20)
) -> dict[str, float | bool]:
    """Ljung-Box on z and z^2 and ARCH-LM on z (all H0: no remaining dependence)."""
    z = z.replace([np.inf, -np.inf], np.nan).dropna()
    lb_z = acorr_ljungbox(z, lags=list(lags))
    lb_z2 = acorr_ljungbox(z**2, lags=list(lags))
    row: dict[str, float | bool] = {
        "z_mean": z.mean(),
        "z_variance": z.var(ddof=1),
    }
    for lag in lags:
        row[f"LB_z_p{lag}"] = lb_z.loc[lag, "lb_pvalue"]
    for lag in lags:
        row[f"LB_z2_p{lag}"] = lb_z2.loc[lag, "lb_pvalue"]
    for lag in lags:
        row[f"ARCH_p{lag}"] = het_arch(z, nlags=lag)[1]
    row["Pass_LB_z"] = all(row[f"LB_z_p{lag}"] > 0.05 for lag in lags)
    row["Pass_LB_z2"] = all(row[f"LB_z2_p{lag}"] > 0.05 for lag in lags)
    row["Pass_ARCH"] = all(row[f"ARCH_p{lag}"] > 0.05 for lag in lags)
    return row


def egarch_candidates(
    residuals: pd.Series, orders: tuple[Order3, ...]
) -> pd.DataFrame:
    rows = []
    for order in orders:
        fit = fit_egarch(residuals, order)
        z = fit.resid / fit.conditional_volatility
        diagnostics = standardized_diagnostics(z)
        rows.append(
            {
                "Model": f"EGARCH{order}-t",
                "AIC": fit.aic,
                "BIC": fit.bic,
                "Converged": fit.convergence_flag == 0,
                **diagnostics,
                "nu": fit.params.get("nu", np.nan),
            }
        )
    table = pd.DataFrame(rows)
    table["Pass_All"] = (
        table["Converged"] & table["Pass_LB_z"] & table["Pass_ARCH"]
    )
    return table.sort_values("AIC").reset_index(drop=True)


def parameter_table(fit: ARIMAResults | ARCHModelResult) -> pd.DataFrame:
    if isinstance(fit, ARCHModelResult):
        std_err, t_values = fit.std_err, fit.tvalues
    else:
        std_err, t_values = fit.bse, fit.tvalues
    table = pd.DataFrame(
        {
            "Coefficient": fit.params,
            "Std. error": std_err,
            "t-stat": t_values,
            "p-value": fit.pvalues,
        }
    )
    table["Significant 5%"] = table["p-value"] < 0.05
    return table


@dataclass(frozen=True)
class MarginalModel:
    asset: str
    arma_order: Order2
    egarch_order: Order3
    arma_fit: ARIMAResults
    egarch_fit: ARCHModelResult
    arma_residuals: pd.Series
    volatility: pd.Series
    std_residuals: pd.Series

    @property
    def label(self) -> str:
        return f"ARMA{self.arma_order} -> EGARCH{self.egarch_order}-t"

    def summary(self) -> dict[str, float | str | bool]:
        params = self.egarch_fit.params
        pvalues = self.egarch_fit.pvalues
        return {
            "Asset": self.asset,
            "Model": self.label,
            "N": len(self.volatility),
            "ARMA AIC": self.arma_fit.aic,
            "ARMA BIC": self.arma_fit.bic,
            "EGARCH AIC": self.egarch_fit.aic,
            "EGARCH BIC": self.egarch_fit.bic,
            "Converged": self.egarch_fit.convergence_flag == 0,
            "alpha[1]": params.get("alpha[1]"),
            "gamma[1]": params.get("gamma[1]"),
            "gamma[1] p-value": pvalues.get("gamma[1]"),
            "beta sum": sum(v for k, v in params.items() if k.startswith("beta")),
            "nu": params.get("nu"),
            "Leverage effect": bool(
                params.get("gamma[1]", 0) < 0 and pvalues.get("gamma[1]", 1) < 0.05
            ),
        }


def fit_marginal(
    y: pd.Series, *, asset: str, arma_order: Order2, egarch_order: Order3
) -> MarginalModel:
    """ARMA filter of returns, then EGARCH-t on the ARMA residuals."""
    y = y.dropna().astype(float)
    arma_fit = _fit_arima(y, arma_order)
    residuals = pd.Series(np.asarray(arma_fit.resid), index=y.index, name=asset)
    egarch_fit = fit_egarch(residuals, egarch_order)
    volatility = egarch_fit.conditional_volatility.rename(asset)
    std_residuals = (residuals / volatility).rename(asset)
    return MarginalModel(
        asset=asset,
        arma_order=arma_order,
        egarch_order=egarch_order,
        arma_fit=arma_fit,
        egarch_fit=egarch_fit,
        arma_residuals=residuals,
        volatility=volatility,
        std_residuals=std_residuals,
    )
