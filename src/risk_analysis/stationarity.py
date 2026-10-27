from __future__ import annotations

import warnings

import pandas as pd
from arch.unitroot import PhillipsPerron
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

CASE_NAMES = {
    "n": "No constant, no trend",
    "c": "Constant",
    "t": "Trend",
    "ct": "Constant + trend",
}


def _decision(p_value: float, null_is_stationary: bool) -> str:
    reject = p_value < 0.05
    if null_is_stationary:
        return "Non-stationary" if reject else "Stationary"
    return "Stationary" if reject else "Unit root not rejected"


def unit_root_table(
    series: dict[str, pd.Series],
    *,
    adf_cases: tuple[str, ...] = ("n", "c", "ct"),
    pp_cases: tuple[str, ...] = ("n", "c", "ct"),
    kpss_cases: tuple[str, ...] = ("c", "ct"),
) -> pd.DataFrame:
    """ADF and PP (H0: unit root) and KPSS (H0: stationary) at the 5% level."""
    rows = []
    for asset, values in series.items():
        x = values.dropna().astype(float)
        for case in adf_cases:
            stat, p_value, lag, *_ = adfuller(x, regression=case, autolag="AIC")
            rows.append(("ADF", asset, case, stat, p_value, lag, _decision(p_value, False)))
        for case in pp_cases:
            test = PhillipsPerron(x, trend=case, test_type="tau")
            rows.append(
                ("PP", asset, case, test.stat, test.pvalue, test.lags,
                 _decision(test.pvalue, False))
            )
        for case in kpss_cases:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InterpolationWarning)
                stat, p_value, lag, _ = kpss(x, regression=case, nlags="auto")
            rows.append(("KPSS", asset, case, stat, p_value, lag, _decision(p_value, True)))

    table = pd.DataFrame(
        rows,
        columns=["Test", "Asset", "Case", "Statistic", "p-value", "Lag", "Conclusion"],
    )
    table["Case"] = table["Case"].map(CASE_NAMES)
    return table


def zivot_andrews_table(
    series: dict[str, pd.Series],
    *,
    cases: tuple[str, ...] = ("c", "t", "ct"),
    trim: float = 0.15,
) -> pd.DataFrame:
    """Zivot-Andrews test with one endogenous break (H0: unit root)."""
    rows = []
    for asset, values in series.items():
        x = values.dropna().astype(float)
        for case in cases:
            stat, p_value, critical, lag, break_index = zivot_andrews(
                x, trim=trim, maxlag=None, regression=case, autolag="AIC"
            )
            rows.append(
                {
                    "Asset": asset,
                    "Case": CASE_NAMES[case],
                    "Statistic": stat,
                    "p-value": p_value,
                    "Lag": lag,
                    "Break date": x.index[break_index],
                    "CV 1%": critical["1%"],
                    "CV 5%": critical["5%"],
                    "CV 10%": critical["10%"],
                    "Conclusion": _decision(p_value, False),
                }
            )
    return pd.DataFrame(rows)
