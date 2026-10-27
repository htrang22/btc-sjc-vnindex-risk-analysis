from __future__ import annotations

import pandas as pd
from statsmodels.tsa.api import VAR


def select_lag_order(data: pd.DataFrame, *, max_lags: int = 20) -> pd.DataFrame:
    """Preliminary VAR lag-order selection by AIC, BIC, FPE and HQIC."""
    selection = VAR(data.astype(float)).select_order(maxlags=max_lags, trend="c")
    return pd.DataFrame(
        {
            "criterion": ["AIC", "BIC", "FPE", "HQIC"],
            "selected_lag": [
                selection.aic,
                selection.bic,
                selection.fpe,
                selection.hqic,
            ],
        }
    ).set_index("criterion")
