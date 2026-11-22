from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

ASSETS = ("BTC", "SJC", "VNINDEX")
ASSET_LABELS = {"BTC": "BTC-VND", "SJC": "Gold SJC", "VNINDEX": "VN-Index"}

# file, return column (returns computed on each asset's own calendar)
RETURN_FILES = {
    "BTC": ("3y-bitcoin-vnd-returns.csv", "btc_return"),
    "SJC": ("3y-gold-returns.csv", "gold_return"),
    "VNINDEX": ("3y-vnindex-returns.csv", "vnindex_return"),
}

# Step 1 / Model 1: own-calendar returns (paper sections 4.2.1-4.2.4)
MODEL1_ARMA_SEARCH = {"BTC": 2, "SJC": 1, "VNINDEX": 6}
MODEL1_ARMA = {"BTC": (0, 0), "SJC": (0, 1), "VNINDEX": (0, 5)}
MODEL1_EGARCH = {"BTC": (1, 1, 2), "SJC": (1, 1, 1), "VNINDEX": (1, 1, 1)}

# Step 1 / synchronized sample: marginal filters feeding DCC and VAR
SYNC_ARMA_SEARCH = 6
SYNC_ARMA = {"BTC": (1, 0), "SJC": (0, 0), "VNINDEX": (0, 5)}
SYNC_EGARCH = {"BTC": (1, 1, 1), "SJC": (1, 1, 1), "VNINDEX": (1, 1, 1)}

EGARCH_CANDIDATES = ((1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2))

# Step 2: pairs and the a-threshold below which DCC collapses to CCC
DCC_PAIRS = (("BTC", "SJC"), ("BTC", "VNINDEX"), ("SJC", "VNINDEX"))
CCC_A_THRESHOLD = 1e-4

# Step 3: TVP-VAR(1) dynamic connectedness
TVP_SPECIFICATIONS = (
    {"name": "baseline_0999", "lambda_beta": 0.999, "kappa_sigma": 0.96, "horizon": 10},
    {"name": "robustness_0997", "lambda_beta": 0.997, "kappa_sigma": 0.96, "horizon": 10},
    {"name": "kappa_094", "lambda_beta": 0.999, "kappa_sigma": 0.94, "horizon": 10},
    {"name": "kappa_098", "lambda_beta": 0.999, "kappa_sigma": 0.98, "horizon": 10},
    {"name": "horizon_05", "lambda_beta": 0.999, "kappa_sigma": 0.96, "horizon": 5},
    {"name": "horizon_20", "lambda_beta": 0.999, "kappa_sigma": 0.96, "horizon": 20},
)
TVP_BASELINE = "baseline_0999"

# Step 4: walk-forward volatility forecasts and 1-day VaR/ES backtests.
# Own-calendar returns with the Model 1 specifications; expanding window.
FORECAST_INITIAL_WINDOW = 500      # first estimation window (observations)
FORECAST_REFIT_EVERY = 20          # re-estimate parameters every 20 observations
FORECAST_HORIZONS = (1, 5)
FORECAST_SIMULATIONS = 2000        # bootstrap paths for EGARCH multi-step forecasts
FORECAST_SEED = 20260801
EWMA_LAMBDA = 0.94                 # RiskMetrics daily decay
ROLLING_WINDOW = 60                # rolling-variance benchmark (about 3 months)
VAR_LEVELS = (0.05, 0.01)          # tail probabilities: 95% and 99% VaR/ES
ES_BACKTEST_SIMULATIONS = 5000

# Step 4C: true holdout. Returns after the original sample end, built with the
# same pipeline (see DATA_PROVENANCE.md). Optional and not redistributed; the
# file overlaps the original sample, only dates after ORIGINAL_SAMPLE_END are used.
ORIGINAL_SAMPLE_END = "2026-08-01"
HOLDOUT_FILES = {
    "BTC": "holdout-bitcoin-vnd-returns.csv",
    "SJC": "holdout-gold-returns.csv",
    "VNINDEX": "holdout-vnindex-returns.csv",
}

# Files shared between steps
STEP1_DIR = "step1_arma_egarch"
STEP2_DIR = "step2_dcc_garch"
STEP3_DIR = "step3_connectedness"
STEP4_DIR = "step4_forecast_var"
SYNC_STD_RESID_FILE = OUTPUTS / STEP1_DIR / "synchronized_std_residuals.csv"
CONDITIONAL_VOLATILITY_FILE = OUTPUTS / "model2_conditional_volatility.csv"
