from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

ASSETS = ("BTC", "SJC", "VNINDEX")
ASSET_LABELS = {"BTC": "BTC-VND", "SJC": "Gold SJC", "VNINDEX": "VN-Index"}

# file, price column, return column (returns computed on each asset's own calendar)
RETURN_FILES = {
    "BTC": ("3y-bitcoin-vnd-returns.csv", "btc_vnd_price", "btc_return"),
    "SJC": ("3y-gold-returns.csv", "gold_price", "gold_return"),
    "VNINDEX": ("3y-vnindex-returns.csv", "vnindex_price", "vnindex_return"),
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

# Files shared between steps
STEP1_DIR = "step1_arma_egarch"
STEP2_DIR = "step2_dcc_garch"
STEP3_DIR = "step3_connectedness"
SYNC_STD_RESID_FILE = OUTPUTS / STEP1_DIR / "synchronized_std_residuals.csv"
CONDITIONAL_VOLATILITY_FILE = OUTPUTS / "model2_conditional_volatility.csv"
