from __future__ import annotations

import numpy as np

from .model import TVPVARResult


def generalized_fevd(model: TVPVARResult, horizon: int = 10) -> np.ndarray:
    """Compute row-normalized generalized FEVD at each time point."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    n_obs, n_vars, _ = model.coefficients.shape
    theta_store = np.zeros((n_obs, n_vars, n_vars))

    for t in range(n_obs):
        coefficient = model.coefficients[t]
        sigma = model.shock_covariances[t]
        moving_average = [
            np.linalg.matrix_power(coefficient, h) for h in range(horizon)
        ]

        theta = np.zeros((n_vars, n_vars))
        for receiver in range(n_vars):
            denominator = sum(
                (phi @ sigma @ phi.T)[receiver, receiver]
                for phi in moving_average
            )
            for shock in range(n_vars):
                numerator = sum(
                    ((phi @ sigma)[receiver, shock]) ** 2
                    for phi in moving_average
                )
                theta[receiver, shock] = (
                    numerator / sigma[shock, shock] / denominator
                )

        row_sums = theta.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0) or not np.isfinite(row_sums).all():
            raise FloatingPointError(f"Invalid GFEVD row sum at observation {t}")
        theta_store[t] = theta / row_sums

    return theta_store
