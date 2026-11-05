from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


@dataclass(frozen=True)
class TVPVARResult:
    dates: pd.Index
    variables: tuple[str, ...]
    lambda_beta: float
    kappa_sigma: float
    initial_covariance: float
    coefficients: np.ndarray
    constants: np.ndarray
    shock_covariances: np.ndarray
    innovations: np.ndarray


def estimate_tvp_var(
    data: pd.DataFrame,
    *,
    lambda_beta: float = 0.999,
    kappa_sigma: float = 0.96,
    initial_covariance: float = 0.01,
) -> TVPVARResult:
    """Estimate the project's TVP-VAR(1) forgetting-factor specification."""
    if not 0 < lambda_beta <= 1:
        raise ValueError("lambda_beta must be in (0, 1]")
    if not 0 < kappa_sigma < 1:
        raise ValueError("kappa_sigma must be in (0, 1)")
    if initial_covariance <= 0:
        raise ValueError("initial_covariance must be positive")
    if data.shape[1] < 2 or len(data) < 3:
        raise ValueError("data must contain at least two variables and three rows")
    if data.isna().any().any() or not np.isfinite(data.to_numpy()).all():
        raise ValueError("data contains missing or non-finite values")

    clean = data.astype(float).copy()
    y = clean.to_numpy()
    n_obs, n_vars = y.shape

    static_fit = VAR(clean).fit(maxlags=1, trend="c")
    initial_parameters = static_fit.params.to_numpy()
    regressors_per_equation = n_vars + 1

    beta_previous = initial_parameters.flatten(order="F")
    n_beta = len(beta_previous)
    state_covariance_previous = np.eye(n_beta) * initial_covariance
    sigma_previous = np.asarray(static_fit.sigma_u)

    beta_store = np.zeros((n_obs, n_beta))
    coefficient_store = np.zeros((n_obs, n_vars, n_vars))
    constant_store = np.zeros((n_obs, n_vars))
    sigma_store = np.zeros((n_obs, n_vars, n_vars))
    innovation_store = np.full((n_obs, n_vars), np.nan)

    beta_store[0] = beta_previous
    coefficient_store[0] = initial_parameters[1:, :].T
    constant_store[0] = initial_parameters[0, :]
    sigma_store[0] = sigma_previous

    identity_beta = np.eye(n_beta)

    for t in range(1, n_obs):
        observation_vector = np.concatenate(([1.0], y[t - 1]))
        observation_matrix = np.kron(
            np.eye(n_vars), observation_vector.reshape(1, -1)
        )

        beta_prediction = beta_previous.copy()
        state_covariance_prediction = (
            state_covariance_previous / lambda_beta
        )
        innovation = y[t] - observation_matrix @ beta_prediction
        forecast_covariance = (
            observation_matrix
            @ state_covariance_prediction
            @ observation_matrix.T
            + sigma_previous
            + np.eye(n_vars) * 1e-10
        )
        kalman_gain = (
            state_covariance_prediction
            @ observation_matrix.T
            @ np.linalg.inv(forecast_covariance)
        )
        beta_new = beta_prediction + kalman_gain @ innovation

        gain_observation = kalman_gain @ observation_matrix
        state_covariance_new = (
            (identity_beta - gain_observation)
            @ state_covariance_prediction
            @ (identity_beta - gain_observation).T
            + kalman_gain @ sigma_previous @ kalman_gain.T
        )
        state_covariance_new = (
            state_covariance_new + state_covariance_new.T
        ) / 2

        sigma_new = (
            kappa_sigma * sigma_previous
            + (1 - kappa_sigma) * np.outer(innovation, innovation)
        )
        sigma_new = (sigma_new + sigma_new.T) / 2
        sigma_new += np.eye(n_vars) * 1e-10

        parameter_matrix = beta_new.reshape(
            (regressors_per_equation, n_vars), order="F"
        )
        beta_store[t] = beta_new
        constant_store[t] = parameter_matrix[0, :]
        coefficient_store[t] = parameter_matrix[1:, :].T
        sigma_store[t] = sigma_new
        innovation_store[t] = innovation

        beta_previous = beta_new
        state_covariance_previous = state_covariance_new
        sigma_previous = sigma_new

    return TVPVARResult(
        dates=clean.index,
        variables=tuple(clean.columns),
        lambda_beta=lambda_beta,
        kappa_sigma=kappa_sigma,
        initial_covariance=initial_covariance,
        coefficients=coefficient_store,
        constants=constant_store,
        shock_covariances=sigma_store,
        innovations=innovation_store,
    )
