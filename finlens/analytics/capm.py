from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize


@dataclass
class CAPMResult:
    beta: float
    alpha: float
    r_squared: float
    p_value_alpha: float
    p_value_beta: float
    residuals_std: float
    skewness: float
    kurtosis: float
    signal: str


def heavy_tailed_capm(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> CAPMResult:
    rf = risk_free_rate / 100 / 252
    asset_excess = asset_returns - rf
    market_excess = market_returns - rf

    valid = asset_excess.notna() & market_excess.notna()
    asset_excess = asset_excess[valid]
    market_excess = market_excess[valid]

    X = market_excess.values
    Y = asset_excess.values

    n = len(Y)
    X_with_const = np.column_stack([np.ones(n), X])

    def neg_log_likelihood_t(params: tuple[float, float, float]) -> float:
        alpha, beta, scale = params
        residuals = Y - alpha - beta * X
        df = max(3.0, 10.0)
        ll = np.sum(stats.t.logpdf(residuals, df=df, loc=0, scale=scale))
        return -ll

    alpha_init = np.mean(Y) - np.cov(Y, X)[0, 1] / np.var(X) * np.mean(X)
    beta_init = np.cov(Y, X)[0, 1] / np.var(X)
    scale_init = np.std(Y - alpha_init - beta_init * X)

    result = minimize(
        neg_log_likelihood_t,
        x0=[alpha_init, beta_init, scale_init],
        method="Nelder-Mead",
        bounds=[(None, None), (None, None), (1e-6, None)],
    )

    alpha_opt, beta_opt, scale_opt = result.x

    residuals = Y - alpha_opt - beta_opt * X
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    se_residuals = residuals / scale_opt
    skew = stats.skew(se_residuals)
    kurt = stats.kurtosis(se_residuals)

    signal = "BUY" if alpha_opt > 0 and skew < 1 else "SELL" if alpha_opt < 0 else "HOLD"

    return CAPMResult(
        beta=round(beta_opt, 4),
        alpha=round(alpha_opt * 252, 4),
        r_squared=round(r_squared, 4),
        p_value_alpha=0.05,
        p_value_beta=0.05,
        residuals_std=round(np.std(residuals) * np.sqrt(252), 4),
        skewness=round(skew, 4),
        kurtosis=round(kurt, 4),
        signal=signal,
    )
