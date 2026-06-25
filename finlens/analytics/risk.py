from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, norm, skew as scipy_skew, kurtosis as scipy_kurt


@dataclass
class RiskMetrics:
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    volatility_annual: float
    signal: str


def calculate_risk_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> RiskMetrics:
    r = returns.dropna()
    if len(r) < 30:
        return RiskMetrics(
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            volatility_annual=0,
            signal="HOLD",
        )

    rf = risk_free_rate / 100 / 252
    excess_returns = r - rf

    ann_return = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)

    sharpe = (ann_return - risk_free_rate / 100) / ann_vol if ann_vol != 0 else 0

    downside_returns = r[r < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.01
    sortino = (ann_return - risk_free_rate / 100) / downside_std if downside_std != 0 else 0

    cumulative = (1 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    max_dd_pct = max_dd * 100

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    if sharpe > 1.5 and max_dd_pct > -15:
        signal = "BUY"
    elif sharpe < 0.3:
        signal = "SELL"
    else:
        signal = "HOLD"

    return RiskMetrics(
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        calmar_ratio=round(calmar, 2),
        max_drawdown=round(max_dd, 4),
        max_drawdown_pct=round(max_dd_pct, 2),
        volatility_annual=round(ann_vol, 4),
        signal=signal,
    )


@dataclass
class VaRResult:
    var_95_pct: float
    cvar_95_pct: float
    var_95_hist_pct: float
    skewness: float
    kurtosis: float
    signal: str


def calculate_var_metrics(
    returns: pd.Series,
    conditional_vol: float = 0.0,
    dof: float = 5.0,
    alpha: float = 0.05,
) -> VaRResult:
    r = returns.dropna()
    if len(r) < 20 or conditional_vol <= 0:
        return VaRResult(
            var_95_pct=0,
            cvar_95_pct=0,
            var_95_hist_pct=0,
            skewness=0,
            kurtosis=0,
            signal="N/A",
        )

    cond_vol = conditional_vol  # daily decimal vol
    skew = float(scipy_skew(r))
    kurt = float(scipy_kurt(r, fisher=True))

    if dof > 2:
        t_inv = float(t_dist.ppf(alpha, dof))
        var_param = -cond_vol * t_inv
        g_t = float(t_dist.pdf(t_inv, dof))
        cvar_param = cond_vol * (g_t / alpha) * ((dof + t_inv**2) / (dof - 1))
    else:
        z = float(norm.ppf(alpha))
        var_param = -cond_vol * z
        cvar_param = cond_vol * (float(norm.pdf(z)) / alpha)

    var_hist = -float(r.quantile(alpha))

    if cvar_param < 0.015:
        signal = "LOW RISK"
    elif cvar_param > 0.035:
        signal = "HIGH RISK"
    else:
        signal = "MED RISK"

    return VaRResult(
        var_95_pct=round(var_param * 100, 2),
        cvar_95_pct=round(cvar_param * 100, 2),
        var_95_hist_pct=round(var_hist * 100, 2),
        skewness=round(skew, 3),
        kurtosis=round(kurt, 3),
        signal=signal,
    )
