from dataclasses import dataclass, field

import pandas as pd

from finlens.analytics.capm import CAPMResult, heavy_tailed_capm
from finlens.analytics.garch import GARCHResult, fit_gjr_garch
from finlens.analytics.indicators import IndicatorSignals, compute_all_indicators
from finlens.analytics.risk import RiskMetrics, VaRResult, calculate_risk_metrics, calculate_var_metrics
from finlens.config import FinLensConfig


@dataclass
class TickerAnalysis:
    ticker: str
    market: str
    price: float
    price_change_pct: float
    indicators: IndicatorSignals
    capm: CAPMResult | None
    garch: GARCHResult
    risk: RiskMetrics
    var: VaRResult | None = None
    composite_score: float = 0.0
    composite_signal: str = "HOLD"


def signal_to_score(signal: str) -> float:
    if signal == "BUY":
        return 1.0
    elif signal == "SELL":
        return -1.0
    return 0.0


def score_to_signal(score: float, config: FinLensConfig) -> str:
    if score >= config.composite.buy_threshold:
        return "BUY"
    elif score <= config.composite.sell_threshold:
        return "SELL"
    return "HOLD"


def compute_composite_score(
    ind: IndicatorSignals,
    capm: CAPMResult | None,
    risk: RiskMetrics,
    config: FinLensConfig,
) -> tuple[float, str]:
    w = config.composite.weights

    score = (
        w.get("sma", 0.15) * signal_to_score(ind.sma_signal)
        + w.get("rsi", 0.20) * signal_to_score(ind.rsi_signal)
        + w.get("macd", 0.15) * signal_to_score(ind.macd_signal)
        + w.get("bb", 0.15) * signal_to_score(ind.bb_signal)
    )

    if capm:
        capm_score = 1.0 if capm.signal == "BUY" else -1.0 if capm.signal == "SELL" else 0.0
        score += w.get("capm_alpha", 0.15) * capm_score

    risk_score = 1.0 if risk.signal == "BUY" else -1.0 if risk.signal == "SELL" else 0.0
    score += w.get("risk_ratio", 0.20) * risk_score

    return round(score, 3), score_to_signal(score, config)


def analyze_ticker(
    ticker: str,
    market_key: str,
    market_config: dict,
    df: pd.DataFrame,
    config: FinLensConfig,
    benchmark_returns: pd.Series | None = None,
) -> TickerAnalysis | None:
    if df.empty or len(df) < 60:
        return None

    close = df["Close"].squeeze()
    price = float(close.iloc[-1])
    price_change_pct = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0.0

    returns = close.pct_change().dropna()

    lookback = min(config.signals.lookback_days, len(returns))
    recent_returns = returns.iloc[-lookback:]

    indicators = compute_all_indicators(df, config.signals.model_dump())

    capm_result: CAPMResult | None = None
    if benchmark_returns is not None:
        try:
            capm_result = heavy_tailed_capm(
                recent_returns,
                benchmark_returns,
                risk_free_rate=market_config.get("risk_free_rate", 0),
            )
        except Exception:
            capm_result = None

    garch_result = fit_gjr_garch(recent_returns, p=config.signals.garch_p, q=config.signals.garch_q)

    var_result = calculate_var_metrics(
        recent_returns,
        conditional_vol=garch_result.conditional_vol,
        dof=garch_result.dof,
    )

    risk_result = calculate_risk_metrics(
        recent_returns,
        risk_free_rate=market_config.get("risk_free_rate", 0),
    )

    comp_score, comp_signal = compute_composite_score(
        indicators,
        capm_result,
        risk_result,
        config,
    )

    return TickerAnalysis(
        ticker=ticker,
        market=market_key,
        price=round(price, 2),
        price_change_pct=round(price_change_pct, 2),
        indicators=indicators,
        capm=capm_result,
        garch=garch_result,
        var=var_result,
        risk=risk_result,
        composite_score=comp_score,
        composite_signal=comp_signal,
    )
