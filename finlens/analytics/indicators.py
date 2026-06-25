from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class IndicatorSignals:
    sma_signal: str
    rsi_signal: str
    macd_signal: str
    bb_signal: str


def calculate_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period).mean()


def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def get_sma_signal(
    close: pd.Series,
    fast_period: int = 20,
    slow_period: int = 50,
) -> str:
    sma_fast = calculate_sma(close, fast_period)
    sma_slow = calculate_sma(close, slow_period)
    if pd.isna(sma_fast.iloc[-1]) or pd.isna(sma_slow.iloc[-1]):
        return "HOLD"
    if sma_fast.iloc[-1] > sma_slow.iloc[-1] and sma_fast.iloc[-2] <= sma_slow.iloc[-2]:
        return "BUY"
    elif sma_fast.iloc[-1] < sma_slow.iloc[-1] and sma_fast.iloc[-2] >= sma_slow.iloc[-2]:
        return "SELL"
    elif sma_fast.iloc[-1] > sma_slow.iloc[-1]:
        return "BUY"
    elif sma_fast.iloc[-1] < sma_slow.iloc[-1]:
        return "SELL"
    return "HOLD"


def get_rsi_signal(rsi: pd.Series, oversold: float = 30, overbought: float = 70) -> str:
    val = rsi.iloc[-1]
    if pd.isna(val):
        return "HOLD"
    if val < oversold:
        return "BUY"
    elif val > overbought:
        return "SELL"
    return "HOLD"


def get_macd_signal(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> str:
    macd_line, signal_line, hist = calculate_macd(close, fast, slow, signal)
    if pd.isna(macd_line.iloc[-1]) or pd.isna(signal_line.iloc[-1]):
        return "HOLD"
    if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
        return "BUY"
    elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
        return "SELL"
    elif macd_line.iloc[-1] > signal_line.iloc[-1]:
        return "BUY"
    elif macd_line.iloc[-1] < signal_line.iloc[-1]:
        return "SELL"
    return "HOLD"


def get_bb_signal(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> str:
    upper, mid, lower = calculate_bollinger_bands(close, period, std_dev)
    current = close.iloc[-1]
    if pd.isna(upper.iloc[-1]):
        return "HOLD"
    if current <= lower.iloc[-1]:
        return "BUY"
    elif current >= upper.iloc[-1]:
        return "SELL"
    return "HOLD"


def compute_all_indicators(
    df: pd.DataFrame,
    config: dict,
) -> IndicatorSignals:
    close = df["Close"].squeeze()

    sma_sig = get_sma_signal(
        close,
        fast_period=config.get("sma_fast", 20),
        slow_period=config.get("sma_slow", 50),
    )
    rsi = calculate_rsi(close, period=config.get("rsi_period", 14))
    rsi_sig = get_rsi_signal(
        rsi,
        oversold=config.get("rsi_oversold", 30),
        overbought=config.get("rsi_overbought", 70),
    )
    macd_sig = get_macd_signal(
        close,
        fast=config.get("macd_fast", 12),
        slow=config.get("macd_slow", 26),
        signal=config.get("macd_signal", 9),
    )
    bb_sig = get_bb_signal(
        close,
        period=config.get("bb_period", 20),
        std_dev=config.get("bb_std", 2.0),
    )

    return IndicatorSignals(
        sma_signal=sma_sig,
        rsi_signal=rsi_sig,
        macd_signal=macd_sig,
        bb_signal=bb_sig,
    )
