import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import yfinance as yf

from finlens.config import FinLensConfig


class DataCache:
    def __init__(self, cache_dir: str = ".finlens_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _cache_path(self, ticker: str) -> Path:
        safe = ticker.replace("^", "_").replace(".", "_")
        return self.cache_dir / f"{safe}.parquet"

    def get(self, ticker: str, max_age_hours: int = 24) -> pd.DataFrame | None:
        path = self._cache_path(ticker)
        if not path.exists():
            return None
        age_hours = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 3600
        if age_hours > max_age_hours:
            return None
        return pd.read_parquet(path)

    def set(self, ticker: str, df: pd.DataFrame) -> None:
        path = self._cache_path(ticker)
        df.to_parquet(path)

    def clear(self) -> None:
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()


def fetch_ticker_data(
    ticker: str,
    period: str = "2y",
    cache: DataCache | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if cache and not force_refresh:
        cached = cache.get(ticker)
        if cached is not None:
            return cached

    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    df = df.sort_index()

    if cache:
        cache.set(ticker, df)

    return df


def fetch_multi_ticker(
    tickers: list[str],
    config: FinLensConfig,
    force_refresh: bool = False,
) -> Iterator[tuple[str, pd.DataFrame]]:
    cache = DataCache(config.cache.dir)
    lookback = config.signals.lookback_days + 60

    for ticker in tickers:
        try:
            df = fetch_ticker_data(
                ticker,
                period=f"{lookback}d",
                cache=cache,
                force_refresh=force_refresh,
            )
            yield ticker, df
        except Exception as e:
            print(f"Warning: failed to fetch {ticker}: {e}")
            continue
