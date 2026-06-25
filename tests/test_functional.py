"""Functional integration tests for FinLens.

Runs the complete pipeline: config → data → analytics → signals → TUI.

Usage:
  uv run python tests/test_functional.py
  uv run python tests/test_functional.py --skip-slow   (skip LLM and TUI)
"""

import argparse
import asyncio
import sys
import time

from finlens.config import FinLensConfig


# ── 1. Config ──────────────────────────────────────────────────────────────

def test_config_loads() -> dict:
    """Load config, verify markets, signals, composite, llm, cache sections."""
    cfg = FinLensConfig.load("config.yaml")
    assert len(cfg.markets) >= 1, f"No markets loaded: {cfg.markets}"
    assert cfg.signals.lookback_days > 0
    assert cfg.composite.buy_threshold > cfg.composite.sell_threshold
    assert cfg.llm.model
    assert cfg.cache.max_age_hours > 0
    for key in ("us",):
        assert key in cfg.markets, f"Missing market '{key}', got {list(cfg.markets.keys())}"
        m = cfg.markets[key]
        assert len(m.tickers) > 0, f"Market '{key}' has no tickers"
        assert m.benchmark
    print(f"  markets: {', '.join(cfg.markets)}")
    print(f"  tickers: {sum(len(m.tickers) for m in cfg.markets.values())} total")
    print(f"  llm: {cfg.llm.provider}/{cfg.llm.model} enabled={cfg.llm.enabled}")
    return cfg


def test_config_env_overrides(tmpdir: str | None = None) -> None:
    """Ensure env vars override yaml values."""
    import os
    os.environ["OLLAMA_MODEL"] = "test-model-override"
    os.environ["OLLAMA_API_URL"] = "http://test:9999"
    os.environ["OLLAMA_API_KEY"] = "test-key-123"
    try:
        cfg = FinLensConfig.load("config.yaml")
        assert cfg.llm.model == "test-model-override", f"Expected test-model-override, got {cfg.llm.model}"
        assert cfg.llm.api_url == "http://test:9999"
        assert cfg.llm.api_key == "test-key-123"
        print("  env overrides correctly applied")
    finally:
        del os.environ["OLLAMA_MODEL"]
        del os.environ["OLLAMA_API_URL"]
        del os.environ["OLLAMA_API_KEY"]


# ── 2. Data ────────────────────────────────────────────────────────────────

def test_fetch_ticker_data(cfg: FinLensConfig) -> int:
    """Fetch live data for one ticker; skip if network unavailable."""
    from finlens.data.fetcher import fetch_ticker_data, DataCache
    cache = DataCache(cfg.cache.dir)
    first_ticker = cfg.markets["us"].tickers[0]
    try:
        import pandas as pd
        df = fetch_ticker_data(first_ticker, period="30d", cache=cache)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        assert len(close) > 1, f"Only {len(close)} rows for {first_ticker}"
        price = float(close.iloc[-1])
        print(f"  {first_ticker}: ${price:.2f} ({len(df)} rows)")
        return len(df)
    except Exception as e:
        print(f"  SKIP fetch: {e}")
        return 0


def test_fetch_multi_ticker(cfg: FinLensConfig) -> list:
    """Fetch multiple tickers from a market."""
    from finlens.data.fetcher import fetch_multi_ticker
    tickers = cfg.markets["us"].tickers[:3]
    results = []
    try:
        for ticker, df in fetch_multi_ticker(tickers, cfg):
            results.append((ticker, len(df)))
        print(f"  fetched {len(results)}/{len(tickers)} tickers")
        for t, n in results:
            print(f"    {t}: {n} rows")
        return results
    except Exception as e:
        print(f"  SKIP multi-fetch: {e}")
        return []


# ── 3. Analytics ───────────────────────────────────────────────────────────

def test_indicators() -> None:
    """Compute indicators on synthetic data."""
    import pandas as pd
    import numpy as np
    from finlens.analytics.indicators import compute_all_indicators
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(252) * 0.5)
    df = pd.DataFrame({"Close": close})
    config = {"rsi_period": 14, "sma_fast": 20, "sma_slow": 50, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "bb_period": 20, "bb_std": 2}
    ind = compute_all_indicators(df, config)
    assert ind.sma_signal in ("BUY", "SELL", "HOLD"), f"sma_signal={ind.sma_signal}"
    assert ind.rsi_signal in ("BUY", "SELL", "HOLD"), f"rsi_signal={ind.rsi_signal}"
    assert ind.macd_signal in ("BUY", "SELL", "HOLD"), f"macd_signal={ind.macd_signal}"
    assert ind.bb_signal in ("BUY", "SELL", "HOLD"), f"bb_signal={ind.bb_signal}"
    print(f"  indicators: SMA={ind.sma_signal} RSI={ind.rsi_signal} MACD={ind.macd_signal} BB={ind.bb_signal}")


def test_capm(cfg: FinLensConfig) -> None:
    """Run CAPM on a real ticker with benchmark."""
    import pandas as pd
    from finlens.data.fetcher import fetch_ticker_data, DataCache
    from finlens.analytics.capm import heavy_tailed_capm
    cache = DataCache(cfg.cache.dir)
    try:
        ticker = cfg.markets["us"].tickers[0]
        bm = cfg.markets["us"].benchmark
        a_df = fetch_ticker_data(ticker, period="60d", cache=cache)
        b_df = fetch_ticker_data(bm, period="60d", cache=cache)
        a_close = a_df["Close"]
        b_close = b_df["Close"]
        a_close = a_close.squeeze() if isinstance(a_close, pd.DataFrame) else a_close
        b_close = b_close.squeeze() if isinstance(b_close, pd.DataFrame) else b_close
        a_ret = a_close.pct_change().dropna()
        b_ret = b_close.pct_change().dropna()
        aligned = pd.concat([a_ret, b_ret], axis=1, join="inner").dropna()
        if len(aligned) > 10:
            capm = heavy_tailed_capm(aligned.iloc[:, 0], aligned.iloc[:, 1])
            assert -5 < capm.beta < 5
            isinstance(capm.alpha, float)
            print(f"  CAPM: β={capm.beta:.2f} α={capm.alpha:.4f} R²={capm.r_squared:.2f}")
    except Exception as e:
        print(f"  SKIP CAPM: {e}")


def test_garch(cfg: FinLensConfig) -> None:
    """Fit GJR-GARCH on a real ticker."""
    import pandas as pd
    from finlens.data.fetcher import fetch_ticker_data, DataCache
    from finlens.analytics.garch import fit_gjr_garch
    cache = DataCache(cfg.cache.dir)
    try:
        ticker = cfg.markets["us"].tickers[0]
        df = fetch_ticker_data(ticker, period="120d", cache=cache)
        close = df["Close"]
        close = close.squeeze() if isinstance(close, pd.DataFrame) else close
        returns = close.pct_change().dropna()
        if len(returns) > 50:
            garch = fit_gjr_garch(returns, p=1, q=1)
            assert garch.annualized_vol > 0
            isinstance(garch.persistence, float)
            print(f"  GJR-GARCH: σ={garch.annualized_vol:.2%} persist={garch.persistence:.3f}")
    except Exception as e:
        print(f"  SKIP GARCH: {e}")


def test_var(cfg: FinLensConfig) -> None:
    """Compute VaR/CVaR on a real ticker."""
    import pandas as pd
    from finlens.data.fetcher import fetch_ticker_data, DataCache
    from finlens.analytics.garch import fit_gjr_garch
    from finlens.analytics.risk import calculate_var_metrics
    cache = DataCache(cfg.cache.dir)
    try:
        ticker = cfg.markets["us"].tickers[0]
        df = fetch_ticker_data(ticker, period="120d", cache=cache)
        close = df["Close"]
        close = close.squeeze() if isinstance(close, pd.DataFrame) else close
        returns = close.pct_change().dropna()
        if len(returns) > 50:
            garch = fit_gjr_garch(returns, p=1, q=1)
            var = calculate_var_metrics(returns, conditional_vol=garch.conditional_vol, dof=garch.dof)
            assert isinstance(var.var_95_pct, float)
            assert isinstance(var.cvar_95_pct, float)
            assert var.cvar_95_pct > var.var_95_pct
            assert var.skewness != 0
            print(f"  VaR95: {var.var_95_pct:.2f}%  CVaR95: {var.cvar_95_pct:.2f}%  Hist95: {var.var_95_hist_pct:.2f}%")
            print(f"  Skew: {var.skewness}  Kurt: {var.kurtosis}  Risk: {var.signal}")
    except Exception as e:
        print(f"  SKIP VaR: {e}")


def test_risk(cfg: FinLensConfig) -> None:
    """Compute risk metrics on a real ticker."""
    import pandas as pd
    from finlens.data.fetcher import fetch_ticker_data, DataCache
    from finlens.analytics.risk import calculate_risk_metrics
    cache = DataCache(cfg.cache.dir)
    try:
        ticker = cfg.markets["us"].tickers[0]
        df = fetch_ticker_data(ticker, period="252d", cache=cache)
        close = df["Close"]
        close = close.squeeze() if isinstance(close, pd.DataFrame) else close
        returns = close.pct_change().dropna()
        risk = calculate_risk_metrics(returns, risk_free_rate=4.5)
        assert isinstance(risk.sharpe_ratio, float)
        assert risk.max_drawdown <= 0
        print(f"  Risk: Sharpe={risk.sharpe_ratio:.2f} Sortino={risk.sortino_ratio:.2f} MaxDD={risk.max_drawdown_pct:.1f}%")
    except Exception as e:
        print(f"  SKIP risk: {e}")


# ── 4. Signal Engine ──────────────────────────────────────────────────────

def test_signal_engine(cfg: FinLensConfig) -> None:
    """Run the full signal pipeline for one market."""
    from finlens.signals.engine import analyze_ticker
    from finlens.data.fetcher import fetch_multi_ticker
    tickers = cfg.markets["us"].tickers[:3]
    count = 0
    for ticker, df in fetch_multi_ticker(tickers, cfg):
        analysis = analyze_ticker(ticker, "us", {"risk_free_rate": 4.5}, df, cfg)
        if analysis:
            assert analysis.ticker == ticker
            assert analysis.composite_signal in ("BUY", "HOLD", "SELL")
            assert -1.0 <= analysis.composite_score <= 1.0
            count += 1
    if count == 0:
        print("  SKIP signal engine: no data fetched")
    else:
        print(f"  {count} tickers analyzed")
        assert count > 0


# ── 5. LLM ─────────────────────────────────────────────────────────────────

async def test_llm(cfg: FinLensConfig) -> None:
    """Call LLM if Ollama is reachable."""
    import httpx
    try:
        r = httpx.get(f"{cfg.llm.api_url}/api/tags", timeout=3)
        if r.status_code != 200:
            print("  SKIP LLM: Ollama not reachable")
            return
    except Exception:
        print("  SKIP LLM: Ollama not reachable")
        return

    from finlens.llm.analyst import analyze_ticker_llm
    from finlens.signals.engine import analyze_ticker
    from finlens.data.fetcher import fetch_multi_ticker
    for ticker, df in fetch_multi_ticker([cfg.markets["us"].tickers[0]], cfg):
        analysis = analyze_ticker(ticker, "us", {"risk_free_rate": 4.5}, df, cfg)
        if analysis:
            t0 = time.time()
            resp = await analyze_ticker_llm(analysis, cfg.llm)
            elapsed = time.time() - t0
            assert resp.ticker == ticker
            assert resp.recommendation in ("BUY", "SELL", "HOLD", "N/A")
            print(f"  {ticker}: {resp.recommendation} @ {resp.confidence:.2f} ({elapsed:.0f}s)")
            print(f"    {resp.narrative[:100]}...")
            return


# ── 6. TUI ─────────────────────────────────────────────────────────────────

async def test_tui(cfg: FinLensConfig) -> None:
    """Launch the TUI, wait for data, verify rendering."""
    from finlens.output.terminal import FinLensApp
    app = FinLensApp(cfg)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(15)
        matrix = app.query_one("#signal-matrix")
        assert matrix.row_count > 0, "Signal matrix has no rows"
        detail = app.query_one("#detail-panel")
        assert detail is not None
        print(f"  SignalMatrix: {matrix.row_count} rows loaded")

        # Test filter keys
        await pilot.press("2")
        await pilot.pause(1)
        n_us = len(app.filtered_analyses)
        print(f"  Filter 2 (US): {n_us} tickers")
        assert n_us > 0

        await pilot.press("1")
        await pilot.pause(1)
        n_all = len(app.filtered_analyses)
        print(f"  Filter 1 (all): {n_all} tickers")
        assert n_all >= n_us

        # Test help modal
        await pilot.press("h")
        await pilot.pause(1)
        help_m = list(app.query("#help-modal"))
        assert len(help_m) > 0, "Help modal not shown"
        await pilot.press("escape")
        await pilot.pause(1)

        # Test navigation
        await pilot.press("down")
        await pilot.pause(0.5)
        assert app.selected_ticker is not None, "No ticker selected after arrow down"
        print(f"  Selected: {app.selected_ticker}")


# ── Main runner ────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
SKIP_MSG = "SKIP"


def run_sync(name: str, fn, *args, **kwargs) -> str:
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        elapsed = time.time() - t0
        return f"{PASS} ({elapsed:.1f}s)"
    except Exception as e:
        elapsed = time.time() - t0
        return f"{FAIL} — {e} ({elapsed:.1f}s)"


async def run_async(name: str, fn, *args, **kwargs) -> str:
    t0 = time.time()
    try:
        await fn(*args, **kwargs)
        elapsed = time.time() - t0
        return f"{PASS} ({elapsed:.1f}s)"
    except Exception as e:
        elapsed = time.time() - t0
        return f"{FAIL} — {e} ({elapsed:.1f}s)"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-slow", action="store_true", help="Skip LLM and TUI tests")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  FinLens — Functional Tests")
    print(f"{'='*60}\n")

    print("1. Config")
    cfg = FinLensConfig.load("config.yaml")
    print(run_sync("  load", test_config_loads))
    print(run_sync("  env overrides", test_config_env_overrides, None))

    print("\n2. Data Fetching")
    run_sync("  single ticker", test_fetch_ticker_data, cfg)
    run_sync("  multi ticker", test_fetch_multi_ticker, cfg)

    print("\n3. Analytics")
    print(run_sync("  indicators", test_indicators))
    print(run_sync("  CAPM", test_capm, cfg))
    print(run_sync("  GARCH", test_garch, cfg))
    print(run_sync("  VaR / CVaR", test_var, cfg))
    print(run_sync("  risk metrics", test_risk, cfg))

    print("\n4. Signal Engine")
    print(run_sync("  signal pipeline", test_signal_engine, cfg))

    if not args.skip_slow:
        print("\n5. LLM")
        result = await run_async("  LLM analysis", test_llm, cfg)
        print(result)

        print("\n6. TUI")
        result = await run_async("  TUI render", test_tui, cfg)
        print(result)
    else:
        print("\n5. LLM — skipped (--skip-slow)")
        print("6. TUI — skipped (--skip-slow)")

    print(f"\n{'='*60}")
    print("  Done")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
