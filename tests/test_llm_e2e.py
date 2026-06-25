"""End-to-end test for LLM (Ollama) integration.

Prerequisites:
  - Ollama running at http://localhost:11434
  - Model configured in config.yaml (default: qwen3.5:9b-mlx) loaded

Usage:
  pytest tests/test_llm_e2e.py -v
  python -m pytest tests/test_llm_e2e.py -v --no-header
  python tests/test_llm_e2e.py  (standalone)
"""

import asyncio
import sys
import time

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from finlens.config import FinLensConfig, LLMConfig
from finlens.llm.analyst import (
    LLMResponse,
    build_prompt,
    call_ollama,
    parse_llm_response,
    analyze_ticker_llm,
)
from finlens.signals.engine import TickerAnalysis


def _make_fake_analysis(ticker: str = "SPY", market: str = "us", price: float = 700.0) -> TickerAnalysis:
    """Build a minimal TickerAnalysis with dummy data for testing."""
    from finlens.analytics.indicators import IndicatorSignals
    from finlens.analytics.capm import CAPMResult
    from finlens.analytics.garch import GARCHResult
    from finlens.analytics.risk import RiskMetrics

    return TickerAnalysis(
        ticker=ticker,
        market=market,
        price=price,
        price_change_pct=1.2,
        indicators=IndicatorSignals(
            sma_signal="BUY", rsi_signal="NEUTRAL", macd_signal="HOLD", bb_signal="SELL",
        ),
        capm=CAPMResult(beta=1.2, alpha=0.05, r_squared=0.85, p_value_alpha=0.01, p_value_beta=0.01, residuals_std=0.02, skewness=-0.1, kurtosis=3.2, signal="BUY"),
        garch=GARCHResult(omega=2.5e-6, alpha=0.08, gamma=0.02, beta=0.87, persistence=0.95, conditional_vol=0.01, annualized_vol=0.18, dof=5.0, signal="NEUTRAL"),
        risk=RiskMetrics(sharpe_ratio=0.65, sortino_ratio=0.82, calmar_ratio=0.42, max_drawdown=-0.153, max_drawdown_pct=-15.3, volatility_annual=0.22, signal="NEUTRAL"),
        composite_score=0.35,
        composite_signal="BUY",
    )


def _check_ollama_running() -> bool:
    """Quick check if Ollama is reachable."""
    if httpx is None:
        return False
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


async def test_e2e_prompt_building() -> None:
    """Verify build_prompt returns well-formed text."""
    a = _make_fake_analysis()
    prompt = build_prompt(a)
    assert a.ticker in prompt
    assert a.market.upper() in prompt
    assert "COMPOSITE:" in prompt
    assert "GJR-GARCH" in prompt
    assert "CAPM" in prompt
    assert "INDICATORS" in prompt
    print(f"[PASS] build_prompt ({len(prompt)} chars)")


async def test_e2e_parse_response_standard() -> None:
    """Parse a well-formed LLM response."""
    content = """Some reasoning text.
[NARRATIVE] Strong momentum with aligned indicators.
[CONFIDENCE] 0.78
[RECOMMENDATION] BUY"""
    resp = parse_llm_response(content, ticker="SPY", model="test")
    assert resp.ticker == "SPY"
    assert resp.recommendation == "BUY"
    assert abs(resp.confidence - 0.78) < 0.01
    assert "Strong momentum" in resp.narrative
    assert resp.model == "test"
    print(f"[PASS] parse_llm_response standard: {resp.recommendation} @ {resp.confidence}")


async def test_e2e_parse_response_minimal() -> None:
    """Parse a response with no markers — should use fallback."""
    content = "The stock shows strong buy signals with good momentum."
    resp = parse_llm_response(content, ticker="AAPL")
    assert resp.ticker == "AAPL"
    assert resp.recommendation == "HOLD"
    assert resp.confidence == 0.5
    assert len(resp.narrative) > 0
    print(f"[PASS] parse_llm_response fallback: {resp.narrative[:40]}...")


async def test_e2e_parse_response_partial() -> None:
    """Parse a response with only some markers."""
    content = "[NARRATIVE] Good momentum.\n[CONFIDENCE] bad-value\n"
    resp = parse_llm_response(content, ticker="QQQ")
    assert resp.recommendation == "HOLD"
    assert resp.confidence == 0.5
    assert resp.narrative == "Good momentum."
    print(f"[PASS] parse_llm_response partial" + f" {resp.narrative}")


async def test_e2e_ollama_call() -> None:
    """Actually call the running Ollama instance and parse the response."""
    if not _check_ollama_running():
        print("[SKIP] Ollama not running")
        return

    cfg = FinLensConfig.load("config.yaml")
    content = await call_ollama("Say 'ok' in one word.", model=cfg.llm.model, base_url=cfg.llm.api_url)
    assert content, "Empty response from Ollama"
    print(f"[PASS] call_ollama raw: {content.strip()}")


async def test_e2e_analyze_ticker() -> None:
    """Full end-to-end: build prompt → call Ollama → parse response."""
    if not _check_ollama_running():
        print("[SKIP] Ollama not running")
        return

    cfg = FinLensConfig.load("config.yaml")
    a = _make_fake_analysis(ticker="SPY")
    resp = await analyze_ticker_llm(a, cfg.llm)
    assert resp.ticker == "SPY"
    assert resp.recommendation in ("BUY", "SELL", "HOLD", "N/A")
    assert 0.0 <= resp.confidence <= 1.0
    assert len(resp.narrative) > 0
    print(f"[PASS] analyze_ticker_llm e2e: {resp.ticker} {resp.recommendation} @ {resp.confidence:.2f}")


async def test_e2e_ollama_with_full_prompt() -> None:
    """End-to-end with a realistic financial prompt, verify marker parsing."""
    if not _check_ollama_running():
        print("[SKIP] Ollama not running")
        return

    cfg = FinLensConfig.load("config.yaml")
    a = _make_fake_analysis(ticker="NVDA", price=820.0)
    prompt = build_prompt(a)
    raw = await call_ollama(prompt, model=cfg.llm.model, base_url=cfg.llm.api_url)
    resp = parse_llm_response(raw, ticker=a.ticker, model=cfg.llm.model)
    assert resp.recommendation in ("BUY", "SELL", "HOLD")
    print(f"[PASS] full prompt e2e: {resp.ticker} {resp.recommendation} @ {resp.confidence:.2f}")
    print(f"  Narrative: {resp.narrative[:120]}...")


async def test_e2e_analyze_batch_llm() -> None:
    """Test batch analysis with multiple tickers."""
    if not _check_ollama_running():
        print("[SKIP] Ollama not running")
        return

    cfg = FinLensConfig.load("config.yaml")
    analyses = [
        _make_fake_analysis("SPY", "us", 700.0),
        _make_fake_analysis("QQQ", "us", 520.0),
    ]
    from finlens.llm.analyst import analyze_batch_llm
    result = await analyze_batch_llm(analyses, cfg.llm)
    assert len(result.responses) == 2
    for resp in result.responses:
        assert resp.ticker in ("SPY", "QQQ")
        assert len(resp.narrative) > 0
    print(f"[PASS] analyze_batch_llm: {len(result.responses)} responses")


async def test_e2e_ollama_models_list() -> None:
    """Test listing models via subprocess."""
    from finlens.llm.analyst import ollama_list_models
    models = ollama_list_models()
    if not models:
        print("[SKIP] ollama list returned empty")
        return
    names = [m["name"] for m in models]
    print(f"[PASS] ollama_list_models ({len(models)} models): {', '.join(names[:5])}...")


async def test_e2e_llm_disabled() -> None:
    """When LLM is disabled, return N/A immediately."""
    cfg = FinLensConfig.load("config.yaml")
    disabled = LLMConfig(enabled=False)
    a = _make_fake_analysis("TEST")
    resp = await analyze_ticker_llm(a, disabled)
    assert resp.recommendation == "N/A"
    assert resp.confidence == 0.0
    print(f"[PASS] LLM disabled: {resp.recommendation}")


async def main() -> None:
    tests = [
        ("Prompt building", test_e2e_prompt_building),
        ("Parse standard", test_e2e_parse_response_standard),
        ("Parse minimal", test_e2e_parse_response_minimal),
        ("Parse partial", test_e2e_parse_response_partial),
        ("Ollama raw call", test_e2e_ollama_call),
        ("Full prompt e2e", test_e2e_ollama_with_full_prompt),
        ("Analyze ticker e2e", test_e2e_analyze_ticker),
        ("Batch analyze", test_e2e_analyze_batch_llm),
        ("List models", test_e2e_ollama_models_list),
        ("LLM disabled", test_e2e_llm_disabled),
    ]
    passed = 0
    failed = 0
    skipped = 0
    print(f"\n{'='*60}")
    print(f"  FinLens — LLM End-to-End Tests")
    print(f"{'='*60}\n")
    for name, coro in tests:
        try:
            t0 = time.time()
            await coro()
            elapsed = time.time() - t0
            print(f"  ✓ {name} ({elapsed:.1f}s)")
            passed += 1
        except Exception as e:
            elapsed = time.time() - t0 if "t0" in dir() else 0
            print(f"  ✗ {name} — {e} ({elapsed:.1f}s)")
            failed += 1
    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
