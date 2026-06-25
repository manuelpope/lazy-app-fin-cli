import json
import os
import subprocess
from dataclasses import dataclass, field

import httpx

from finlens.config import LLMConfig
from finlens.signals.engine import TickerAnalysis


@dataclass
class LLMResponse:
    ticker: str = ""
    narrative: str = ""
    confidence: float = 0.5
    recommendation: str = "HOLD"
    model: str = ""


@dataclass
class BatchLLMResult:
    responses: list[LLMResponse] = field(default_factory=list)
    model: str = ""

SYSTEM_PROMPT = """You are a quantitative financial analyst specializing in emerging markets.
Analyze stock signals, CAPM regressions, GJR-GARCH volatility models, and risk metrics.
Be concise. Focus on: indicator alignment, risk-adjusted returns, volatility regime, CAPM alpha.
End with exactly:
[NARRATIVE] <2-3 sentence analysis>
[CONFIDENCE] <0.0-1.0>
[RECOMMENDATION] BUY / HOLD / SELL"""


def build_prompt(analysis: TickerAnalysis) -> str:
    ind, capm, garch, risk = analysis.indicators, analysis.capm, analysis.garch, analysis.risk
    capm_block = f"β={capm.beta}, α(ann)={capm.alpha}, R²={capm.r_squared}" if capm else "N/A"

    return f"""Analyze {analysis.ticker} ({analysis.market.upper()})
Price: ${analysis.price} ({analysis.price_change_pct:+.2f}%)

INDICATORS: SMA={ind.sma_signal} RSI={ind.rsi_signal} MACD={ind.macd_signal} BB={ind.bb_signal}
CAPM: {capm_block}
GJR-GARCH: σ={garch.annualized_vol*100:.1f}% persist={garch.persistence}
RISK: Sharpe={risk.sharpe_ratio} Sortino={risk.sortino_ratio} MaxDD={risk.max_drawdown_pct:.1f}%
COMPOSITE: {analysis.composite_score:+.2f} ({analysis.composite_signal})"""


def build_batch_prompt(analyses: list[TickerAnalysis]) -> str:
    lines = []
    for a in analyses:
        prompt = build_prompt(a)
        lines.append(prompt)
    joined = "\n\n---\n\n".join(lines)
    return f"Analyze these tickers and give a brief recommendation for each:\n\n{joined}"


def ollama_list_models() -> list[dict]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        models = []
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if parts:
                models.append({"name": parts[0], "size": parts[2] if len(parts) > 2 else "?"})
        return models
    except Exception:
        return []


def ollama_api_models(base_url: str) -> list[str]:
    try:
        import httpx
        r = httpx.get(f"{base_url}/api/tags", timeout=5)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


async def call_ollama(
    prompt: str,
    model: str = "qwen3.5:9b-mlx",
    base_url: str = "http://localhost:11434",
    api_key: str = "",
) -> str:
    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    msg = data["choices"][0]["message"]
    content = msg.get("content", "") or msg.get("reasoning", "")
    return content


def parse_llm_response(content: str, ticker: str = "", model: str = "") -> LLMResponse:
    narrative = ""
    confidence = 0.5
    recommendation = "HOLD"

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("[NARRATIVE]"):
            narrative = line.replace("[NARRATIVE]", "").strip()
        elif line.startswith("[CONFIDENCE]"):
            try:
                confidence = float(line.replace("[CONFIDENCE]", "").strip())
            except ValueError:
                pass
        elif line.startswith("[RECOMMENDATION]"):
            recommendation = line.replace("[RECOMMENDATION]", "").strip().upper()

    if not narrative:
        narrative = content[:400]

    return LLMResponse(
        ticker=ticker,
        narrative=narrative,
        confidence=confidence,
        recommendation=recommendation,
        model=model,
    )


def parse_batch_response(content: str, tickers: list[str], model: str = "") -> list[LLMResponse]:
    responses = []
    for ticker in tickers:
        resp = parse_llm_response(content, ticker=ticker, model=model)
        responses.append(resp)
    if not responses:
        for ticker in tickers:
            responses.append(LLMResponse(ticker=ticker, narrative=content[:300], confidence=0.5, recommendation="HOLD", model=model))
    return responses


async def analyze_ticker_llm(
    analysis: TickerAnalysis,
    config: LLMConfig,
) -> LLMResponse:
    if not config.enabled:
        return LLMResponse(ticker=analysis.ticker, narrative="LLM disabled", confidence=0, recommendation="N/A")

    prompt = build_prompt(analysis)
    try:
        content = await call_ollama(prompt, model=config.model, base_url=config.api_url, api_key=config.api_key)
        return parse_llm_response(content, ticker=analysis.ticker, model=config.model)
    except httpx.ConnectError:
        return LLMResponse(ticker=analysis.ticker, narrative=f"Cannot connect to Ollama at {config.api_url}", confidence=0, recommendation="N/A")
    except Exception as e:
        return LLMResponse(ticker=analysis.ticker, narrative=f"Error: {e}", confidence=0, recommendation="N/A")


async def analyze_batch_llm(
    analyses: list[TickerAnalysis],
    config: LLMConfig,
) -> BatchLLMResult:
    if not config.enabled:
        return BatchLLMResult(
            responses=[LLMResponse(ticker=a.ticker, narrative="LLM disabled") for a in analyses],
        )

    results = BatchLLMResult(model=config.model)
    for analysis in analyses:
        resp = await analyze_ticker_llm(analysis, config)
        results.responses.append(resp)

    return results
