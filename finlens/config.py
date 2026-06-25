import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]


class MarketConfig(BaseModel):
    name: str
    tickers: list[str]
    benchmark: str
    risk_free_rate: float


class SignalsConfig(BaseModel):
    lookback_days: int = 252
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    sma_fast: int = 20
    sma_slow: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    garch_p: int = 1
    garch_q: int = 1


class CompositeConfig(BaseModel):
    weights: dict[str, float] = Field(default_factory=lambda: {
        "sma": 0.15,
        "rsi": 0.20,
        "macd": 0.15,
        "bb": 0.15,
        "capm_alpha": 0.15,
        "risk_ratio": 0.20,
    })
    buy_threshold: float = 0.30
    sell_threshold: float = -0.30


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen3.5:9b-mlx"
    api_url: str = "http://localhost:11434"
    api_key_env: str = ""
    api_key: str = ""
    enabled: bool = True


class CacheConfig(BaseModel):
    dir: str = ".finlens_cache"
    max_age_hours: int = 24


class FinLensConfig(BaseModel):
    markets: dict[str, MarketConfig]
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    composite: CompositeConfig = Field(default_factory=CompositeConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "FinLensConfig":
        if load_dotenv is not None:
            load_dotenv()
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            data: dict[str, Any] = yaml.safe_load(f)

        if "llm" not in data:
            data["llm"] = {}
        llm = data["llm"]
        if os.environ.get("OLLAMA_MODEL"):
            llm["model"] = os.environ["OLLAMA_MODEL"]
        if os.environ.get("OLLAMA_API_URL"):
            llm["api_url"] = os.environ["OLLAMA_API_URL"]
        if os.environ.get("OLLAMA_API_KEY"):
            llm["api_key"] = os.environ["OLLAMA_API_KEY"]

        return cls(**data)

    def model_post_init(self, __context: Any) -> None:
        if self.llm.api_key_env and not self.llm.api_key:
            self.llm.api_key = os.environ.get(self.llm.api_key_env, "")
