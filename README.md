# FinLens

Terminal stock signal scanner with lazy daily analysis. Indicators, CAPM (t-dist), GJR-GARCH, risk metrics, and LLM analysis via Ollama.

## Quick start

```bash
uv sync
uv run finlens
```

## Install

```bash
pip install -e .
# or
uv pip install -e .
```

## Usage

```bash
finlens           # TUI interactiva
finlens scan      # modo headless
finlens --help    # ayuda CLI
```

### TUI keys

| Key | Action |
|-----|--------|
| `1`-`4` | Filter markets |
| `l L A` | LLM analysis (ticker / market / all) |
| `s` | Rescan |
| `h` | Help |
| `q` | Quit |

## Config

Edit `config.yaml` for tickers, signals, LLM, cache. Environment variables in `.env` override config values:

```
OLLAMA_MODEL=qwen3.5:9b-mlx
OLLAMA_API_URL=http://localhost:11434
OLLAMA_API_KEY=
```

## Tests

```bash
uv run python tests/test_llm_e2e.py
```

## Docs

Abrir `docs/manual.html` en el navegador.
