from pathlib import Path

import typer

from finlens.config import FinLensConfig
from finlens.output.terminal import FinLensApp

def _resolve_config(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    pkg_dir = Path(__file__).resolve().parent.parent
    alt = pkg_dir / path
    if alt.exists():
        return alt
    return candidate


app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file"),
    force_refresh: bool = typer.Option(False, "--force", "-f", help="Force data refresh"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    if ctx.invoked_subcommand is None:
        try:
            cfg_path = _resolve_config(config)
            if not cfg_path.exists():
                typer.echo(f"[red]Config file not found at: {cfg_path}[/red]")
                raise typer.Exit(code=1)

            fin_config = FinLensConfig.load(cfg_path)
            application = FinLensApp(fin_config)
            application.run()

        except Exception as e:
            typer.echo(f"[red]Error: {e}[/red]")
            if verbose:
                import traceback
                traceback.print_exc()
            raise typer.Exit(code=1)


@app.command()
def scan(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to config file"),
    force_refresh: bool = typer.Option(False, "--force", "-f"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    import pandas as pd
    from finlens.data.fetcher import DataCache, fetch_multi_ticker, fetch_ticker_data
    from finlens.signals.engine import analyze_ticker

    try:
        cfg_path = _resolve_config(config)
        if not cfg_path.exists():
            typer.echo(f"[red]Config file not found: {cfg_path}[/red]")
            raise typer.Exit(code=1)

        fin_config = FinLensConfig.load(cfg_path)
        cache = DataCache(fin_config.cache.dir)
        typer.echo("[cyan]Scanning markets...[/cyan]")

        for market_key, market_config in fin_config.markets.items():
            benchmark_returns: pd.Series | None = None
            try:
                bm_df = fetch_ticker_data(
                    market_config.benchmark,
                    period=f"{fin_config.signals.lookback_days + 120}d",
                    cache=cache,
                    force_refresh=force_refresh,
                )
                bm_close = bm_df["Close"].squeeze()
                benchmark_returns = bm_close.pct_change().dropna()
            except Exception:
                pass

            for ticker, df in fetch_multi_ticker(market_config.tickers, fin_config, force_refresh):
                analysis = analyze_ticker(
                    ticker,
                    market_key,
                    {"risk_free_rate": market_config.risk_free_rate},
                    df,
                    fin_config,
                    benchmark_returns=benchmark_returns,
                )
                if analysis:
                    typer.echo(f"  {ticker}: {analysis.composite_signal} ({analysis.composite_score:+.2f})")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"[red]{e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
