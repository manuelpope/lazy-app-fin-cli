import asyncio
from datetime import datetime

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Label, Static, Tree

import pandas as pd

from finlens.analytics.capm import CAPMResult
from finlens.config import FinLensConfig
from finlens.data.fetcher import DataCache, fetch_multi_ticker, fetch_ticker_data
from finlens.llm.analyst import LLMResponse, analyze_batch_llm, analyze_ticker_llm, ollama_list_models
from finlens.signals.engine import TickerAnalysis, analyze_ticker


class MarketNav(Tree[str]):
    def __init__(self, markets: dict[str, str], **kwargs):
        super().__init__("MARKETS", **kwargs)
        self.markets = markets

    def on_mount(self) -> None:
        root = self.root
        root.expand()
        for key, name in self.markets.items():
            child = root.add(name, data=key)
            child.add("Loading...")


class SignalMatrix(DataTable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._selected_ticker: str | None = None

    def on_mount(self) -> None:
        self.add_columns(
            "TICKER",
            "PRICE",
            "CHG%",
            "SMA",
            "RSI",
            "MACD",
            "BB",
            "CAPM",
            "SHARPE",
            "VOL",
            "SCORE",
            "SIGNAL",
        )
        self.cursor_type = "row"

    def update_rows(self, analyses: list[TickerAnalysis]) -> None:
        self.clear()
        for a in analyses:
            signal_icon = self._signal_icon(a.composite_signal)
            row = [
                a.ticker,
                f"${a.price}",
                f"{a.price_change_pct:+.2f}%",
                self._ind_icon(a.indicators.sma_signal),
                self._ind_icon(a.indicators.rsi_signal),
                self._ind_icon(a.indicators.macd_signal),
                self._ind_icon(a.indicators.bb_signal),
                self._capm_icon(a.capm),
                f"{a.risk.sharpe_ratio:.2f}",
                f"{a.garch.annualized_vol * 100:.1f}%",
                f"{a.composite_score:+.2f}",
                f"{signal_icon} {a.composite_signal}",
            ]
            self.add_row(*row)

    def _signal_icon(self, signal: str) -> str:
        return {"BUY": "[green]▲[/]", "SELL": "[red]▼[/]", "HOLD": "[dim]■[/]"}.get(signal, "[dim]?[/]")

    def _ind_icon(self, signal: str) -> str:
        return {"BUY": "[green]▲[/]", "SELL": "[red]▼[/]", "HOLD": "[dim]■[/]"}.get(signal, "[dim]?[/]")

    def _capm_icon(self, capm: CAPMResult | None) -> str:
        if capm is None:
            return "[dim]N/A[/]"
        if capm.signal == "BUY":
            return "[green]▲[/]"
        if capm.signal == "SELL":
            return "[red]▼[/]"
        return "[dim]■[/]"


class DetailPanel(Static):
    def __init__(self, **kwargs):
        super().__init__("Select a ticker to see details", **kwargs)

    def show_analysis(self, analysis: TickerAnalysis | None) -> None:
        if analysis is None:
            super().update("Select a ticker to see details")
            return

        a = analysis
        ind = a.indicators
        capm = a.capm
        garch = a.garch
        risk = a.risk

        signal_color = self._color_for_signal(a.composite_signal)

        var = a.var
        var_block = ""
        cvar_color = "red" if var and var.cvar_95_pct > 3.5 else "green" if var and var.cvar_95_pct < 1.5 else "yellow"
        if var:
            var_block = f"""
[bold cyan]── VaR / CVaR (t-Student, ν={garch.dof}) ──[/]
VaR 95%: {var.var_95_pct:.2f}%   CVaR 95%: [{cvar_color}]{var.cvar_95_pct:.2f}%[/]   Hist 95%: {var.var_95_hist_pct:.2f}%
Skewness: {var.skewness}   Kurtosis: {var.kurtosis}   [{cvar_color}]Risk: {var.signal}[/]"""

        content = f"""[bold]{a.ticker}[/]  ${a.price}  {a.price_change_pct:+.2f}%
[dim]Market:[/] {a.market.upper()}

[bold cyan]── Indicators ──[/]
SMA(20/50):     {ind.sma_signal}
RSI(14):        {ind.rsi_signal}
MACD:           {ind.macd_signal}
Bollinger Bands:{ind.bb_signal}

[bold cyan]── CAPM (Heavy-tailed) ──[/]
[dim]β:[/] {capm.beta if capm else 'N/A'}   [dim]α (ann):[/] {capm.alpha if capm else 'N/A'}   [dim]R²:[/] {capm.r_squared if capm else 'N/A'}
[dim]Skew:[/] {capm.skewness if capm else 'N/A'}   [dim]Kurt:[/] {capm.kurtosis if capm else 'N/A'}

[bold cyan]── GJR-GARCH(1,1) ──[/]
[dim]ω:[/] {garch.omega:.6f}   [dim]α:[/] {garch.alpha}   [dim]γ:[/] {garch.gamma}   [dim]β:[/] {garch.beta}
[dim]Persistence:[/] {garch.persistence}   [dim]σ (ann):[/] {garch.annualized_vol * 100:.2f}%{var_block}

[bold cyan]── Risk Metrics ──[/]
Sharpe: {risk.sharpe_ratio}   Sortino: {risk.sortino_ratio}   Calmar: {risk.calmar_ratio}
Max Drawdown: {risk.max_drawdown_pct:.2f}%   Vol: {risk.volatility_annual * 100:.2f}%

[bold {signal_color}]── COMPOSITE SCORE: {a.composite_score:+.2f} → {a.composite_signal} ──[/]"""

        super().update(content)

    def _rsi_val(self, signal: str) -> str:
        return ""

    def _color_for_signal(self, signal: str) -> str:
        return {
            "BUY": "green",
            "SELL": "red",
            "HOLD": "white",
        }.get(signal, "white")


class HelpModal(Static):
    def __init__(self, **kwargs):
        super().__init__(HELP_TEXT, **kwargs)


HELP_TEXT = """\
[bold]finlens[/] - Terminal Signal Scanner
Version 0.1.0

Navigation:
  ↑ / ↓          Navigate tickers in the matrix
  Tab            Cycle focus: Nav ↔ Matrix ↔ Detail
  Enter          Select / expand

Actions:
  s              Force scan (refresh data)
  w              Toggle watch mode (auto-refresh)
  l              LLM: analyze selected ticker
  L              LLM: analyze filtered market group
  A              LLM: analyze ALL tickers
  c              Open config
  h / ?          Toggle this help

Markets (Nav sidebar):
  1              All markets
  2              US only
  3              Argentina only
  4              Bonds only

Quit:
  q / Ctrl+C     Exit finlens

Indicators Legend:
  ▲ BUY          Bullish signal
  ▼ SELL         Bearish signal
  ■ HOLD         Neutral / no signal

Press h to close this help.
"""


class LLMOverlay(Static):
    def __init__(self, title: str, **kwargs):
        super().__init__("[bold yellow]⏳ LLM Analysis[/]\n\nPress Escape to close", **kwargs)
        self._title = title
        self._responses: list[LLMResponse] = []

    def set_loading(self, model: str) -> None:
        super().update(f"[bold yellow]⏳ {self._title}[/]\n[dim]Model: {model}[/]\n\nAnalyzing...")

    def add_result(self, resp: LLMResponse) -> None:
        self._responses.append(resp)
        self._refresh_display()

    def add_error(self, ticker: str, error: str) -> None:
        err = LLMResponse(ticker=ticker, recommendation="ERR", confidence=0.0, narrative=f"Error: {error}", model="")
        self._responses.append(err)
        self._refresh_display()

    def _refresh_display(self) -> None:
        lines = [f"[bold yellow]{self._title}[/]"]
        done = len(self._responses)
        lines.append(f"[dim]Model: {self._responses[0].model if self._responses else '?'} | Done: {done}[/]")
        lines.append("")
        for r in self._responses:
            color = {"BUY": "green", "SELL": "red", "HOLD": "white"}.get(r.recommendation, "white")
            lines.append(f"[bold]{r.ticker}[/]  [{color}]{r.recommendation}[/]  [{color}][confidence: {r.confidence:.2f}][/]")
            lines.append(f"  {r.narrative[:200]}")
            lines.append("")
        lines.append("[dim]Press Escape or Space to close[/]")
        super().update("\n".join(lines))

    def on_key(self, event) -> None:
        if event.key in ("escape", "space"):
            app = self.app
            if hasattr(app, "show_help"):
                app.show_help = False
            if hasattr(app, "show_llm"):
                app.show_llm = False
            event.stop()


class FinLensApp(App):
    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        height: 100%;
        layout: horizontal;
    }

    #sidebar {
        width: 28;
        background: $panel;
        border-right: solid $border;
    }

    #content-area {
        width: 1fr;
        layout: vertical;
    }

    #matrix-panel {
        height: 1fr;
        border-bottom: solid $border;
    }

    #detail-panel {
        height: 40%;
        padding: 1;
    }

    DataTable {
        height: 100%;
    }

    MarketNav {
        height: 100%;
    }

    DetailPanel {
        background: $panel;
        padding: 1 2;
        overflow-y: auto;
    }

    HelpModal {
        align: center middle;
        background: $surface;
        border: solid $accent;
        width: 70;
        height: 70;
        padding: 2;
    }

    LLMOverlay {
        align: center middle;
        background: $surface;
        border: solid $success;
        width: 80;
        height: 80%;
        padding: 1 2;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("h", "toggle_help", "Help", show=True),
        Binding("s", "scan", "Scan", show=True),
        Binding("w", "watch", "Watch", show=True),
        Binding("l", "llm_ticker", "LLM(T)", show=False),
        Binding("L", "llm_market", "LLM(M)", show=True),
        Binding("A", "llm_all", "LLM(A)", show=True),
        Binding("c", "config", "Config", show=True),
        Binding("1", "filter_all", "All", show=False),
        Binding("2", "filter_us", "US", show=False),
        Binding("3", "filter_ar", "Arg", show=False),
        Binding("4", "filter_bonds", "Bonds", show=False),
        Binding("escape", "dismiss_modal", "Close", show=False, priority=True),
        Binding("tab", "cycle_focus", "Cycle Focus", show=True),
    ]

    selected_ticker = reactive[str | None](None)
    show_help = reactive(False)
    show_llm = reactive(False)
    watch_mode = reactive(False)

    def __init__(self, config: FinLensConfig):
        super().__init__()
        self.config = config
        self.analyses: list[TickerAnalysis] = []
        self.filtered_analyses: list[TickerAnalysis] = []
        self.current_filter: str = "all"

    def compose(self) -> ComposeResult:
        yield Header()

        market_names = {k: v.name for k, v in self.config.markets.items()}

        with Horizontal(id="main-container"):
            with Vertical(id="sidebar"):
                yield Label("[bold]MARKETS[/bold]", id="sidebar-title")
                yield MarketNav(market_names, id="market-nav")
                yield Static(
                    "\n\n[dim]Keys:[/dim]\n"
                    "[dim]1[/dim] All\n"
                    "[dim]2[/dim] US\n"
                    "[dim]3[/dim] AR\n"
                    "[dim]4[/dim] Bonds\n"
                    "[dim]m/g/a/L[/dim] Tabs\n"
                    "[dim]h[/dim] Help",
                    id="sidebar-keys",
                )

            with Vertical(id="content-area"):
                yield SignalMatrix(id="signal-matrix")
                yield DetailPanel(id="detail-panel")

        yield Footer()

    def watch_show_llm(self, show_llm: bool) -> None:
        if show_llm:
            self.mount(LLMOverlay(id="llm-overlay", title="LLM Analysis"))
        else:
            try:
                self.query_one("#llm-overlay").remove()
            except Exception:
                pass

    def watch_show_help(self, show_help: bool) -> None:
        if show_help:
            self.mount(HelpModal(id="help-modal"))
            self.query_one("#help-modal", HelpModal).focus()
        else:
            try:
                self.query_one("#help-modal").remove()
            except Exception:
                pass

    def on_mount(self) -> None:
        matrix = self.query_one("#signal-matrix", SignalMatrix)
        matrix.focus()

        self.sub_title = "Loading..."
        self.run_worker(self.load_data)

    def _get_selected_analysis(self) -> TickerAnalysis | None:
        if not self.selected_ticker:
            return None
        return next((a for a in self.filtered_analyses if a.ticker == self.selected_ticker), None)

    async def _run_llm_analysis(self, target: str) -> None:
        config = self.config.llm
        analyses: list[TickerAnalysis] = []
        title = ""

        if target == "ticker":
            a = self._get_selected_analysis()
            if a is None and self.filtered_analyses:
                a = self.filtered_analyses[0]
            if a is None:
                self.notify("No tickers available for LLM", timeout=3)
                return
            analyses = [a]
            title = f"LLM: {a.ticker}"
        elif target == "market":
            analyses = self.filtered_analyses.copy()
            if not analyses:
                self.notify("Empty market filter", timeout=3)
                return
            title = f"LLM: {self.current_filter.upper()} ({len(analyses)} tickers)"
        elif target == "all":
            analyses = self.analyses.copy()
            title = f"LLM: ALL ({len(analyses)} tickers)"

        self.show_llm = True
        await asyncio.sleep(0.1)
        try:
            overlay = self.query_one("#llm-overlay", LLMOverlay)
            overlay.set_loading(config.model)
        except Exception:
            return
        await asyncio.sleep(0.1)

        for analysis in analyses:
            try:
                resp = await analyze_ticker_llm(analysis, config)
                overlay.add_result(resp)
            except Exception as e:
                overlay.add_error(analysis.ticker, str(e))

    def action_llm_ticker(self) -> None:
        self.run_worker(self._run_llm_analysis("ticker"))

    def action_llm_market(self) -> None:
        self.run_worker(self._run_llm_analysis("market"))

    def action_llm_all(self) -> None:
        self.run_worker(self._run_llm_analysis("all"))

    async def load_data(self) -> None:
        self.analyses = []
        self.filtered_analyses = []
        cache = DataCache(self.config.cache.dir)

        for market_key, market_config in self.config.markets.items():
            tickers = market_config.tickers
            market_rf = market_config.risk_free_rate

            benchmark_returns: pd.Series | None = None
            try:
                bm_df = fetch_ticker_data(
                    market_config.benchmark,
                    period=f"{self.config.signals.lookback_days + 120}d",
                    cache=cache,
                )
                bm_close = bm_df["Close"].squeeze()
                benchmark_returns = bm_close.pct_change().dropna()
            except Exception:
                pass

            for ticker, df in fetch_multi_ticker(tickers, self.config):
                analysis = analyze_ticker(
                    ticker,
                    market_key,
                    {"risk_free_rate": market_rf},
                    df,
                    self.config,
                    benchmark_returns=benchmark_returns,
                )
                if analysis:
                    self.analyses.append(analysis)

        self.filtered_analyses = self.analyses.copy()
        self.update_tables()
        self.sub_title = f"Last scan: {datetime.now().strftime('%H:%M')} | {len(self.analyses)} tickers"

    def update_tables(self) -> None:
        matrix = self.query_one("#signal-matrix", SignalMatrix)
        matrix.update_rows(self.filtered_analyses)

        detail = self.query_one("#detail-panel", DetailPanel)
        if self.selected_ticker:
            selected = next((a for a in self.filtered_analyses if a.ticker == self.selected_ticker), None)
            detail.show_analysis(selected)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        if isinstance(event.node.data, str) and event.node.data != "root":
            self.current_filter = event.node.data
            if self.current_filter == "all":
                self.filtered_analyses = self.analyses.copy()
            else:
                self.filtered_analyses = [a for a in self.analyses if a.market == self.current_filter]
            self.update_tables()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        row_index = event.cursor_row
        if row_index is not None and row_index < len(self.filtered_analyses):
            analysis = self.filtered_analyses[row_index]
            self.selected_ticker = analysis.ticker
            detail = self.query_one("#detail-panel", DetailPanel)
            detail.show_analysis(analysis)

    def action_dismiss_modal(self) -> None:
        if self.show_help:
            self.show_help = False
        elif self.show_llm:
            self.show_llm = False
        if not self.show_help and not self.show_llm:
            try:
                self.query_one("#signal-matrix", SignalMatrix).focus()
            except Exception:
                pass

    def action_toggle_help(self) -> None:
        self.show_help = not self.show_help
        if not self.show_help:
            self.query_one("#signal-matrix", SignalMatrix).focus()

    def action_scan(self) -> None:
        self.sub_title = "Scanning..."
        self.run_worker(self.load_data)

    def action_watch(self) -> None:
        self.watch_mode = not self.watch_mode
        if self.watch_mode:
            self.sub_title = f"Watch mode ON | {datetime.now().strftime('%H:%M')}"
        else:
            self.sub_title = f"Watch mode OFF | {datetime.now().strftime('%H:%M')}"

    def action_config(self) -> None:
        self.notify("Config editing coming soon", timeout=3)

    def action_cycle_focus(self) -> None:
        pass

    def action_filter_all(self) -> None:
        self.filtered_analyses = self.analyses.copy()
        self.update_tables()

    def action_filter_us(self) -> None:
        self.filtered_analyses = [a for a in self.analyses if a.market == "us"]
        self.update_tables()

    def action_filter_ar(self) -> None:
        self.filtered_analyses = [a for a in self.analyses if a.market == "argentina"]
        self.update_tables()

    def action_filter_bonds(self) -> None:
        self.filtered_analyses = [a for a in self.analyses if a.market == "bonds"]
        self.update_tables()
