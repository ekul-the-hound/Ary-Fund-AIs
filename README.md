# ARY QUANT — Hedge Fund AI Research Assistant

Quantamental research platform for a student-run hedge fund (long-term
focus). Runs a local LLM against SEC filings, market data, macroeconomic
indicators, and a normalized data registry to screen stocks, analyze
filings, scan portfolio risks, generate multi-page investment theses, and
run probabilistic quant models — all from a Streamlit dashboard.

Everything runs locally. No paid APIs, no cloud LLM costs.

---

## Overview

Six independent layers, each importable on its own:

- **Data layer** — SEC EDGAR fetcher with XBRL parsing and Form 4/13F
  support, yfinance market data with technicals, FRED macro dashboard,
  news and WSB sentiment aggregates, sanctions and geopolitical supply-
  chain feeds, SQLite portfolio DB, a normalized `DataRegistry` (~100
  canonical fields, per-field source priority, conflict resolution), pure-
  computation derived signals (factor exposures, regime labels, sector
  heatmaps, composite risk scores), and a refresh scheduler with hourly /
  daily / weekly / event-driven orchestration.
- **Agent layer** — `ask_agent()` is the single entry point every module
  calls. `AgentRequest` goes in, `AgentResponse` comes back. Mock, Ollama
  success, and Ollama failure all return the same shape so the pipeline
  never crashes from a backend outage. Includes: filing analyzer,
  data-point analyzer, risk scanner, short-form thesis generator,
  multi-page thesis essay, and a self-review pass. An explicit data-gap
  protocol prevents the model from inventing analysis for missing inputs.
- **Quant layer** — VaR & Expected Shortfall (historical / parametric /
  Monte Carlo), volatility estimators and forecasting, rule-based and HMM
  regime classifiers, Kelly criterion, volatility-targeted position sizing,
  Black-Scholes pricing and Greeks, Avellaneda-Stoikov market-making
  reservation prices, Poisson jump-arrival intensity, GBM Monte Carlo,
  Hurst exponent, Ornstein-Uhlenbeck mean reversion, and additional
  research modules (HRP, RMT, SABR, Longstaff-Schwartz, ergodicity,
  wavelet regimes, Lyapunov exponents, Lempel-Ziv complexity, sandpile,
  FFT analysis, GAN synthetic data generation).
- **Screener** — TradingView-style stock screener over a curated ~600-
  ticker US universe (S&P 500 + large/mid caps). Live prices and 1-day
  change via batched `yf.download` (5-minute cache). Fundamentals lazy-
  loaded from `MarketData.get_fundamentals` with 24-hour SQLite cache. 15
  filter pills wired to real filtering logic. No synthetic data — missing
  cells show "—" rather than fabricated values.
- **UI layer** — Streamlit dashboard with: sidebar ticker picker (backed
  by the full ~600-symbol universe), portfolio overview cards, interactive
  price chart with MA / RSI / vol overlays, risk & thesis panels, macro
  context, streaming chat agent (grounded per-ticker with full context),
  and a quant playground tab (HMM regime, Hurst exponent, OU mean
  reversion, GBM Monte Carlo simulation).
- **Test layer** — per-module unit tests, registry integration test
  (44 passing), and an end-to-end smoke test, all defaulting to mock mode.

`config.py` centralizes model selection, API keys, paths, logging, and
risk thresholds. `main.py` is the top-level CLI entry point.

---

## Model Configuration

Three model tags in `config.AGENT_MODELS`:

- **`mock`** — deterministic stub, no model server required. Used in tests
  and when Ollama is unreachable.
- **`dev`** — `phi3:3.8b` via Ollama. Current default; validates the full
  pipeline end-to-end on modest hardware.
- **`prod`** — `qwen3:14b-instruct` (or 30B variant); swap in once
  thresholds are confirmed.

Switch models by editing `DEFAULT_AGENT_MODEL` in `config.py`. No other
code changes needed — callers route through `AGENT_MODELS`, never a raw
model string. Per-call overrides are also supported via
`AgentRequest(..., model_tag="mock")`.

---

## Project Structure

```
.
├── config.py                       # model tags, API keys, paths, thresholds
├── main.py                         # CLI entry point; runs the full pipeline
│
├── app.py                          # Streamlit dashboard (main entry point)
├── screener.py                     # TradingView-style screener tab
├── playground.py                   # Quant playground tab
├── chat.py                         # Streaming chat agent tab
├── universe.py                     # ~600-ticker US universe + validation
├── providers.py                    # Unified market data provider interface
├── earnings_calendar.py            # Earnings date tracking
│
├── base_agent.py                   # AgentRequest / AgentResponse / ask_agent
├── filing_analyzer.py              # 10-K / 10-Q metric extraction
├── data_point_analyzer.py          # Per-field signal interpretation
├── data_point_analyzer_section.py  # Section-level data point rendering
├── risk_scanner.py                 # Portfolio-level risk flagging
├── thesis_generator.py             # Short-form thesis (BUY / HOLD / SELL)
├── thesis_essay.py                 # Multi-page investment memo
├── thesis_review.py                # Self-review pass on generated theses
├── essay.py                        # Essay rendering utilities
│
├── sec_fetcher.py                  # SEC EDGAR REST + XBRL parser
├── market_data.py                  # yfinance + technicals (RSI, MACD, BBands)
├── macro_data.py                   # FRED: rates, inflation, VIX, recession prob
├── sentiment_news.py               # News + WSB aggregates with VADER fallback
├── geo_supply.py                   # Sanctions, freight, energy, supply chain
├── portfolio_db.py                 # SQLite: positions, trades, snapshots
├── pipeline.py                     # Daily refresh + agent context builder
├── data_registry.py                # Normalized schema, upserts, snapshots
├── derived_signals.py              # Factors, regime, heatmaps, risk scores
├── refresh_scheduler.py            # Hourly / daily / weekly orchestration
├── portfolio_construction.py       # Portfolio optimization utilities
│
├── var_es.py                       # Historical / parametric / MC VaR & ES
├── volatility.py                   # Vol estimators + forecasting
├── regime.py                       # Rule-based regime classifier
├── regime_hmm.py                   # HMM-based regime classifier
├── kelly.py                        # Kelly criterion utilities
├── position_sizing.py              # Kelly + vol-target combined sizing
├── black_scholes.py                # European option pricing + Greeks
├── avellaneda_stoikov.py           # Market-making reservation prices
├── poisson.py                      # Jump-arrival intensity
├── gbm.py                          # Geometric Brownian motion
├── hurst.py                        # Hurst exponent / long-memory tests
├── ou_process.py                   # Ornstein-Uhlenbeck mean reversion
├── monte_carlo.py                  # General Monte Carlo utilities
├── sequential_monte_carlo.py       # Particle filter / SMC
├── longstaff_schwartz.py           # LSM American option pricing
├── sabr.py                         # SABR stochastic vol model
├── hrp.py                          # Hierarchical risk parity
├── rmt.py                          # Random matrix theory
├── mst.py                          # Minimum spanning tree (correlation)
├── wavelet_regimes.py              # Wavelet-based regime detection
├── regime_hmm.py                   # Hidden Markov Model regimes
├── lyapunov.py                     # Lyapunov exponent (chaos measure)
├── lempel_ziv.py                   # Lempel-Ziv complexity
├── fft_analysis.py                 # FFT spectral analysis
├── ergodicity.py                   # Ergodicity economics utilities
├── sandpile.py                     # Bak-Tang-Wiesenfeld sandpile model
├── omori.py                        # Omori law (aftershock clustering)
├── girsanov.py                     # Girsanov measure change
├── gan_synthetic.py                # GAN-based synthetic data generation
├── wave_function_collapse.py       # WFC for scenario generation
│
├── test_all.py                     # Full consolidated test suite
│
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── setup.py
├── activate.bat                    # Windows convenience: activate the venv
├── .env.example                    # Template for local secrets
└── README.md
```

Runtime artifacts (`data/cache/`, `data/portfolio.db`, `logs/`) are
created automatically on first run and excluded from version control.

---

## Installation

### 1. Prerequisites

- Python 3.11+
- Git
- [Ollama](https://ollama.com) — only needed for `dev` or `prod` mode

### 2. Clone and install

```bash
git clone https://github.com/ekul-the-hound/Ary-Fund-AIs.git
cd Ary-Fund-AIs

python -m venv .venv
.venv\Scripts\activate               # Windows
# source .venv/bin/activate          # Linux / macOS

pip install -r requirements.txt
```

Or, for an editable install with dev tools:

```bash
pip install -e ".[dev]"
```

For the heavier quant extras (`statsmodels`, `arch` for GARCH,
`hmmlearn` for HMM regimes):

```bash
pip install -e ".[quant]"
```

### 3. Pull a model (skip for mock mode)

```bash
ollama serve                         # leave running in its own terminal
ollama pull phi3:3.8b                # current dev model
```

### 4. API keys

Copy `.env.example` → `.env` and fill in:

```dotenv
FRED_API_KEY=your_fred_key_here
SEC_USER_AGENT=Your Name your@email.com
```

- **FRED** is free: [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- **SEC EDGAR** requires a descriptive User-Agent header with contact
  info — no key needed.

`config.py` loads `.env` automatically via `python-dotenv` on import.

### 5. Verify the stack

```bash
# Offline mock check — no Ollama required
python -c "from base_agent import ask_agent, AgentRequest; print(ask_agent(AgentRequest(prompt='ping', context={}, model_tag='mock')).generated_json)"

# Market data check
python -c "import yfinance as yf; print(yf.Ticker('AAPL').history(period='1d')['Close'].iloc[-1])"
```

---

## Running the Project

Full pipeline (data refresh → agent analysis → risk + thesis) over
`config.WATCHLIST`:

```bash
python main.py
```

Streamlit dashboard:

```bash
streamlit run app.py
```

Scheduled data refresh:

```bash
python refresh_scheduler.py hourly
python refresh_scheduler.py daily
python refresh_scheduler.py weekly
```

### Switching agent backends

In `config.py`:

```python
DEFAULT_AGENT_MODEL: str = "dev"    # "mock" | "dev" | "prod"
```

When Ollama is unreachable, `ask_agent` falls back to mock automatically.

---

## Running Tests

```bash
pytest                              # full suite via pytest.ini
python test_all.py                  # consolidated test runner
pytest -x                           # stop at first failure
```

Tests default to `mock` mode — they run offline and deterministically.

---

## Current Status

**Built:**
- Data layer: SEC EDGAR (XBRL, Form 4, 13F), market data with
  technicals, FRED macro, news + sentiment, sanctions and supply-chain
  feeds, portfolio DB, normalized `DataRegistry` with ~100 canonical
  fields and derived signals, refresh scheduler.
- Agent layer: model-agnostic routing, filing analyzer, data-point
  analyzer, risk scanner, short-form thesis, multi-page essay, self-
  review pass, data-gap protocol.
- Quant layer: VaR/ES, volatility, regime (rule-based + HMM), Kelly,
  position sizing, Black-Scholes, Avellaneda-Stoikov, Poisson, GBM,
  Hurst, OU process, plus a full suite of research models (HRP, RMT,
  MST, SABR, Longstaff-Schwartz, wavelets, ergodicity, GAN synthesis,
  and more).
- Screener: ~600-ticker universe, live prices, lazy fundamentals,
  15 filter pills, 24-hour SQLite cache, no synthetic data.
- UI: Streamlit dashboard with screener, portfolio overview, price
  charts, risk & thesis panels, macro context, streaming chat agent,
  and quant playground.
- Tests: 44+ passing across unit, integration, and smoke tests.

**In progress:**
- Wiring `pipeline.build_agent_context()` to read directly from the
  registry via `reg.snapshot()`.
- Populating `factor_returns` from Ken French's data library.
- Promoting `prod` model tag once validation is complete.

---

## Future Work

- **RAG module** — ChromaDB vector store + `nomic-embed-text` embeddings
  via Ollama, hooked into `pipeline.build_agent_context()`. More
  practical than fine-tuning for retrieval over filings and fund notes.
- **Daily scheduler hooks** — auto-scan holdings at market open, push
  risk flags to Slack.
- **PDF reports** — one-click investment memos combining thesis essay,
  charts, and risk summary.
- **Fine-tuning** — LoRA on smaller open models (Qwen3-1.7B / 4B via
  Unsloth) trained on SEC filings + fund notes. Deferred for MVP.

---

## Success Metrics (MVP targets)

- 85%+ accuracy on filing metric extraction
- 80%+ recall on portfolio risk flags vs analyst review
- < 2 min per full single-ticker analysis
- Full portfolio scan (10 holdings) completes end-to-end without crashes