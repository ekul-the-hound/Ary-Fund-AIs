# Ary Fund — Hedge Fund AI Research Assistant

Quantamental research assistant for a student-run hedge fund (long-term
focus). Runs a local LLM against SEC filings, market data, macroeconomic
indicators, and a normalized data registry to parse filings, scan portfolio
risks, generate multi-page investment theses, and produce quant risk
reports.

Everything runs locally. No paid APIs, no cloud LLM costs. The agent layer
is model-agnostic — a `mock` backend (offline, deterministic), a `dev`
backend (a small instruction-tuned model via Ollama), and a `prod` slot for
a larger model once validated.

---

## Overview

Five independent layers, each importable on its own:

- **Data layer** — `sec_fetcher`, `market_data`, `macro_data`,
  `sentiment_news`, `geo_supply`, `portfolio_db`, `pipeline`,
  `data_registry`, `derived_signals`, `refresh_scheduler`. Raw fetchers
  write into a normalized `DataRegistry` (canonical schema, per-field
  source priority, idempotent upserts, conflict resolution). Derived
  signals compute factor exposures, regime labels, sector heatmaps,
  composite risk scores, and macro stress on top of the registry. The
  scheduler orchestrates hourly / daily / weekly / event-driven refreshes
  with lazy module loading so a single missing dep doesn't block the rest.
- **Agent layer** — `base_agent`, `filing_analyzer`,
  `data_point_analyzer`, `risk_scanner`, `thesis_generator`,
  `thesis_essay`, `thesis_review`. `ask_agent()` is the single entry point
  every downstream module calls. `AgentRequest` goes in, `AgentResponse`
  comes back. Mock mode, Ollama success, and Ollama failure all produce
  the same shape, so the pipeline never crashes from a backend outage.
  The thesis stack now produces multi-section, multi-page memos with an
  explicit data-gap protocol — when inputs are missing, the model
  acknowledges the gap rather than inventing analysis around it.
- **Quant layer** — `var_es`, `volatility`, `regime`, `regime_hmm`,
  `kelly`, `position_sizing`, `black_scholes`, `avellaneda_stoikov`,
  `poisson`, `gbm`, `hurst`, `ou_process`. Historical / parametric / MC
  VaR & ES, close-to-close and Yang-Zhang vol estimators with forecasting,
  rule-based and HMM regime classifiers, Kelly + volatility-targeted
  sizing, options pricing and Greeks, market-making models, jump-arrival
  intensity, and stochastic process toolkits.
- **UI layer** — `app.py`. Streamlit dashboard: ticker picker, portfolio
  cards, price chart with MA / RSI / vol overlays, risk & thesis panels,
  macro context, and a debug view of the raw agent context JSON.
- **Test layer** — per-module unit tests plus an end-to-end smoke test,
  all defaulting to mock mode so they run offline and deterministically.

`config.py` centralizes model selection, API keys, paths, logging, and
risk thresholds. `main.py` is the top-level CLI entry point.

---

## Model Configuration

Three model tags in `config.AGENT_MODELS`:

- **`mock`** — deterministic stub, no model server required. Used in tests
  and when Ollama is unreachable.
- **`dev`** — a small instruction-tuned model served via Ollama. Current
  default; lets the full pipeline run end-to-end while the `prod` stack is
  validated.
- **`prod`** — reserved for a larger Qwen-family model once thresholds are
  confirmed.

Switch models by editing `DEFAULT_AGENT_MODEL` in `config.py`. No other
code changes — callers route through `AGENT_MODELS`, never a raw model
string. Per-call overrides are also supported via
`AgentRequest(..., model_tag="mock")`.

---

## Project Structure

```
.
├── config.py                       # model tags, API keys, paths, thresholds
├── main.py                         # CLI entry point; runs the full pipeline
│
├── agent/
│   ├── base_agent.py               # AgentRequest / AgentResponse / ask_agent
│   ├── filing_analyzer.py          # 10-K / 10-Q metric extraction
│   ├── data_point_analyzer.py      # Per-field signal interpretation
│   ├── risk_scanner.py             # Portfolio-level risk flagging
│   ├── thesis_generator.py         # Short-form thesis (BUY/HOLD/SELL)
│   ├── thesis_essay.py             # Multi-page investment memo
│   └── thesis_review.py            # Self-review pass on generated theses
│
├── data/
│   ├── sec_fetcher.py              # SEC EDGAR REST + XBRL parser
│   ├── market_data.py              # yfinance + technicals (RSI, MACD, BBands)
│   ├── macro_data.py               # FRED: rates, inflation, VIX, recession prob
│   ├── sentiment_news.py           # News + WSB aggregates with VADER fallback
│   ├── geo_supply.py               # Sanctions, freight, energy, supply chain
│   ├── portfolio_db.py             # SQLite: positions, trades, snapshots
│   ├── pipeline.py                 # Daily refresh + agent context builder
│   ├── data_registry.py            # Normalized schema, upserts, snapshots
│   ├── derived_signals.py          # Factors, regime, heatmaps, risk scores
│   └── refresh_scheduler.py        # Hourly / daily / weekly orchestration
│
├── quant/
│   ├── var_es.py                   # Historical / parametric / MC VaR & ES
│   ├── volatility.py               # Vol estimators + forecasting
│   ├── regime.py                   # Trend + drawdown + vol classifier
│   ├── regime_hmm.py               # HMM-based regime classifier
│   ├── kelly.py                    # Kelly criterion utilities
│   ├── position_sizing.py          # Kelly + vol-target combined sizing
│   ├── black_scholes.py            # European option pricing + Greeks
│   ├── avellaneda_stoikov.py       # Market-making bid/ask reservation prices
│   ├── poisson.py                  # Jump-arrival intensity
│   ├── gbm.py                      # Geometric Brownian motion
│   ├── hurst.py                    # Hurst exponent / long-memory tests
│   └── ou_process.py               # Ornstein-Uhlenbeck mean reversion
│
├── ui/
│   └── app.py                      # Streamlit dashboard
│
├── tests/
│   ├── conftest.py                 # Shared pytest fixtures
│   ├── test_base_agent.py
│   ├── test_filing_analyzer.py
│   ├── test_risk_scanner.py
│   ├── test_thesis_generator.py
│   ├── test_main.py
│   ├── test_market_data_ext.py
│   ├── test_sec_fetcher_ext.py
│   ├── test_data_registry.py
│   ├── test_integration.py
│   └── test_end_to_end_smoke.py
│
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── setup.py
├── activate.bat                    # Windows convenience: activate the venv
├── .env.example                    # Template for local secrets
└── README.md
```

Caches and the portfolio database are created at runtime under `data/`
(e.g. `data/cache/`, `data/sec_cache/`, `data/portfolio.db`) and are
excluded from version control by `.gitignore`.

---

## Installation

### 1. Prerequisites

- Python 3.11+
- Git
- [Ollama](https://ollama.com) — only needed if you run anything other
  than `mock` mode

### 2. Clone and install

```bash
git clone https://github.com/ekul-the-hound/Ary-Fund-AIs.git
cd Ary-Fund-AIs

python -m venv .venv
.venv\Scripts\activate               # Windows
# source .venv/bin/activate          # Linux/macOS

pip install -r requirements.txt
```

Or, for an editable install with dev tools:

```bash
pip install -e ".[dev]"
```

For the heavier quant extras (statistical models, GARCH, etc.):

```bash
pip install -e ".[quant]"
```

### 3. Pull a model (skip for mock mode)

```bash
ollama serve                         # leave running in its own terminal
ollama pull <model_name>             # see config.AGENT_MODELS for current dev tag
```

### 4. API keys

Copy `.env.example` → `.env` and fill in:

```dotenv
FRED_API_KEY=your_fred_key_here
SEC_AGENT_NAME=Your Name
SEC_AGENT_EMAIL=you@example.com
```

- **FRED** is free: [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- **SEC EDGAR** requires a descriptive User-Agent with contact info — no
  key, just the name/email.

`config.py` loads `.env` automatically via `python-dotenv` on import.

### 5. Verify the stack

```bash
# Mock mode — no Ollama required
python -c "from agent.base_agent import ask_agent, AgentRequest; print(ask_agent(AgentRequest(prompt='ping', context={}, model_tag='mock')).generated_json)"

# Market data sanity check
python -c "import yfinance as yf; print(yf.Ticker('AAPL').history(period='1d')['Close'].iloc[-1])"
```

---

## Running the Project

Full pipeline (data refresh → agent analysis → risk + thesis) over the
default watchlist defined in `config.WATCHLIST`:

```bash
python main.py
```

Daily data refresh only, no LLM calls:

```bash
python -m data.pipeline
```

Scheduler-driven refreshes (granular by data half-life):

```bash
python -m data.refresh_scheduler hourly
python -m data.refresh_scheduler daily
python -m data.refresh_scheduler weekly
```

Streamlit dashboard:

```bash
streamlit run ui/app.py
```

### Switching agent backends

In `config.py`:

```python
DEFAULT_AGENT_MODEL: str = "dev"    # "mock" | "dev" | "prod"
```

When Ollama is unreachable, `ask_agent` falls back to mock output
automatically — the response contract is preserved either way.

---

## Running Tests

```bash
pytest                              # full suite
pytest -k filing_analyzer           # single module
pytest tests/test_end_to_end_smoke.py
pytest tests/test_integration.py    # full registry round-trip
pytest -x                           # stop at first failure
```

Pytest config lives in `pytest.ini` / `pyproject.toml`. Tests default to
`mock` mode, so they run offline and deterministically. The integration
test verifies that every fetcher writes into the registry correctly and
that downstream snapshots return the expected normalized fields.

---

## Current Status

**Built:**
- **Data layer:** SEC fetcher with CIK lookup and XBRL parsing
  (14+ canonical fields per company, plus derived FCF, R&D intensity,
  SBC/revenue, dilution YoY); Form 4 parsing with 10b5-1 and P/S code
  detection; 13F informationtable parsing; market data with technicals
  and multi-stock comparison; FRED macro dashboard with VIX term-structure
  ratio; news + WSB sentiment aggregates with VADER fallback; sanctions /
  freight / energy / supply-chain feeds; SQLite portfolio DB with FIFO
  P&L; the `DataPipeline` orchestrator; a normalized `DataRegistry` with
  ~100 canonical fields, per-field source priority, and conflict
  resolution; derived signals for factor exposures, regime labels, sector
  heatmaps, and composite risk scores; a refresh scheduler with lazy
  module loading.
- **Agent layer:** model-agnostic `ask_agent` core with mock/dev/prod
  routing, filing analyzer, data-point analyzer, risk scanner, short-form
  thesis generator, multi-page thesis essay, and a self-review pass.
  Explicit data-gap protocol prevents the model from inventing analysis
  for missing inputs.
- **Quant layer:** VaR & Expected Shortfall (historical + parametric +
  MC); volatility estimators and forecasts; rule-based and HMM regime
  classifiers; Kelly and volatility-targeted position sizing;
  Black-Scholes pricing and Greeks; Avellaneda-Stoikov market-making
  reservation prices; Poisson jump-arrival intensity; geometric Brownian
  motion, Hurst exponent, and Ornstein-Uhlenbeck process utilities.
- **UI:** Streamlit dashboard with portfolio overview, price charts with
  overlays, risk & thesis panels, macro context, and raw-context debug
  view.
- **Tests:** per-module unit tests, registry integration test, end-to-end
  smoke test — all defaulting to mock mode.
- **Packaging:** `pyproject.toml`, `requirements.txt`, `setup.py`,
  `pytest.ini`, `.env` support via `python-dotenv`, `.gitignore` covering
  secrets, caches, DBs, logs, and venvs.

**In progress:**
- Wiring `pipeline.build_agent_context()` to read directly from the
  registry via `reg.snapshot(ticker, [...])` rather than touching raw
  tables.
- Replacing the stubbed `recompute_global_risk_pulse()` with a proper
  market-wide composite.
- Populating `factor_returns` from Ken French's data library so factor
  exposures compute against live data.
- Building an ETF-issuer-CSV ingestion script that calls the existing
  `set_etf_holdings()` API.
- Promoting the `prod` model tag once validation is complete.

---

## Future Work

- **RAG over project data** — a `rag/` module using ChromaDB for local
  vector storage and `nomic-embed-text` via Ollama for embeddings,
  hooking into `pipeline.build_agent_context()`. Practical alternative to
  fine-tuning, especially for retrieval over filings and fund notes.
- **Optional fine-tuning** — LoRA on smaller open models (e.g.
  Qwen3-1.7B / 4B via Unsloth) trained on SEC filings + fund notes.
  Skipped for the MVP; base instruction-tuned models plus strong prompts
  carry most of the load.
- **Daily scheduler hooks** — auto-scan holdings at market open, push
  high-risk flags to a Slack channel.
- **PDF reports** — one-click investment memos combining thesis essay +
  charts + risk summary.
- **Candidate screener** — rank 500+ tickers by fundamental quality
  using the registry's normalized fields.

---

## Success Metrics (MVP targets)

- 85%+ accuracy on filing metric extraction
- 80%+ recall on portfolio risk flags vs analyst review
- < 2 min per full single-ticker analysis
- Full portfolio scan (10 holdings) completes end-to-end without crashes