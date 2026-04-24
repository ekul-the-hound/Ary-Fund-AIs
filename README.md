# Ary Fund — Hedge Fund AI Research Assistant

Quantamental research assistant for a student-run hedge fund
(long-term focus). Runs a local LLM against SEC filings, market data, and
macroeconomic indicators to parse filings, scan portfolio risks, generate
investment theses, and produce quant risk reports.

Everything runs locally. No paid APIs, no cloud LLM costs. The agent layer
is model-agnostic — it ships with a `mock` backend (offline, deterministic),
a `dev` backend (a small instruction-tuned model via Ollama), and a `prod`
slot for a larger model once it's validated.

---

## Overview

The system is organized as four independent, importable layers:

- **Data layer** (`data/sec_fetcher.py`, `data/market_data.py`,
  `data/macro_data.py`, `data/portfolio_db.py`, `data/pipeline.py`) — SEC
  EDGAR fetcher with CIK lookup and XBRL parsing, yfinance market data with
  technicals (RSI, MACD, Bollinger), FRED macro dashboard with a composite
  recession probability, SQLite portfolio DB with FIFO P&L, and a
  `DataPipeline` orchestrator that builds the structured context injected
  into the LLM.
- **Agent layer** (`agent/base_agent.py`, `agent/filing_analyzer.py`,
  `agent/risk_scanner.py`, `agent/thesis_generator.py`) — `ask_agent()` is
  the single entry point every downstream module calls. `AgentRequest` goes
  in, `AgentResponse` comes back, always with the same four keys in
  `generated_json`: `risks`, `thesis`, `price_direction`, `confidence`.
  Mock mode, Ollama success, and Ollama failure all produce the same shape,
  so the pipeline never crashes from a backend outage.
- **Quant layer** (`quant/var_es.py`, `quant/volatility.py`,
  `quant/regime.py`, `quant/position_sizing.py`) — historical + parametric
  + Monte Carlo VaR/ES, close-to-close and Yang-Zhang volatility estimators
  with forecasting, a pragmatic rule-based regime classifier (trend +
  drawdown + vol), and Kelly + volatility-target position sizing.
- **UI layer** (`ui/app.py`) — Streamlit dashboard: ticker picker,
  portfolio cards, price chart with MA / RSI / vol overlays, risk & thesis
  panels, macro context, and a debug view of the raw agent context JSON.

`config.py` centralizes model selection, API keys, data paths, logging,
and risk thresholds. `main.py` is the top-level CLI entry point.

---

## Model Configuration

The agent layer defines three model tags in `config.py`:

- **`mock`** — deterministic stub, no model server required. Used in tests
  and when Ollama is unreachable.
- **`dev`** — a small instruction-tuned model served via Ollama. Current
  default; lets the full pipeline run end-to-end on modest hardware while
  the `prod` stack is validated.
- **`prod`** — reserved for a larger Qwen-family model once thresholds are
  confirmed.

Switch models by editing `DEFAULT_AGENT_MODEL` in `config.py`. No other
code changes — callers route through `AGENT_MODELS`, never a raw model
string.

---

## Project Structure

```
.
├── config.py                  # model tags, API keys, paths, risk thresholds
├── main.py                    # CLI entry point; runs the full pipeline
│
├── agent/
│   ├── __init__.py
│   ├── base_agent.py          # AgentRequest / AgentResponse / ask_agent
│   ├── filing_analyzer.py     # 10-K / 10-Q metric extraction
│   ├── risk_scanner.py        # Portfolio-level risk flagging
│   └── thesis_generator.py    # BUY / HOLD / SELL thesis writer
│
├── data/
│   ├── __init__.py
│   ├── sec_fetcher.py         # SEC EDGAR REST + XBRL parser (rate-limited)
│   ├── market_data.py         # yfinance + technicals (RSI, MACD, BBands)
│   ├── macro_data.py          # FRED: rates, inflation, VIX, recession prob
│   ├── portfolio_db.py        # SQLite: positions, trades, snapshots
│   └── pipeline.py            # Daily refresh + agent context builder
│
├── quant/
│   ├── __init__.py
│   ├── var_es.py              # Historical / parametric / MC VaR & ES
│   ├── volatility.py          # Vol estimators + forecasting
│   ├── regime.py              # Trend + drawdown + vol regime classifier
│   └── position_sizing.py     # Kelly + volatility-targeted sizing
│
├── ui/
│   ├── __init__.py
│   └── app.py                 # Streamlit dashboard
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Shared pytest fixtures
│   ├── test_base_agent.py
│   ├── test_filing_analyzer.py
│   ├── test_risk_scanner.py
│   ├── test_thesis_generator.py
│   ├── test_main.py
│   └── test_end_to_end_smoke.py
│
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── setup.py
├── activate.bat               # Windows convenience: activate the venv
├── .env.example               # Template for local secrets
└── README.md
```

Caches and the portfolio database are created at runtime under `data/`
(e.g. `data/cache/`, `data/portfolio.db`) and are excluded from version
control.

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

### 3. Pull a model (skip for mock mode)

```bash
ollama serve                         # leave this running
ollama pull <model_name>             # see config.AGENT_MODELS for the current dev model
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
  key, just the name/email above.

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

Streamlit dashboard:

```bash
streamlit run ui/app.py
```

### Switching agent backends

In `config.py`:

```python
DEFAULT_AGENT_MODEL: str = "dev"    # "mock" | "dev" | "prod"
```

Or force per-call with `AgentRequest(..., model_tag="mock")`. When Ollama
is unreachable, `ask_agent` falls back to mock output automatically — the
four-key response contract is preserved either way.

---

## Running Tests

```bash
pytest                              # full suite
pytest -k filing_analyzer           # single module
pytest tests/test_end_to_end_smoke.py
pytest -x                           # stop at first failure
```

Pytest config lives in `pytest.ini` / `pyproject.toml`. Tests default to
`mock` mode, so they run offline and deterministically — the full suite
should pass in seconds.

The end-to-end smoke test runs the whole pipeline against a single ticker
with mocked LLM responses to prove the wiring holds.

---

## Current Status

**Built:**
- **Data layer (~3,600 lines):** SEC fetcher with CIK lookup and XBRL
  parsing, market data with technicals and multi-stock comparison, FRED
  macro dashboard with recession probability, SQLite portfolio DB with
  FIFO P&L, and the `DataPipeline` orchestrator.
- **Agent layer:** model-agnostic `ask_agent` core with mock/dev/prod
  routing, filing analyzer, risk scanner, thesis generator. Graceful
  fallback to mock when Ollama is unreachable.
- **Quant layer:** VaR & Expected Shortfall (historical + parametric +
  Monte Carlo), volatility estimators and forecasts, rule-based regime
  classifier, Kelly + volatility-target position sizing.
- **UI:** Streamlit dashboard with portfolio overview, price charts with
  overlays, risk & thesis panels, macro context, and raw-context debug
  view (~900 lines).
- **Tests:** per-module unit tests plus an end-to-end smoke test, all
  defaulting to mock mode.
- **Packaging:** `pyproject.toml`, `requirements.txt`, `setup.py`,
  `pytest.ini`, `.env` support via `python-dotenv`.

**In progress:**
- Promoting the `prod` model tag once validation is complete.
- Filling in prompt templates under the directory reserved by
  `config.AGENT_PROMPT_TEMPLATES_DIR`.

---

## Future Work

- **Fine-tuning** — LoRA fine-tune on SEC filings + fund notes. Skipped
  for the MVP; the base instruction-tuned model plus strong prompts gets
  most of the way there.
- **Daily scheduler** — auto-scan holdings at market open, push high-risk
  flags to a Slack channel.
- **PDF reports** — one-click investment memos combining thesis + charts +
  risk summary.
- **Candidate screener** — rank 500+ tickers by fundamental quality.

---

## Success Metrics (MVP targets)

- 85%+ accuracy on filing metric extraction
- 80%+ recall on portfolio risk flags vs analyst review
- < 2 min per full single-ticker analysis
- Full portfolio scan (10 holdings) completes end-to-end without crashes
