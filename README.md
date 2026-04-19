# Ary Fund — Hedge Fund AI Research Assistant

Quantamental research assistant for a student-run hedge fund (~$6M AUM,
long-term focus). Runs a local LLM against SEC filings, market data, and
macroeconomic indicators to parse filings, scan portfolio risks, generate
investment theses, and produce quant risk reports.

Everything runs locally. No paid APIs, no cloud LLM costs. The agent layer
is model-agnostic — it ships with a `mock` backend (offline, deterministic),
a `dev` backend (Phi-3-mini via Ollama, fits the RTX 2080), and a `prod`
slot reserved for Qwen3.

---

## Overview

The system is organized as four independent, importable layers. The repo is
currently flat (all modules at the root) — module names reflect which layer
they belong to:

- **Data layer** (`sec_fetcher.py`, `market_data.py`, `macro_data.py`,
  `portfolio_db.py`, `pipeline.py`) — SEC EDGAR fetcher with CIK lookup and
  XBRL parsing, yfinance market data with technicals (RSI, MACD, Bollinger),
  FRED macro dashboard with a composite recession probability, SQLite
  portfolio DB with FIFO P&L, and a `DataPipeline` orchestrator that builds
  the structured context injected into the LLM.
- **Agent layer** (`base_agent.py`, `filing_analyzer.py`, `risk_scanner.py`,
  `thesis_generator.py`) — `ask_agent()` is the single entry point every
  downstream module calls. `AgentRequest` goes in, `AgentResponse` comes
  back, always with the same four keys in `generated_json`: `risks`,
  `thesis`, `price_direction`, `confidence`. Mock mode, Ollama success, and
  Ollama failure all produce the same shape, so the pipeline never crashes
  from a backend outage.
- **Quant layer** (`var_es.py`, `volatility.py`, `regime.py`,
  `position_sizing.py`) — historical + parametric + Monte Carlo VaR/ES,
  close-to-close and Yang-Zhang volatility estimators with forecasting, a
  pragmatic rule-based regime classifier (trend + drawdown + vol), and Kelly
  + volatility-target position sizing.
- **UI layer** (`app.py`) — Streamlit dashboard: ticker picker, portfolio
  cards, price chart with MA/RSI/vol overlays, risk & thesis panels, macro
  context, and a debug view of the raw agent context JSON.

`config.py` centralizes model selection, API keys, data paths, logging, and
risk thresholds. `main.py` is the top-level CLI entry point.

---

## Hardware & Model

- **GPU:** RTX 2080 (8GB VRAM)
- **RAM:** 64GB
- **OS:** Windows 10
- **Python:** 3.11+
- **Current default LLM:** `phi3:3.8b` via Ollama (`dev` tag) — fits
  comfortably in the 2080's VRAM budget and validates the full pipeline
  end-to-end while the Qwen stack is still being sized.
- **Future production LLM:** `qwen3:14b-instruct` (`prod` tag) once
  hardware/quant thresholds are confirmed; the 30B MoE variant is a
  stretch target.
- **Offline default:** `mock` — returns deterministic JSON, no model
  server needed. Used in tests and when Ollama is unreachable.

Switch models by editing `DEFAULT_AGENT_MODEL` in `config.py`. No other
code needs to change — callers route through `AGENT_MODELS`, never a raw
model string.

---

## Project Structure

```
.
├── config.py                  # model tags, API keys, paths, risk thresholds
├── main.py                    # CLI entry point; runs the full pipeline
├── app.py                     # Streamlit dashboard (run via streamlit)
│
├── sec_fetcher.py             # SEC EDGAR REST + XBRL parser (rate-limited)
├── market_data.py             # yfinance + technicals (RSI, MACD, BBands)
├── macro_data.py              # FRED: rates, inflation, VIX, recession prob
├── portfolio_db.py            # SQLite: positions, trades, snapshots
├── pipeline.py                # Daily refresh + agent context builder
│
├── base_agent.py              # AgentRequest / AgentResponse / ask_agent
├── filing_analyzer.py         # 10-K / 10-Q metric extraction
├── risk_scanner.py            # Portfolio-level risk flagging
├── thesis_generator.py        # BUY / HOLD / SELL thesis writer
│
├── var_es.py                  # Historical / parametric / MC VaR & ES
├── volatility.py              # Vol estimators + forecasting
├── regime.py                  # Trend + drawdown + vol regime classifier
├── position_sizing.py         # Kelly + volatility-targeted sizing
│
├── conftest.py                # Shared pytest fixtures
├── test_base_agent.py
├── test_filing_analyzer.py
├── test_risk_scanner.py
├── test_thesis_generator.py
├── test_main.py
├── test_end_to_end_smoke.py
│
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── setup.py
├── activate.bat               # Windows: activate the venv in one command
├── .env                       # Local secrets (see .env.example)
└── README.md
```

A `data/` subfolder is created at runtime by `config.py` to hold the
portfolio DB, per-source caches, and LLM prompt templates. It is not
committed — `.gitignore` excludes it.

---

## Installation

### 1. Prerequisites

- Python 3.11 ([python.org/downloads](https://python.org/downloads) — check
  "Add to PATH" during install)
- Git
- Ollama ([ollama.com](https://ollama.com)) — only needed if you run
  anything other than `mock` mode

CUDA is not required for the current `dev` model (`phi3:3.8b`) — Ollama
handles acceleration automatically. CUDA 12.1 becomes relevant only when
moving to the Qwen `prod` model.

### 2. Clone and install

```bash
git clone <your-repo-url> ary-fund
cd ary-fund

python -m venv .venv
.venv\Scripts\activate               # Windows
# source .venv/bin/activate          # Linux/macOS

pip install -r requirements.txt
```

Or, for an editable install with dev tools:

```bash
pip install -e ".[dev]"
```

On Windows there's also `activate.bat` that wraps the venv activation.

### 3. Pull a model (optional — skip for mock mode)

```bash
ollama serve                 # leave this running in its own terminal
ollama pull phi3:3.8b        # current dev model, ~2.3GB
```

### 4. API keys

Copy `.env.example` → `.env` and fill in:

```dotenv
FRED_API_KEY=your_fred_key_here
SEC_AGENT_NAME=Your Name
SEC_AGENT_EMAIL=you@school.edu
```

- **FRED** is free: [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- **SEC EDGAR** requires a descriptive User-Agent with contact info — no
  key, just the name/email above.

`config.py` loads `.env` automatically via `python-dotenv` on import.

### 5. Verify the stack

```bash
# Mock mode — no Ollama required
python -c "from base_agent import ask_agent, AgentRequest; print(ask_agent(AgentRequest(prompt='ping', context={}, model_tag='mock')).generated_json)"

# If Ollama is running with phi3:3.8b pulled
python -c "import ollama; print(ollama.chat(model='phi3:3.8b', messages=[{'role':'user','content':'ping'}])['message']['content'])"

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
python pipeline.py
```

Streamlit dashboard:

```bash
streamlit run app.py
```

### Switching agent backends

In `config.py`, change:

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
pytest test_end_to_end_smoke.py
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
  macro dashboard with recession probability, SQLite portfolio DB with FIFO
  P&L, and the `DataPipeline` orchestrator.
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
- Swapping the `prod` tag onto Qwen3 once VRAM/quant thresholds are
  confirmed.
- Migrating the flat module layout into proper `data/`, `agent/`, `quant/`,
  `ui/` subpackages.
- Filling in prompt templates under `data/prompts/agent/` (reserved by
  `config.AGENT_PROMPT_TEMPLATES_DIR`).

---

## Future Work

- **Fine-tuning** — LoRA fine-tune on SEC filings + fund notes. Skipped for
  the MVP; the base instruction-tuned model plus strong prompts gets most
  of the way there.
- **Daily scheduler** — auto-scan holdings at market open, push high-risk
  flags to Slack.
- **PDF reports** — one-click investment memos combining thesis + charts +
  risk summary.
- **Candidate screener** — rank 500+ tickers by fundamental quality.
- **Hardware upgrade path** — RTX 4090 unlocks Qwen 70B Q4 quantization
  (~10× throughput over the current stack).

---

## Success Metrics (MVP targets)

- 85%+ accuracy on filing metric extraction
- 80%+ recall on portfolio risk flags vs analyst review
- < 2 min per full single-ticker analysis
- Full portfolio scan (10 holdings) completes end-to-end without crashes