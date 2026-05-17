# ARY QUANT — Hedge Fund AI Research Assistant

Quantamental research platform for a student-run hedge fund (long-term
focus). Runs a local LLM against SEC filings, market data, macroeconomic
indicators, and a normalized data registry to screen stocks, analyze
filings, scan portfolio risks, generate multi-page institutional
investment memos, produce PDF reports, and run probabilistic quant
models — all from a Streamlit dashboard.

Everything runs locally. No paid APIs beyond a free FRED key. No cloud
LLM costs.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Layer](#data-layer)
4. [Agent Layer](#agent-layer)
5. [Quant Layer](#quant-layer)
6. [Screener](#screener)
7. [Report Layer (PDF)](#report-layer-pdf)
8. [UI Layer (Streamlit)](#ui-layer-streamlit)
9. [Model Configuration](#model-configuration)
10. [Project Structure](#project-structure)
11. [Installation](#installation)
12. [Running the Project](#running-the-project)
13. [Running Tests](#running-tests)
14. [Data Sources Reference](#data-sources-reference)
15. [Current Status](#current-status)
16. [Future Work](#future-work)
17. [Success Metrics](#success-metrics)

---

## Overview

ARY QUANT is organized into seven independent layers. Each layer is
importable on its own and talks to the others through a small set of
well-defined contracts:

- **Data layer** — ingests and normalizes everything from external
  sources into a SQLite `DataRegistry`. The registry is the single source
  of truth for every downstream consumer.
- **Agent layer** — LLM-powered analysis modules that read from the
  registry-backed context snapshot and produce structured verdicts, risk
  flags, and investment memos.
- **Quant layer** — pure-function statistical models (no I/O) that take
  price series and return structured dicts.
- **Screener** — real-time stock screening over a ~600-ticker US
  universe with live prices and lazy-loaded fundamentals.
- **Report layer** — PDF investment memo generator built on ReportLab,
  driven by snapshot context and chart artifacts.
- **UI layer** — Streamlit dashboard wiring all layers together.
- **Test layer** — comprehensive `test_all.py` with all tests defaulting
  to mock mode for offline, deterministic runs.

---

## Architecture

### Registry-first context contract

`pipeline.build_agent_context(ticker)` is the gateway from raw data to
the LLM. It returns a single registry-backed snapshot with the following
top-level keys:

| Key | Type | Contents |
|---|---|---|
| `ticker` | str | Ticker symbol |
| `as_of` | str | ISO-format snapshot date |
| `price` | float\|None | Latest adjusted close |
| `prices` | dict | `last`, `change_pct`, `market_cap`, `52w_high`, `52w_low` |
| `metrics` | dict | ~15 fundamentals: `trailing_pe`, `grossMargins`, `freeCashflow`, `longTermDebt`, etc. |
| `filings` | list[dict] | Recent 10-K, 10-Q, 8-K filing metadata |
| `macro` | dict | `vix`, `fed_funds_rate`, `yield_curve_spread`, `recession_prob`, etc. |
| `portfolio` | dict | Position size, weight, cost basis, shares (empty if not held) |
| `sentiment` | dict | `wsb_mentions_24h`, `wsb_score`, `news_count_7d`, `news_tone_7d` |
| `geo_signals` | dict | Sanctions flags, supply chain signals |
| `risk_scores` | dict | `macro_stress`, `supply_chain`, `sanctions_pressure`, `commodity_sensitivity`, `energy_crisis` |
| `derived_signals` | dict | `rsi_14`, `macd_hist`, `atr_14`, `sma_50`, `sma_200`, `realized_vol_30d`, `drawdown`, `regime` |
| `provenance` | dict | Per-field `{source_id, as_of, confidence}` metadata |
| `freshness` | dict | Per-section latest-as-of timestamps |

Absence semantics are explicit: scalars → `None`, dicts → `{}`,
lists → `[]`. The context builder never substitutes zero, empty string,
or a fabricated value for missing data.

### Master pipeline loop (main.py)

For every ticker in `config.WATCHLIST` (or the `--tickers` CLI flag),
`main.py` runs the following steps in order:

1. `pipeline.run_daily_refresh(tickers, db_path, config)` — once,
   upfront. Ingests SEC filings, market data, and macro series.
2. `pipeline.build_agent_context(ticker, db_path, config)` — builds
   the registry-backed snapshot.
3. `filing_analyzer.summarize_filings_by_year(...)` — groups filings
   by year and form type.
4. `filing_analyzer.extract_key_metrics_for_agent(...)` — extracts
   the normalized metric dict fed to the LLM.
5. `base_agent.ask_agent(AgentRequest(...))` — LLM call (or mock).
   Returns `AgentResponse` with `generated_json` containing `risks`,
   `thesis`, `price_direction`, `confidence`.
6. `risk_scanner.compute_risk_flags(...)` — rule-based risk engine
   using the agent's risk list plus fundamental and macro thresholds.
7. `thesis_generator.generate_thesis(...)` — heuristic BUY/HOLD/SELL
   verdict from metrics and risk flags.
8. `portfolio_db.save_agent_opinion(ticker, opinion, db_path)` —
   persists the final verdict.

Errors on any single ticker are logged and skipped; the batch continues.
A compact summary table is printed at the end.

---

## Data Layer

### DataRegistry (`data_registry.py`)

The registry is the normalized, per-field store that sits between raw
fetchers and every downstream consumer. Key properties:

- ~100 canonical fields organized into namespaces:
  `ticker.fundamental.*`, `ticker.price.*`, `ticker.signal.*`,
  `ticker.sentiment.*`, `ticker.ownership.*`, `ticker.risk.*`,
  `global.*`, `commodity.*`, `freight.*`
- Per-field source priority list (e.g. `ticker.fundamental.revenue_ttm`
  prefers `sec_xbrl` → `finnhub` → `yfinance`)
- Idempotent upserts with conflict resolution (highest-priority source
  wins; tie → most recent wins)
- Confidence scoring per data point (0.0–1.0)
- `reg.snapshot(ticker, fields)` returns a flat dict in one SQL round
  trip — no raw-table reads by downstream consumers
- `reg.latest(field, ticker)` returns the single highest-priority value
  for a field

### SEC EDGAR (`sec_fetcher.py`)

The most data-rich module. All requests are rate-limited to ~8/sec
(SEC allows 10/sec).

**Filing types pulled:**
- 10-K (annual), 10-Q (quarterly), 8-K (material events), DEF 14A
  (proxy), S-1, S-3/S-3ASR, 424B2/3/5 (offerings)
- 13F-HR (institutional holdings)
- SC 13D / SC 13G (ownership > 5%)
- Form 4 (insider transactions)

**XBRL companyfacts endpoint** — the highest-yield SEC data source.
Canonical fields extracted per company:

| Field | XBRL concept(s) |
|---|---|
| Revenue TTM | `Revenues`, `RevenueFromContractWithCustomer…` |
| Net income TTM | `NetIncomeLoss`, `ProfitLoss` |
| EPS diluted TTM | `EarningsPerShareDiluted` |
| CapEx TTM | `PaymentsToAcquirePropertyPlantAndEquipment` |
| R&D TTM | `ResearchAndDevelopmentExpense` |
| Effective tax rate | `EffectiveIncomeTaxRateContinuingOperations` |
| Goodwill | `Goodwill` |
| Pension liability | `DefinedBenefitPensionPlanLiabilitiesNoncurrent` |
| Shares diluted | `WeightedAverageNumberOfDilutedSharesOutstanding` |
| SBC TTM | `ShareBasedCompensation`, `AllocatedShareBasedCompensationExpense` |
| Long-term debt | `LongTermDebtNoncurrent`, `LongTermDebt` |
| Total assets | `Assets` |
| Total liabilities | `Liabilities` |
| FCF TTM | Derived: operating cash flow − CapEx |

**Form 4 parsing:**
- Open-market buy codes: P (purchase), A (grant)
- Open-market sell codes: S (sale), F (tax payment), D (sale to issuer)
- 10b5-1 plan reference detection
- Insider net USD in rolling 30-day windows

**Ownership filings:**
- 13D / 13G: activist and passive ownership > 5%
- 13F: institutional holdings snapshot (informationtable XML)

**8-K event detection:**
- Item 5.02 → officer/director change (CEO/CFO turnover)
- Item 5.03 → bylaws change
- Item 8.01 → material events (buyback announcements, ratings)
- Item 1.01 → material agreement
- Item 2.02 → results of operations
- Item 3.02 → unregistered securities
- Share buyback, secondary offering, ATM offering (form-type filters)

### Market Data (`market_data.py`)

- OHLCV price history: default 5 years, configurable via
  `config.PRICE_HISTORY_YEARS`
- Technical indicators: RSI-14, MACD (12/26/9), Bollinger Bands (20/2σ),
  ATR-14, SMA-20/50/200, EMA-12/26
- Fundamentals: P/E, forward P/E, EV/EBITDA, PEG, P/S, P/B, FCF yield,
  gross/operating/net margins, ROE, ROA, debt/equity, current ratio,
  revenue growth, EPS growth, dividend yield, payout ratio
- Financial statements: annual and quarterly income statement, balance
  sheet, cash flow statement
- Analyst consensus: recommendation key (`strong_buy` → `strong_sell`),
  price target, number of analysts
- ETF holdings API: `set_etf_holdings(etf_ticker, holdings)` — called
  by the ETF loaders below
- SQLite fundamentals cache: 24-hour TTL

### Macro Data (`macro_data.py`)

All series pulled from FRED (St. Louis Fed):

| Series | Description |
|---|---|
| VIXCLS | VIX — CBOE Volatility Index |
| VXVCLS | VIX3M — 3-month VIX |
| VIX3M/VIX term ratio | Computed: contango vs backwardation |
| BAMLH0A0HYM2 | ICE BofA US HY option-adjusted spread |
| BAMLC0A0CM | ICE BofA US IG option-adjusted spread |
| RECPROUSM156N | Smoothed US recession probability |
| UMCSENT | University of Michigan consumer sentiment |
| STLFSI4 | St. Louis Fed financial stress index |
| T10Y2Y | 10y−2y Treasury spread (yield curve) |
| FEDFUNDS | Effective federal funds rate |
| CPIAUCSL | CPI all urban consumers |
| PCEPI | PCE price index |

Composite recession probability score is derived from the raw FRED series
plus the yield curve and financial stress index.

### Sentiment & News (`sentiment_news.py`)

Sources (all free, all best-effort — a single failure doesn't block
others):

| Source | Signal |
|---|---|
| yfinance news endpoint | Recent headlines per ticker |
| Stocktwits public API | Message volume + bullish/bearish sentiment score |
| Google News RSS | Broad headline coverage when others are thin |
| GDELT 2.0 events | Financial-themed event counts (geopolitical relevance) |
| Reddit r/wallstreetbets | Mention counts (best-effort scrape) |
| VADER sentiment | Polarity scoring for news headlines (FinBERT optional) |

Outputs written to the registry: `ticker.sentiment.wsb_mentions_24h`,
`ticker.sentiment.wsb_score`, `ticker.sentiment.news_count_7d`,
`ticker.sentiment.news_tone_7d`.

### Geopolitical & Supply Chain (`geo_supply.py`)

| Source | What it provides |
|---|---|
| OFAC SDN list (US Treasury) | Sanctioned entity additions, daily diff |
| EU Consolidated Sanctions | JSON feed of EU-designated entities |
| UK HM Treasury Sanctions | CSV of UK-designated entities |
| UN Security Council Sanctions | JSON feed |
| GDELT 2.0 | Global financial-themed event counts (hourly) |
| EIA Open Data | Oil, natgas, electricity spot prices and inventories |
| Stooq | Baltic Dry Index proxy |
| USAspending.gov | Government contract awards |

Sanctions diffs are computed daily (today's SDN set vs. yesterday's) and
written as `sanctioned_entity_added` events on the registry.

### ETF Holdings (`etf_providers.py` + `etf_holdings_loader.py`)

Three production adapters:
- **`IsharesProvider`** — BlackRock iShares CSV format
- **`SpdrProvider`** — State Street SPDR XLSX / CSV format
- **`GenericIssuerProvider`** — column-mapped CSV for any flat tabular
  file (Vanguard, Invesco, etc.), configured via `ProviderConfig`
- **`LocalFileProvider`** — ingests a file the user downloaded manually
  (most reliable in production — issuer URLs rotate frequently)

Normalized `RawHolding` records are written through
`market_data.set_etf_holdings()`.

### Factor Returns (`factor_returns_loader.py`)

Pulls from Ken French's daily data library:
- `F-F_Research_Data_Factors_daily` → `Mkt-RF`, `SMB`, `HML`, `RF`
- `F-F_Momentum_Factor_daily` → `MOM`

Written into the `factor_returns` table so
`DerivedSignals.recompute_factor_exposures()` has a regression target.
Run once after setup, then on the weekly cadence.

### Derived Signals (`derived_signals.py`)

Pure-Python computation over the registry. No external I/O.

1. **Technical regime signals** — 30-day realized vol (annualized),
   252-day peak-to-current drawdown
2. **Relative strength vs sector ETFs** — 20/60/120-day return spread
   vs the relevant SPDR ETF (11 sectors: XLK, XLF, XLV, XLI, XLE,
   XLP, XLY, XLU, XLB, XLRE, XLC)
3. **Sector heatmap** — 5/20/60-day returns for all 11 SPDR ETFs
4. **Factor exposures** — Fama-French 5 + momentum regression when
   `factor_returns` table is populated
5. **Composite risk scores** — `macro_stress`, `sanctions_pressure`,
   `commodity_sensitivity`, `energy_crisis_score`, `supply_chain_score`
   (each ∈ [0,1])

### Global Risk Pulse (`global_risk_pulse.py`)

Market-wide composite risk index replacing the previous stub.

Pipeline:
1. Selects a universe (watchlist / positions / full registry / custom)
2. Filters for staleness and minimum field coverage
3. Builds a weighting vector: equal / market-cap / liquidity / hybrid
4. Computes six independently normalized subcomponents:
   - **volatility** — cross-sectional weighted mean of 30d realized vol,
     z-scored against long-run baseline
   - **breadth** — `-(2 × pct_above_50dma − 1)`; blends in
     `pct_positive_5d`
   - **correlation** — median pairwise correlation of 60d daily returns;
     high correlation = systemic risk
   - **concentration** — `2 × HHI − 1` of the weighting vector
   - **dispersion** — cross-sectional std-dev of per-ticker risk scores
   - **macro_regime** — rule-based blend of VIX, 2s10s, recession prob,
     and financial stress index
5. Weighted sum → `pulse_score ∈ [-1, +1]`
   (positive = risk-off / stressed; negative = risk-on / calm)
6. Returns structured object: score, components, weights, coverage,
   confidence, timestamp, provenance, diagnostics

Scale convention matches the St. Louis Fed Financial Stress Index.
Correlation matrix is sub-sampled at 500 tickers (deterministic by
ticker hash) to keep runtime bounded.

### Refresh Scheduler (`refresh_scheduler.py`)

Cadence-driven orchestrator. Each task is gated by `is_due()` against
`last_run` so re-running is idempotent. Lazy module loading means a
single missing dependency doesn't block the other tasks.

| Cadence | Tasks |
|---|---|
| **Hourly** (every ~50 min) | Prices, social/news sentiment, GDELT geopolitical events, sanctions spot-check |
| **Daily** (~20h cooldown) | SEC filings, FRED macro, fundamentals, Ken French factors |
| **Weekly** (~6d cooldown) | 13F holdings, factor exposure recompute, ADV registrations |
| **Market open scan** | Holdings risk scan + Slack push for elevated flags |
| **Event-driven** | 8-K material event, rating change, ownership > 5% |

Suggested cron entries are included in the module docstring. The
scheduler is designed to be invoked from cron, APScheduler, or a
Streamlit button — it does not own a process loop.

### Notifiers (`notifiers.py`)

Slack incoming webhook adapter. `notify_slack(message)` posts to the
configured workspace. Best-effort: a network failure or missing
`SLACK_WEBHOOK_URL` logs a warning but does not raise, so the risk scan
completes regardless. Additional channels (Discord, email, PagerDuty)
are slotted in without touching call sites.

---

## Agent Layer

### Base Agent (`base_agent.py`)

`ask_agent(AgentRequest)` is the single LLM call site. Nothing else in
the codebase calls Ollama directly.

- `AgentRequest` fields: `prompt`, `context`, `model_tag`, `temperature`,
  `max_tokens`
- `AgentResponse` fields: `generated_json`, `raw_text`, `model_used`,
  `elapsed_ms`, `error`
- `generated_json` always contains: `risks` (list), `thesis` (str),
  `price_direction` (str), `confidence` (float) — same shape whether
  the backend is mock, Ollama success, or Ollama failure
- JSON mode enforced via Ollama's `format="json"` parameter
- Token budget estimated before dispatch; requests exceeding
  `config.MAX_TOKENS` are truncated at the prompt level
- Hard timeout: `config.AGENT_TIMEOUT` seconds (default 30s); exceeded
  calls fall back to mock output
- Three backends: `mock` (deterministic stub), `dev` (`phi3:3.8b`),
  `prod` (`qwen3:14b-instruct`)
- Ollama base URL configurable via `OLLAMA_BASE_URL` env var (supports
  remote GPU workstation on LAN)

### Filing Analyzer (`filing_analyzer.py`)

Two public functions consumed by `main.py`:

- `summarize_filings_by_year(filings)` — groups filing metadata by year
  and form type; flags material 8-K events (Item 5.02, buybacks, etc.)
- `extract_key_metrics_for_agent(metrics, filings)` — normalizes the
  fundamentals dict into the flat schema the LLM prompt expects, derives
  FCF margin, R&D intensity, SBC/revenue, and dilution YoY from raw
  XBRL fields

### Data-Point Analyzer (`data_point_analyzer.py` + `data_point_analyzer_section.py`)

Per-field signal interpretation module used by the Streamlit UI to
render contextualized commentary on individual metrics. Each metric gets
a plain-English interpretation (e.g. "Gross margin of 43% is above the
S&P 500 median — strong pricing power relative to COGS").
`data_point_analyzer_section.py` handles section-level grouping and
rendering within the dashboard.

### Risk Scanner (`risk_scanner.py`)

Rule-based risk-flag engine. No LLM calls, no DB writes. Output shape:

```python
{
  "levels": {
      "fundamental": "HIGH" | "MEDIUM" | "LOW",
      "macro":       "HIGH" | "MEDIUM" | "LOW",
      "market":      "HIGH" | "MEDIUM" | "LOW",
      "agent":       "HIGH" | "MEDIUM" | "LOW",
      "combined":    "HIGH" | "MEDIUM" | "LOW",
  },
  "reasons": {
      "fundamental": [str, ...],
      "macro":       [str, ...],
      "market":      [str, ...],
      "agent":       [str, ...],
  },
}
```

Risk thresholds (all tunable in `config.RISK_THRESHOLDS`):

| Threshold | Default |
|---|---|
| Debt/EBITDA high | 3.0× |
| Debt/EBITDA medium | 2.0× |
| Interest coverage low | 2.0× |
| FCF yield low | 2% |
| Recession prob high | 60% |
| Recession prob medium | 35% |
| VIX high | 28 |
| VIX medium | 20 |
| Drawdown high | 25% |
| Drawdown medium | 15% |
| Realized vol high | 45% annualized |
| Realized vol medium | 30% annualized |

### Thesis Generator (`thesis_generator.py`)

Heuristic BUY / HOLD / SELL verdict engine. Combines fundamental
quality scores, macro backdrop assessment, and risk-flag levels into a
structured verdict dict including: `outlook`, `direction`, `confidence`,
`risks`, `opportunities`, `bias_scores`.

### Thesis Essay (`thesis_essay.py`)

LLM-powered 2+ page institutional investment memo. Additive to the
heuristic thesis — it interprets rather than summarizes.

8-section structure:
1. **Executive Summary** — single thesis statement in the form
   "[Company] is [quality], but at [valuation] [return expectation]
   unless [key condition]." Plus biggest risk and strongest catalyst.
2. **Business & Financial Performance** — metric-by-metric connected
   narrative with implied meaning (margin structure, cash conversion,
   capital efficiency, ROIC)
3. **Valuation** — P/E, forward P/E, EV/EBITDA vs S&P 500 average
   (~20–22× trailing P/E), sector, and historical range
4. **Filings & Management Signal** — management tone, notable
   disclosures, Form 4 insider activity
5. **Macro & Industry Context** — specific macro variable transmission
   mechanisms to revenue, margins, or the multiple
6. **Risks** — top 3 ranked risks with trigger (specific number),
   transmission mechanism, and price impact
7. **Catalysts** — top 2–3 catalysts with threshold, timeframe, and
   price move rationale
8. **Verdict** — clear BUY / HOLD / AVOID with one-year return
   expectation and the single data point that would change the verdict

Data-gap protocol: if a required metric is absent, the essay
acknowledges it in one sentence and moves on — no fabricated numbers.
Fallback: if Ollama is unavailable, a deterministic essay assembled from
the heuristic numbers is returned, so the pipeline always produces
usable prose.

### Thesis Review (`thesis_review.py`)

Committee-style self-review pass. A second LLM call (or deterministic
fallback) reads the essay and the raw metrics as a senior portfolio
manager, scoring each of the 8 sections 1–10 and providing the single
most important fix per section — including a specific number, threshold,
or comparison and a one-sentence explanation of why it changes
estimates, multiples, or the stock.

### Chat Agent (`chat.py`)

Streaming conversational interface separate from the JSON-mode agents.

- Tokens stream to the UI as the model generates them
- Every turn's system prompt carries the full current context: essay +
  risk flags + thesis + key metrics + macro snapshot
- Per-ticker conversation history keyed in `st.session_state`
- Switching tickers resets to a fresh conversation
- ARY QUANT persona: buy-side analyst, concise, evidence-anchored,
  skeptical of consensus narratives, never invents financials
- Fallback to canned response when Ollama is unreachable

---

## Quant Layer

All modules are pure functions — no I/O, no global state, graceful
NaN/empty/short-series handling, structured dict returns.

### Core risk models

**`var_es.py`** — Value at Risk & Expected Shortfall

Sign convention: VaR and ES are reported as positive numbers
representing losses (e.g. a one-day 95% VaR of 0.023 = 2.3% potential
loss). Three methods:
- `historical_var(returns, confidence)` — empirical quantile
- `parametric_var(returns, confidence)` — Gaussian assumption
- `monte_carlo_var(returns, confidence, n_sims)` — simulation-based
- `var_es_report(returns)` — combined dict with all three at 95% and 99%

**`volatility.py`**

- `compute_returns(prices)` — log returns
- `rolling_volatility(returns, window)` — rolling std
- `realized_volatility(returns, window)` — annualized realized vol
- `annualize_volatility(vol, periods)` — √252 scaling
- `forecast_volatility(returns)` — EWMA-based 1-step forecast
- `garch_forecast(returns)` — GARCH(1,1) via `arch` package (optional)

**`regime.py`** — Rule-based regime classifier

Combines three signals deterministically:
1. Trend: fast vs slow MA crossover
2. Drawdown: peak-to-current decline over window
3. Volatility: current rolling vol vs its historical median

Labels: BULL / BEAR / CHOP / CRASH. Includes confidence score.
Designed for testability — no probabilistic fitting, fully deterministic.

**`regime_hmm.py`** — Gaussian HMM regime classifier

- `fit_hmm_regime(returns, n_states)` — fits 2 or 3-state Gaussian HMM
- Returns per-date state labels, state means/variances, transition matrix
- `hmmlearn` is an optional dependency; module degrades gracefully

### Position sizing

**`kelly.py`** — Kelly criterion (4 variants)
- `kelly_discrete(win_prob, win_frac, lose_frac)` — classical binary
- `kelly_continuous(mean_return, variance)` — continuous approximation
- `kelly_multi_asset(means, cov_matrix)` — multi-asset Kelly vector
- `kelly_with_drawdown_cap(...)` — capped at configurable max fraction
- `kelly_report(...)` — structured dict summarizing all variants

**`position_sizing.py`** — Combined sizing report
- `kelly_fraction(edge, odds)` — edge/odds convenience wrapper
- `volatility_target_size(target_vol, realized_vol, capital)` — scales
  position to hit target portfolio vol
- `position_sizing_report(...)` — opinionated combined recommendation

### Options & derivatives

**`black_scholes.py`**
- `bs_price(S, K, T, r, sigma, option_type)` — European call/put price
- `bs_greeks(S, K, T, r, sigma)` — delta, gamma, theta, vega, rho
- `implied_vol(market_price, S, K, T, r)` — Newton-Raphson IV solver

**`avellaneda_stoikov.py`** — Optimal market-making
- `reservation_price(mid, q, gamma, sigma, T)` — inventory-adjusted
  mid price
- `optimal_spread(gamma, sigma, T, k)` — bid/ask half-spread
- `compute_quotes(...)` — full bid/ask from reservation price + spread
- `calibrate_k_from_fills(fills)` — estimates order-arrival intensity
  from historical fill data

**`sabr.py`** — SABR stochastic volatility model for smile fitting

**`longstaff_schwartz.py`** — LSM algorithm for American option pricing
via Monte Carlo regression

### Stochastic processes

**`gbm.py`** — Geometric Brownian Motion
- `estimate_gbm_params(prices)` — MLE for μ and σ
- `simulate_gbm(S0, mu, sigma, T, n_paths, n_steps)` — path simulation
- `gbm_from_prices(prices, T, n_paths)` — convenience wrapper

**`hurst.py`** — Long-memory and mean-reversion test
- `hurst_exponent(prices)` — R/S analysis, returns H ∈ [0,1]
  (H < 0.5 = mean-reverting, H ≈ 0.5 = random walk, H > 0.5 =
  trending)
- `rolling_hurst(prices, window)` — rolling H over time

**`ou_process.py`** — Ornstein-Uhlenbeck mean reversion
- `fit_ou_process(prices)` — MLE for κ (mean reversion speed),
  μ (long-run mean), σ (diffusion)
- Returns half-life of mean reversion in trading days

**`poisson.py`** — Jump intensity models
- `poisson_mle(times)` — constant intensity λ from inter-arrival times
- `exponential_decay_intensity(t, mu, alpha, beta)` — Hawkes kernel
- `hawkes_mle(times)` — self-exciting Hawkes process MLE
- `simulate_poisson(lam, T)` / `simulate_hawkes(mu, alpha, beta, T)`
- `detect_jumps(returns, threshold)` — flags returns exceeding
  N-sigma as jump events

### Portfolio construction

**`hrp.py`** — Hierarchical Risk Parity (Lopez de Prado)

**`rmt.py`** — Random Matrix Theory noise cleaning for correlation
matrices (Marchenko-Pastur distribution)

**`mst.py`** — Minimum Spanning Tree of the correlation matrix for
portfolio structure visualization

**`portfolio_construction.py`** — General portfolio optimization
utilities including mean-variance, maximum Sharpe, and risk parity

**`sequential_monte_carlo.py`** — Particle filter / SMC for online
state estimation

### Research / exotic models

These modules are standalone research tools, not wired into the main
pipeline:

| Module | What it models |
|---|---|
| `monte_carlo.py` | General-purpose Monte Carlo with variance reduction |
| `wavelet_regimes.py` | Wavelet decomposition-based regime detection |
| `lyapunov.py` | Lyapunov exponent (chaos / sensitivity to initial conditions) |
| `lempel_ziv.py` | Lempel-Ziv complexity (market randomness measure) |
| `fft_analysis.py` | FFT spectral analysis of price series |
| `ergodicity.py` | Ergodicity economics: time-average vs ensemble-average growth |
| `sandpile.py` | Bak-Tang-Wiesenfeld self-organized criticality model |
| `omori.py` | Omori law: aftershock clustering / volatility clustering analog |
| `girsanov.py` | Girsanov measure change for change-of-numeraire pricing |
| `gan_synthetic.py` | GAN-based synthetic financial time series generation |
| `wave_function_collapse.py` | Quantum-mechanics-inspired price uncertainty visualization |
| `yield_curve_3d.py` | 3-D yield curve surface visualization |

---

## Screener

TradingView-style stock screener tab (`screener.py`).

**Universe:** ~600 tickers — S&P 500 plus curated large/mid caps not in
the index. Free-text search accepts any valid US ticker symbol.

**Data pipeline:**
1. `_fetch_live_prices_batch(symbols)` — single `yf.download` call
   returns last-close, prev-close, and volume for the entire universe.
   Cached 5 minutes via `@st.cache_data(ttl=300)`.
2. `_fetch_fundamentals_one(symbol)` — wraps `MarketData.get_fundamentals`,
   lazy-loaded only for the visible/filtered subset.
   SQLite-cached 24 hours.

**Columns displayed:**
Price, change %, volume, market cap, P/E, forward P/E, PEG, P/S, P/B,
EV/EBITDA, revenue, gross profit, EBITDA, net income, FCF, operating
cash flow, total debt, debt/equity, current ratio, ROE, ROA, profit
margin, gross margin, operating margin, revenue growth, EPS growth,
dividend yield, payout ratio, ex-dividend date, analyst rating.

**Filter pills (15):**
Sector, market cap range, P/E range, forward P/E, PEG, dividend yield,
revenue growth, EPS growth, gross margin, debt/equity, analyst rating,
beta range, FCF positive, ROE threshold, profit margin.

**Design:** No synthetic data. Missing cells display "—". Stale-data
banner shown if live prices fail. Offline fallback uses a static seed
list — clearly labeled so the user knows they're seeing stale data.

---

## Report Layer (PDF)

Four-module ReportLab-based PDF generator.

**`template.py`** — `SECTION_ORDER` list and formatting constants that
drive the renderer. The single file to edit when adding or reordering
sections.

**`content_builder.py`** — One `build_<section>(ctx)` function per
section. Each returns a list of ReportLab flowables. Rules every
builder follows:
- Source-driven: builders consume the snapshot context dict only
- Missing-data behavior: absent section → section header + labeled
  placeholder (section never silently disappears)
- Deterministic: same inputs → same output
- Bounded prose: long inputs (essay, filings list) are truncated with
  an ellipsis rather than overflowing

**`charts.py`** — Adapts chart artifacts (file path, PNG bytes,
Matplotlib Figure, or chart-ready data dict) into ReportLab `Image`
flowables sized to the report column. Returns a labeled placeholder
if a chart cannot be loaded — the report keeps rendering.

**`pdf_renderer.py`** — Owns the `SimpleDocTemplate` construction
(page size A4, margins), the page template (header rule, footer with
ticker + page number), deterministic metadata stamping, and the
failure-handling contract: if rendering raises, no partial PDF is
written to `output_path`.

**CLI** (`__main__.py`):
```bash
# Single-ticker memo from a saved snapshot
python -m report --ticker AAPL \
    --snapshot-id abc123 \
    --context ./snapshots/aapl_2026-05-15.json \
    --output ./reports/

# Portfolio-scope memo
python -m report --scope "core_long_book" \
    --snapshot-id day-2026-05-15 \
    --context ./snapshots/portfolio_2026-05-15.json \
    --output ./reports/
```

---

## UI Layer (Streamlit)

Entry point: `streamlit run app.py`

**Sidebar**
- Ticker picker backed by the full ~600-symbol universe
- Lookback period selector (1M, 3M, 6M, 1Y, 2Y, 5Y)
- Model backend selector (mock / dev / prod)
- Manual refresh button

**Portfolio overview tab**
- Position cards: current price, change %, market cap, cost basis,
  unrealized P&L
- Portfolio-level VaR and realized volatility summary

**Main analysis tab**
- Interactive Plotly price chart with toggle overlays: SMA-20/50/200,
  RSI-14, realized vol, Bollinger bands
- Risk panel: combined risk level badge + per-dimension breakdown
  (fundamental, macro, market, agent) with reasons
- Thesis panel: short-form verdict + essay (streaming or pre-loaded)
- Data-point analyzer: per-metric commentary cards
- Macro context: FRED indicators with visual stress indicators
- Raw context JSON debug view (collapsible)

**Screener tab** — see [Screener](#screener)

**Quant playground tab** (`playground.py`)
- HMM regime plot (2 or 3 states, probabilistic)
- Hurst exponent chart with rolling H and interpretation
- Ornstein-Uhlenbeck mean-reversion fit
- GBM Monte Carlo forward simulation
- Each sub-section fails softly: missing data or absent dependency
  shows a clear message rather than a stack trace

**Chat tab** (`chat.py`) — see [Chat Agent](#chat-agent-chatpy)

---

## Model Configuration

Three model tags in `config.AGENT_MODELS`:

| Tag | Model | Use |
|---|---|---|
| `mock` | Deterministic stub | Tests, CI, offline development |
| `dev` | `phi3:3.8b` via Ollama | Default; validates pipeline on modest hardware |
| `prod` | `qwen3:14b-instruct` | Swap in once dev validation is complete |

Change `DEFAULT_AGENT_MODEL` in `config.py` to switch globally. No
other code changes needed. Per-call override:
`AgentRequest(..., model_tag="mock")`.

Remote GPU workstation: set `OLLAMA_BASE_URL=http://192.168.x.x:11434`
in `.env` to point at a machine other than localhost.

---

## Project Structure

```
.
├── config.py                       # model tags, API keys, paths, thresholds
├── main.py                         # CLI entry point; orchestrates full pipeline
├── __main__.py                     # python -m report PDF CLI
│
├── app.py                          # Streamlit dashboard (main UI entry point)
├── screener.py                     # TradingView-style screener tab (~1500 lines)
├── playground.py                   # Quant playground tab
├── chat.py                         # Streaming chat agent tab
├── universe.py                     # ~600-ticker US universe + ticker validation
├── providers.py                    # Unified market data provider interface
├── earnings_calendar.py            # Earnings date tracking
│
├── base_agent.py                   # AgentRequest / AgentResponse / ask_agent
├── filing_analyzer.py              # 10-K / 10-Q metric extraction
├── data_point_analyzer.py          # Per-field signal interpretation (~48KB)
├── data_point_analyzer_section.py  # Section-level data point rendering
├── risk_scanner.py                 # Rule-based risk-flag engine
├── thesis_generator.py             # Heuristic BUY / HOLD / SELL verdict
├── thesis_essay.py                 # LLM multi-page investment memo
├── thesis_review.py                # Committee-style self-review pass
├── essay.py                        # Essay rendering utilities
│
├── sec_fetcher.py                  # SEC EDGAR (~56KB): filings, XBRL, Form 4, 13F
├── market_data.py                  # yfinance + technicals + fundamentals (~48KB)
├── macro_data.py                   # FRED macro series
├── sentiment_news.py               # News, Stocktwits, GDELT, WSB, VADER
├── geo_supply.py                   # Sanctions, EIA, freight, contracts
├── etf_providers.py                # iShares / SPDR / Vanguard / generic adapters
├── etf_holdings_loader.py          # ETF holdings ingestion orchestrator
├── factor_returns_loader.py        # Ken French FF3 + momentum daily factors
├── global_risk_pulse.py            # Market-wide composite risk pulse (~48KB)
├── portfolio_db.py                 # SQLite: positions, trades, FIFO P&L
├── portfolio_construction.py       # Portfolio optimization utilities
├── pipeline.py                     # Registry-first context builder (~52KB)
├── data_registry.py                # Normalized schema, upserts, snapshots (~44KB)
├── derived_signals.py              # Pure-Python signal computation
├── refresh_scheduler.py            # Cadence-driven orchestrator
├── notifiers.py                    # Slack webhook notifications
│
├── pdf_renderer.py                 # ReportLab page layout + failure handling
├── content_builder.py              # Section builders from snapshot context
├── charts.py                       # Chart artifact → ReportLab flowable
├── template.py                     # Section order + formatting constants
│
├── var_es.py                       # VaR & Expected Shortfall (3 methods)
├── volatility.py                   # Vol estimators + GARCH forecast
├── regime.py                       # Rule-based regime classifier
├── regime_hmm.py                   # Gaussian HMM regime classifier
├── kelly.py                        # Kelly criterion (4 variants)
├── position_sizing.py              # Kelly + vol-target combined sizing
├── black_scholes.py                # European option pricing + full Greeks
├── avellaneda_stoikov.py           # Optimal market-making quotes
├── poisson.py                      # Poisson / Hawkes jump intensity
├── gbm.py                          # Geometric Brownian Motion Monte Carlo
├── hurst.py                        # Hurst exponent / long-memory
├── ou_process.py                   # OU mean-reversion fit + half-life
├── monte_carlo.py                  # General Monte Carlo utilities
├── sequential_monte_carlo.py       # Particle filter / SMC
├── longstaff_schwartz.py           # LSM American option pricing
├── sabr.py                         # SABR stochastic vol model
├── hrp.py                          # Hierarchical risk parity
├── rmt.py                          # Random matrix theory
├── mst.py                          # Minimum spanning tree (correlation)
├── wavelet_regimes.py              # Wavelet-based regime detection
├── lyapunov.py                     # Lyapunov exponent
├── lempel_ziv.py                   # Lempel-Ziv complexity
├── fft_analysis.py                 # FFT spectral analysis
├── ergodicity.py                   # Ergodicity economics utilities
├── sandpile.py                     # BTW sandpile model
├── omori.py                        # Omori law (aftershock clustering)
├── girsanov.py                     # Girsanov measure change
├── gan_synthetic.py                # GAN synthetic data generation
├── wave_function_collapse.py       # WFC scenario generation
├── yield_curve_3d.py               # 3-D yield curve visualization
│
├── test_all.py                     # Comprehensive test suite (~200KB)
├── conftest.py                     # Shared pytest fixtures
│
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── setup.py
├── activate.bat
├── .env.example
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11+
- Git
- [Ollama](https://ollama.com) — only needed for `dev` or `prod` mode

### Clone and install

```bash
git clone https://github.com/ekul-the-hound/Ary-Fund-AIs.git
cd Ary-Fund-AIs

python -m venv .venv
.venv\Scripts\activate               # Windows
# source .venv/bin/activate          # Linux / macOS

pip install -r requirements.txt
```

Editable install with dev tools:

```bash
pip install -e ".[dev]"
```

Heavier quant extras (`statsmodels`, `arch` for GARCH, `hmmlearn` for
HMM regimes):

```bash
pip install -e ".[quant]"
```

PDF report layer:

```bash
pip install reportlab
```

### Pull a model

```bash
ollama serve                         # leave running in its own terminal
ollama pull phi3:3.8b                # current dev model
```

### API keys

Copy `.env.example` → `.env`:

```dotenv
FRED_API_KEY=your_fred_key_here
SEC_USER_AGENT=Your Name your@email.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/...   # optional
OLLAMA_BASE_URL=http://localhost:11434           # override for remote GPU
```

- **FRED** is free: [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- **SEC** requires a User-Agent header — no key, just name/email
- **Slack** is optional — skipped gracefully if absent

### Verify the stack

```bash
# Offline mock check — no Ollama required
python -c "from base_agent import ask_agent, AgentRequest; print(ask_agent(AgentRequest(prompt='ping', context={}, model_tag='mock')).generated_json)"

# Market data
python -c "import yfinance as yf; print(yf.Ticker('AAPL').history(period='1d')['Close'].iloc[-1])"

# FRED macro
python -c "from macro_data import MacroData; import os; m = MacroData(api_key=os.environ['FRED_API_KEY']); print(m.get_series_latest('VIXCLS'))"
```

---

## Running the Project

Full pipeline over `config.WATCHLIST`:

```bash
python main.py
```

Override the ticker list:

```bash
python main.py --tickers AAPL MSFT NVDA GOOGL
```

Streamlit dashboard:

```bash
streamlit run app.py
```

Scheduled refreshes:

```bash
python refresh_scheduler.py hourly
python refresh_scheduler.py daily
python refresh_scheduler.py weekly
python refresh_scheduler.py scan    # market open risk scan + Slack push
```

Load Ken French factor returns (run once, then weekly):

```bash
python factor_returns_loader.py
```

Generate a PDF memo from a saved snapshot:

```bash
python -m report --ticker AAPL \
    --context ./snapshots/aapl_2026-05-15.json \
    --output ./reports/
```

---

## Running Tests

```bash
pytest                  # via pytest.ini
python test_all.py      # consolidated runner
pytest -x               # stop at first failure
pytest -k "registry"    # run tests matching a keyword
```

All tests use `model_tag="mock"` — fully offline and deterministic.

---

## Data Sources Reference

| Source | Category | What it provides | Key required |
|---|---|---|---|
| SEC EDGAR | Filings | 10-K/Q/8-K, XBRL facts, Form 4, 13D/G/F | No (User-Agent required) |
| yfinance | Market | Prices, fundamentals, statements, technicals | No |
| FRED | Macro | Rates, VIX, spreads, recession prob, sentiment | Free |
| Stocktwits | Sentiment | Message volume + bull/bear sentiment | No |
| Google News RSS | News | Headline coverage per ticker | No |
| GDELT 2.0 | Geo/News | Financial-themed event counts | No |
| OFAC SDN | Sanctions | US Treasury sanctioned entities | No |
| EU Sanctions | Sanctions | EU consolidated sanctions list | No |
| UK HM Treasury | Sanctions | UK sanctions list | No |
| UN SC | Sanctions | UN Security Council sanctions | No |
| EIA Open Data | Energy | Oil, gas, electricity spot + inventory | No |
| Stooq | Freight | Baltic Dry Index proxy | No |
| USAspending.gov | Contracts | Government contract awards | No |
| Ken French Library | Factors | FF3 + momentum daily factors | No |
| Reddit WSB | Social | Mention counts (best-effort scrape) | No |

---

## Current Status

**Fully built:**
- Data layer: SEC EDGAR (XBRL, Form 4, 13F), market data, FRED macro,
  sentiment/news, sanctions + supply-chain, ETF holdings (iShares/SPDR/
  Vanguard/generic), Ken French factors, portfolio DB, DataRegistry
  (~100 canonical fields), derived signals (factor exposures, regime,
  sector heatmaps, risk scores), global risk pulse (6 subcomponents,
  [-1,+1] scale), refresh scheduler (hourly/daily/weekly/event/scan),
  Slack notifier.
- Agent layer: model-agnostic routing, filing analyzer, data-point
  analyzer, risk scanner, heuristic thesis generator, LLM multi-page
  essay (8 sections), committee-style self-review, data-gap protocol,
  streaming chat agent.
- Quant layer: VaR/ES (3 methods), vol estimators + GARCH, rule-based
  and HMM regime classifiers, Kelly (4 variants), volatility-targeted
  sizing, Black-Scholes + full Greeks, Avellaneda-Stoikov, Poisson/
  Hawkes, GBM, Hurst, OU process, plus the full research module suite.
- Screener: ~600-ticker universe, live prices, lazy fundamentals, 15
  filter pills, 24h SQLite cache, no synthetic data.
- Report layer: PDF generator (ReportLab) with 8 sections, chart
  adapter, page layout, and CLI.
- UI: Streamlit dashboard with all tabs (main analysis, screener, quant
  playground, streaming chat).
- Tests: comprehensive `test_all.py`, all defaulting to mock mode.

**In progress:**
- Promoting `prod` model tag to Qwen3 once dev validation is complete.

---

## Future Work

- **RAG module** — ChromaDB + `nomic-embed-text` via Ollama, hooked into
  `pipeline.build_agent_context()`. Retrieval over historical filings
  and fund notes; more practical than fine-tuning given VRAM constraints.
- **Fine-tuning** — LoRA on Qwen3-1.7B / 4B via Unsloth trained on SEC
  filings + fund notes. Deferred; base instruction models + strong
  prompts carry the MVP load.
- **Candidate screener scoring** — composite quality score across all
  ~600 tickers from registry fundamental fields, ranked and filterable.

---

## Success Metrics

| Metric | Target |
|---|---|
| Filing metric extraction accuracy | ≥ 85% |
| Portfolio risk flag recall vs analyst review | ≥ 80% |
| Full single-ticker analysis time | < 2 min |
| 10-holding portfolio scan (end-to-end) | Completes without crashes |