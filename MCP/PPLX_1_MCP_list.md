# ARY QUANT — MCP Connector List (Perplexity)

The fund's data/compute stack, ported to Perplexity. **The good news: the stack
transfers essentially unchanged.** Perplexity added custom remote MCP connector
support on March 13, 2026 — same protocol, same URLs, same keys, same host-nothing
design.

*Verified July 2026 against Perplexity's connector docs and each vendor's MCP pages.
Endpoints were not live-tested — verify on first connect.*

---

## What changed vs. the Claude version

| | Claude | Perplexity |
|---|---|---|
| **Plan needed** | Team (paid) | **Pro, Max, or Enterprise** — custom remote connectors are *not* on the free tier |
| **Where to add** | Settings → Connectors → Add custom connector | **Settings → Connectors → + Custom connector → Remote** |
| **Setup fields** | Name, URL, (OAuth advanced) | Name, **MCP Server URL (HTTPS required)**, Description, **Authentication** (OAuth / API Key / None), **Transport** (Streamable HTTP or SSE), Icon |
| **OAuth callback** | n/a | Register `https://www.perplexity.ai/rest/connections/oauth_callback` if a server needs a Client ID/Secret |
| **Enabling per chat** | `+` → Connectors → toggle | `+` → Connectors and sources → check the connector |
| **Rule that built the list** | Host nothing (URL only) | **Unchanged** |

Two Perplexity-specific notes:
- **Transport field:** most of our connectors are Streamable HTTP. If one fails, try SSE.
- **Dynamic client registration:** if an OAuth server supports discovery
  (`/.well-known/oauth-authorization-server`), endpoints are detected automatically —
  no Client ID/Secret needed. Only supply those if Perplexity asks.

---

## Tier A — Official vendor-hosted (the backbone; add these)

| # | Connector | MCP Server URL | Auth | Transport | Covers |
|---|---|---|---|---|---|
| A1 | **Financial Modeling Prep (FMP)** — *anchor* | `https://financialmodelingprep.com/mcp?apikey=YOUR_KEY` | None (key in URL) | Streamable HTTP | Fundamentals, statements, ratios, analyst estimates, earnings, **SEC filings, insider trades, economic indicators** |
| A2 | **Alpha Vantage** | `https://mcp.alphavantage.co/mcp?apikey=YOUR_KEY` | None (key in URL) | Streamable HTTP | Prices (real-time + historical), technicals (RSI/MACD/Bollinger…), news & sentiment, forex/crypto/commodities, macro |
| A3 | **HowRisky** — *risk math* | `https://mcp.howrisky.ai` | **API Key** (`X-API-Key`) | Streamable HTTP | CVaR, VaR, ruin probability, fat-tail Monte Carlo, portfolio comparison, Kelly |
| A4 | **FXMacroData** — *macro* | `https://fxmacrodata.com/mcp` | OAuth (auto-discovery) — fallback `?api_key=YOUR_KEY` | Streamable HTTP | USD macro **free, no key**: policy rate, CPI, GDP, employment, release calendar, FX spot |
| A5 | **FlashAlpha** — *options (optional)* | `https://lab.flashalpha.com/mcp-oauth` (OAuth) or `https://lab.flashalpha.com/mcp` (key) | OAuth or API Key | Streamable HTTP | GEX, Greeks, IV, key levels. **5 req/day free** — options-only |

**Core:** A1 + A2 + A3 + A4. Three free keys (FMP, Alpha Vantage, HowRisky); FXMacroData's USD tier needs none.

## Tier B — Community-hosted (free, no SLA; optional)

| # | Connector | MCP Server URL | Auth | Covers |
|---|---|---|---|---|
| B1 | **SEC EDGAR** | `https://secedgar.caseyjhand.com/mcp` | None | 10-K/10-Q/8-K text, XBRL, Form 4, 13D/G/F, full-text search since 1993 |
| B2 | **FinanceQuery** | `https://finance-query.com/mcp` | None | 36 tools: quotes, technicals, news, filings — no-key generalist |
| B3 | **earnings.video** | `https://mcp.earnings.video/mcp` | None shown | **Speaker-identified earnings transcripts** + per-call insights, 7,000+ companies. *Verify free tier* |
| B4 | **Helium** — *news bias scoring* | `https://heliumtrades.com/mcp` | None (first 50 queries) | 3.2M+ articles, source-bias scoring across 15–37 dimensions, balanced synthesis. *50 free total, then $0.02/query* |
| B5 | **tvremix (TradingView)** | `https://tvremix.xyz/mcp` | OAuth | TradingView OHLCV, technicals, screeners. *Verify free tier* |

## Tier C — Via third-party gateway (adds a dependency; optional)

| # | Connector | MCP Server URL | Auth | Covers |
|---|---|---|---|---|
| C1 | **FinanceKit** (via MCPize) | `https://financekit-mcp.mcpize.run/mcp` | None | **Sharpe, Sortino, Beta, correlation matrix**, max drawdown, technical verdicts. 100 calls/mo |
| C2 | **Finnhub** | Pipedream: `https://mcp.pipedream.net/v2` | API Key | Quotes, profiles, recommendation trends, news (~30 req/min) |
| C3 | **Polygon.io** | Pipeworx: `https://gateway.pipeworx.io/polygon-io/mcp` | API Key | Tick-level trades/quotes, aggregates, options (~5 calls/min) |

---

## Perplexity's own Finance data — the new wrinkle

Perplexity Finance has **built-in** market data, SEC-filing links, and earnings info
natively — and its financial data links directly to SEC filings so figures are
traceable. This overlaps some of the stack.

**How to think about it:** Perplexity's native Finance tab is your **fast lookup and
citation layer**; the MCP connectors are your **depth and compute layer**. Concretely:

- Quick quote, chart, or "what did they report?" → **native Finance tab**, no connector needed.
- Full statements, ratios, insider data, screening → **FMP**.
- Risk math (CVaR, Monte Carlo), Sharpe/correlation → **HowRisky / FinanceKit** (nothing native does this).
- Transcripts, macro series, bias-scored news → **earnings.video / FXMacroData / Helium**.

Don't add a connector to duplicate what the Finance tab already does well — add them
for what it *can't* do (compute, deep fundamentals, specialized data).

---

## Setup order (Perplexity)

1. **Perplexity Pro (or Max)** on the fund email — required for custom connectors.
2. **Free keys:** FMP, Alpha Vantage, HowRisky. (FXMacroData USD, EDGAR, FinanceQuery, Helium need none.)
3. **Settings → Connectors → + Custom connector → Remote**, one at a time:
   - Name, MCP Server URL, Description, Authentication, Transport (Streamable HTTP), acknowledge risk → **Add**
   - Click the connector card to complete auth.
4. **Test each** — in a chat, `+` → Connectors and sources → check it, then:
   - FMP: *"Pull AAPL's latest income statement and key ratios."*
   - Alpha Vantage: *"Latest NVDA quote and 14-day RSI."*
   - HowRisky: *"Calculate the CVaR of a 60/40 portfolio."*
   - FXMacroData: *"Show available USD indicators."*
   - EDGAR: *"Get MSFT's most recent 10-K."*
5. If a Tier B/C URL fails, drop it — Tier A covers the essentials.

## Realities (shared account)

- **Free-tier limits are shared:** Alpha Vantage ~25/day, HowRisky 100/mo, FinanceKit 100/mo, FlashAlpha 5/day, Helium 50 total. Considered research, not bulk pulls.
- **Cross-check** decision-critical numbers across FMP and Alpha Vantage; EDGAR is primary source for filings — and Perplexity's native SEC links make verification one click.
- **Keys live in the connector config** on the shared login — fine for free student-fund keys; don't reuse elsewhere.
- **Trust boundary unchanged:** every connector reads public data only. Nothing of the fund's is exposed.
