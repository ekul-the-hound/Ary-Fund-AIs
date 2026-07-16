# ARY QUANT — Perplexity Setup Guide

Standing up the fund's research stack on a shared Perplexity account. Replaces the
Claude setup guide for this stack.

**Companion files:** `PPLX_1_MCP_list.md` (connectors + URLs) · `PPLX_2_space_instructions.md`
(house rules) · `PPLX_3_prompt_library.md` (the 28 methods as prompts).

---

## The four pieces

1. **The account** — a shared **Perplexity Pro** (or Max) on the fund email. Custom
   remote MCP connectors are **not available on the free tier**.
2. **MCP connectors** — the same hosted stack as before; Perplexity added custom
   remote connector support in March 2026.
3. **Spaces** — 4 Spaces carrying the always-on house rules (the closest thing to
   Skills that Perplexity has).
4. **Prompt library + portfolio memory** — the deep methods, pasted on demand; the
   book stored in the Portfolio & Risk Space.

Still **host nothing**. Every connector is a URL someone else runs.

---

## Step-by-step

### 1. Create the shared Pro account  (~10 min)
Sign up on the **fund email**, choose **Pro** (or Max). Store the login in a shared
password manager — everyone uses this one account, so connectors and Spaces are
inherited automatically on any device.

### 2. Get the free API keys  (~15-20 min)
- **FMP** — financialmodelingprep.com → free account → API key
- **Alpha Vantage** — alphavantage.co/support/#api-key → instant key
- **HowRisky** — howrisky.ai → free account → API key
- No key needed: FXMacroData (USD tier), SEC EDGAR, FinanceQuery, Helium (first 50)
- Optional: Finnhub, Polygon, FlashAlpha

Keep keys in the shared password manager.

### 3. Add the connectors  (~20-30 min)
**Settings → Connectors → + Custom connector → Remote.** One at a time, testing each:

| Field | What to enter |
|---|---|
| Name | e.g. "FMP" |
| MCP Server URL | from `PPLX_1_MCP_list.md` (HTTPS required) |
| Description | short note on what it does |
| Authentication | OAuth / **API Key** / **None** (see the list) |
| Transport | **Streamable HTTP** (try SSE only if it fails) |

Check the acknowledgement box → **Add** → click the connector card to run the auth
flow. Recommended order: FMP → Alpha Vantage → HowRisky → FXMacroData → SEC EDGAR.

*OAuth note:* if a server supports discovery, endpoints are detected automatically. If
Perplexity asks for a Client ID/Secret, register the redirect URL
`https://www.perplexity.ai/rest/connections/oauth_callback`.

### 4. Test each connector  (~10-15 min)
In a chat: **`+` → Connectors and sources** → check the connector, then ask:
- FMP: *"Pull AAPL's latest income statement and key ratios."*
- Alpha Vantage: *"Latest NVDA quote and 14-day RSI."*
- HowRisky: *"Calculate the CVaR of a 60/40 portfolio."*
- FXMacroData: *"Show available USD indicators."*
- SEC EDGAR: *"Get MSFT's most recent 10-K."*

These double as **liveness checks** on the community URLs. If one fails, drop it —
Tier A covers the essentials.

### 5. Build the 4 Spaces  (~20-30 min)
Create each Space, paste its instruction block from `PPLX_2_space_instructions.md`,
set the source filter (**Finance**, plus Web where useful), and enable its connectors:

| Space | Instructions | Connectors |
|---|---|---|
| **ARY Equity Research** | Block 1 | FMP, Alpha Vantage |
| **ARY Portfolio & Risk** | Block 2 | HowRisky, FinanceKit |
| **ARY Macro & Market** | Block 3 | FXMacroData, Alpha Vantage |
| **ARY Filings & Events** | Block 4 | SEC EDGAR, FMP, earnings.video |

### 6. Load the prompt library  (~10 min)
Upload `PPLX_3_prompt_library.md` as a **file into each relevant Space** so it's
retrievable in-context, and keep a copy somewhere the team can copy-paste from
(shared doc). Both — the file makes it referenceable, the doc makes it pasteable.

### 7. Enter the portfolio  (~20-30 min, once)
In the **Portfolio & Risk** Space, enter the book using the format in prompt #26.
Upload it as a file *and* paste it in — Perplexity's memory is less predictable than
Claude's, so the file is the reliable copy.

### 8. Full dry-run  (~15-20 min)
One end-to-end workup: Equity Research Space → paste the memo prompt → a ticker →
verify it pulls live data, applies the house rules, and produces a memo with real
cited numbers.

### 9. Onboard the team  (~5 min/person)
Log into the fund email → Spaces and connectors are already there → in each chat,
`+` → Connectors and sources → check what's needed. No per-person setup.

---

## Time estimate

| Step | Time |
|---|---|
| 1. Pro account | 10 min |
| 2. API keys | 15-20 min |
| 3. Add connectors | 20-30 min |
| 4. Test connectors | 10-15 min |
| 5. Build 4 Spaces | 20-30 min |
| 6. Load prompt library | 10 min |
| 7. Enter portfolio | 20-30 min |
| 8. Dry-run | 15-20 min |
| 9. Onboard team | 5 min/person |

**Core setup (1-8): about 2 to 2.5 hours.** Realistically **plan half a day — 2.5-3
hours with buffer**. It's ~30 min longer than the Claude version because you're
building 4 Spaces by hand instead of loading a skills folder, and each connector form
has more fields (auth type, transport).

**Time-boxed path (~60-75 min):** Pro account → FMP + Alpha Vantage → the "ARY Equity
Research" Space only → the memo + verdict prompts. That's a working research setup;
add the rest later.

---

## What's different from Claude (know these)

- **No Skills.** Spaces hold condensed rules; deep method is pasted. This is the real
  cost of the move — the discipline isn't automatic beyond what fits in a Space.
- **Space instructions are length-limited** — that's why the blocks are condensed and
  the library is separate. Don't try to paste a whole skill into a Space.
- **Memory is weaker.** Keep the portfolio as an uploaded FILE in the Space, not just
  in memory.
- **Perplexity's native Finance tab is a bonus** — built-in market data with SEC-filing
  links, so figures are traceable in one click. Use it for fast lookup; use connectors
  for depth and compute. Don't add a connector to duplicate what it does natively.
- **Citations are the upside.** Perplexity cites by default, which suits the
  ground-truth discipline well — every figure traceable.

## Common gotchas

- **Free tier can't add custom connectors** — must be Pro/Max/Enterprise.
- **Transport mismatch** — if a connector won't connect on Streamable HTTP, try SSE.
- **OAuth wants a Client ID** — only if the server lacks dynamic registration; use the
  callback URL above.
- **Community URLs have no SLA** — SEC EDGAR, FinanceQuery, earnings.video, tvremix
  can move or throttle. Tier A (FMP, Alpha Vantage, HowRisky, FXMacroData) is stable.
- **Shared free-tier limits** — one account = one quota. Alpha Vantage ~25/day,
  HowRisky 100/mo, FinanceKit 100/mo, Helium 50 total, FlashAlpha 5/day.
- **Keys visible on the shared login** — fine for free student-fund keys; don't reuse
  them anywhere sensitive.
