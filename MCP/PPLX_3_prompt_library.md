# ARY QUANT — Perplexity Prompt Library

The 28 skills, reshaped as **paste-in prompts**. Perplexity has no skill-loading
mechanism, so deep methodology travels as text you paste at the start of a question.

**How to use:** work inside the matching Space (which supplies the always-on house
rules — see `PPLX_2_space_instructions.md`), then paste the relevant prompt below and
add your ticker/question. Space = discipline; prompt = method.

**The rule inside every prompt:** never fabricate a number. If a figure isn't
retrieved, say so in one sentence and move on.

---

# WRITING, REVIEWING & DISCIPLINE

## 1. Investment Memo (house style)
> Write an investment memo on [TICKER] in ARY QUANT house style. Use exactly these 8 sections:
> 1. **Executive Summary** — one thesis sentence in this exact form: "[Company] is [quality], but at [valuation] [return expectation] unless [key condition]." Then the single biggest risk and single strongest catalyst. Nothing else.
> 2. **Business & Financial Performance** — a connected narrative, metric by metric, each with its implied meaning (margin structure, cash conversion, capital efficiency, ROIC). An argument, not a list.
> 3. **Valuation** — P/E, forward P/E, EV/EBITDA vs. the S&P 500 (~20-22x trailing), the sector, and the company's own history. State whether the multiple is deserved and why.
> 4. **Filings & Management Signal** — management tone (cited from filings/transcripts, never asserted) and Form 4 insider activity.
> 5. **Macro & Industry Context** — the specific transmission mechanism from macro to THIS company's revenue, margins, or multiple.
> 6. **Risks** — top 3 ranked. Each: trigger (a number), transmission mechanism, price impact.
> 7. **Catalysts** — top 2-3. Each: threshold, timeframe, price-move rationale.
> 8. **Verdict** — BUY/HOLD/AVOID, 1-year return expectation, 0-100 conviction, and the single data point that would change the verdict.
>
> Rules: every major claim carries a number. Decisive, not hedged. Company-specific throughout — if a paragraph would read the same for any peer, cut it. Separate business quality from stock attractiveness. ~1,200-2,000 words. Never fabricate a figure; acknowledge gaps in one sentence.

## 2. Committee Memo Review
> Review this memo as a senior PM preparing it for investment committee. For EACH of the 8 sections: score 1-10, state the single biggest weakness in 1-2 sentences, and give the single most important fix — which must include a specific number, threshold, or comparison, plus one sentence on WHY it matters.
>
> CARDINAL RULE: every number you cite must come from the retrieved data. Never invent a metric. Before calling a metric "missing," check whether it's actually available — if it is, cite the real value and say it was underused; only if genuinely absent say "not in the data" and never state a value for it.
>
> Format each section as:
> ## [N]. [Section] [SCORE: X/10]
> Main weakness: ...
> Fix 1: ... This matters because ...
>
> Flag if present: "bullish" + "expensive" used together without reconciliation; valuation inferred from margins instead of actual multiples; management tone cited without filing evidence; "positive surprise" without a numeric definition; filler words standing in for a number.
> Prioritize judgment over coverage — pick the fixes that matter most.

## 3. Cognitive Bias Audit
> Audit this thesis/decision for cognitive bias. For each bias found, output:
> [Bias] — DETECTED / not evident
> Where: [the specific claim, number, or decision affected]
> Evidence: [why you think it's operating]
> Correction: [the concrete debiasing step and how it changes the analysis]
>
> Check: **Confirmation** (is all evidence one-directional? was the bear case steelmanned?). **Anchoring** (fixated on the 52wk high, IPO price, entry price, or first analyst target? re-derive value with the anchor hidden). **Framing** (restate key metrics in both frames — "90% retention" vs "10% churn" — does the reaction stay stable?). **Overconfidence** (point estimates instead of ranges? no serious downside case? widen the range). **Recency** (extrapolating the last quarter forever? check the full cycle and base rate). **Herding** (is this just the consensus thesis? state the variant perception or admit there's no edge). **Loss aversion** (decisions driven by cost basis/P&L rather than forward thesis? ask: would I buy this today at this price?). **Hindsight** (post-hoc certainty without a written prediction?).
>
> Report only the 2-4 biases MATERIAL to this decision, ranked by impact. A flag without a correction is worthless.

---

# JUDGING THE NUMBERS

## 4. Metric Interpretation
> Interpret [METRIC/the metrics panel] for [TICKER]. Every interpretation is ONE sentence: value → anchor → implication. An unanchored number is not interpretation.
> Anchors: S&P 500 median (trailing P/E ~20-22x), the SECTOR, the company's own history, or a house threshold (Debt/EBITDA 3.0x elevated / 2.0x moderate; interest coverage 2.0x; FCF yield 2%; vol 45%/30%; drawdown 25%/15%; RSI 75/25).
> Decompose ROE: high ROE with modest leverage = genuine efficiency; high ROE from leverage = balance-sheet engineering. Say which.
> Group a full panel by theme (valuation / profitability / leverage-liquidity / cash / growth), leading with the metric that most drives the assessment. Directional and honest — good, bad, or mixed. Missing metric = "—", never a fabricated comparison.

## 5. Valuation Analysis
> Judge [TICKER]'s valuation. Compare its multiples against all three: (1) the S&P 500 (~20-22x trailing P/E), (2) its SECTOR, (3) its own historical range.
> Then the deserved-multiple test: base ~22x at 0% growth, rising toward a ~40x cap at ~30%+ growth. Is the ACTUAL multiple consistent with the ACTUAL growth and returns on capital? (34x on 25% growth is defensible; 34x on 5% growth is not.)
> Use only the multiples that matter for this name — not all of them. (Bank: P/B, P/E. High-growth software: EV/S, fwd P/E, FCF yield. Mature industrial: EV/EBITDA, FCF yield.)
> End with THE FLIP POINT: the multiple, growth rate, or margin at which the verdict changes.
> Use real market multiples — never infer valuation from margins/FCF when actual multiples exist. Separate business quality from stock attractiveness.

## 6. Distress & Quality Scoring
> Score [TICKER] on all three models and read them together.
> **Altman Z** — pick the variant by sector: use Z'' for financials/real-estate/utilities (or if revenue is missing): Z'' = 6.56(WC/TA) + 3.26(RE/TA) + 6.72(EBIT/TA) + 1.05(BookEq/TL); zones <1.10 distress / 1.10-2.60 grey / >2.60 safe. Otherwise original Z = 1.2(WC/TA) + 1.4(RE/TA) + 3.3(EBIT/TA) + 0.6(MktCap/TL) + 1.0(Sales/TA); zones <1.81 distress / 1.81-2.99 grey / >2.99 safe.
> **Piotroski F (9 tests, 1pt each)** — Profitability: NI>0; OCF>0; ROA improving; OCF>NI. Leverage/liquidity: LTD/TA decreased; current ratio increased; no share issuance. Efficiency: gross margin increased; asset turnover increased. 8-9 strong, 0-2 weak. If <5 tests evaluable, it's None. It's a TREND measure.
> **Beneish M** = -4.84 + 0.92(DSRI) + 0.528(GMI) + 0.404(AQI) + 0.892(SGI) + 0.115(DEPI) - 0.172(SGAI) + 4.679(TATA) - 0.327(LVGI). Threshold -1.78: above = likely manipulator. TATA (accruals) carries the largest weight. A red flag for digging, NOT a fraud verdict.
> Read together: Altman = can it go bankrupt (solvency now); Piotroski = getting better or worse (trend); Beneish = are earnings believable (honesty). DIVERGENCE is informative. Report the sub-tests that drove each score. Sparse inputs → None, never a guess.

## 7. Sector-Relative Peer Analysis
> Compare [TICKER] to its ACTUAL sector peers, not the broad market — 3x debt/EBITDA is alarming for software and unremarkable for a utility.
> Establish the sector correctly first. Then z-score the thesis-relevant metrics against real sector peers: debt/EBITDA, net debt/EBITDA, interest coverage, FCF yield, ROIC, operating margin, gross margin, cash conversion.
> Translate to plain language: "operating margin ~1.5 SD above sector — a genuine margin leader among peers," not "z = 1.5."
> PAIR with absolute thresholds — sector-relative gives rank within cohort; absolutes still flag outright danger. A name can be average for a levered sector and still absolutely risky. Report both.
> If there aren't enough real peers for a distribution, say so and fall back to absolute thresholds. Never fabricate a sector mean.

## 8. Moat & Competitive Analysis
> Assess [TICKER]'s moat. NAME THE MECHANISM from these five, or conclude honestly that there isn't one: (1) network effects, (2) switching costs, (3) intangibles/brand/patents, (4) cost advantage, (5) efficient scale. "Good management," "great products," and "strong culture" are NOT moats — they're outcomes that erode without a mechanism.
> Then TEST it with numbers: ROIC above WACC persistently (the definitional test); ROIC/margins vs. sector peers (a moat should rank top of cohort); margin stability THROUGH cycles (pricing power holds margins in downturns); actual pricing power (raised prices ahead of inflation without volume loss?); share trend (gaining share while spending more to defend it = a shrinking moat).
> Map only the 2-3 BINDING competitive forces (rivalry / entrants / substitutes / buyer power — a 30% customer negates a lot of moat / supplier power).
> Judge DIRECTION: widening (ROIC up, share gains, deepening switching costs) or eroding (new entrant traction, price competition, churn creeping, patents expiring, platform shift). Name the specific threat vector.
> Connect it: moat → duration of high returns → deserved multiple. But moat quality ≠ stock attractiveness — a wide moat at 40x can be worse than a narrow moat at 8x.

## 9. DCF & Reverse DCF
> Value [TICKER] from cash flows. Structure: 5-10 years explicit, then terminal.
> (1) Revenue growth with a stated driver, FADING toward a mature rate — almost nothing compounds at 25% for a decade. (2) Operating margin path justified by moat/peer evidence — margin expansion is where DCFs smuggle in the bull case. (3) FCF after tax, capex, working capital — treat stock-based comp as a real cost, don't add it back while holding share count flat. (4) A simple defensible discount rate (~8-10% stable large cap, higher for risk) — state it. (5) Terminal: perpetuity growth 2-3%, NEVER above long-run GDP. FLAG the terminal share — if >80% of value, the "forecast" is one assumption.
> Then run the REVERSE DCF — the most useful mode: solve backward from today's price. What growth/margin path justifies the current price? Then judge it against history and peers: "At $180 the market prices ~14% growth for a decade at 30% margins; they've never exceeded 11% over 5 years and peer-best margin is 26%."
> Present a RANGE (grid over the 2-3 assumptions that matter), bull/base/bear, and name the breakeven assumption that kills the upside if it moves.
> Triangulate against multiples — a gap is a question, not an answer. DCFs are least reliable for pre-profit names, mid-cycle cyclicals, and financials.

---

# RISK & MACRO

## 10. Risk Flags (four axes)
> Score [TICKER]'s risk on four independent axes, each HIGH/MEDIUM/LOW with its numeric driver, then combined.
> **Fundamental**: Debt/EBITDA >2.0x MEDIUM / >3.0x HIGH; interest coverage <2.0x HIGH; FCF yield <2% HIGH. Fold in distress models (grey/distress Altman zone, weak Piotroski, high Beneish M all escalate).
> **Macro**: recession prob >35% MEDIUM / >60% HIGH; VIX >20 MEDIUM / >28 HIGH; inverted curve contributes.
> **Market**: realized vol >30% MEDIUM / >45% HIGH; drawdown >15% MEDIUM / >25% HIGH; RSI >=75 or <=25 MEDIUM.
> **Agent**: qualitative risks from filings the rules miss (litigation, customer concentration, guidance cut).
> COMBINED = the WORST material axis, not an average. Don't let a benign blend hide a HIGH axis.
> Match the response to the axis: fundamental → sizing; macro → timing/hedging; market → entry; agent → verify in filings.

## 11. Macro Regime Read
> Read the macro backdrop for [TICKER]. Bands: recession prob <=10% supportive / 10-30% mildly supportive / 30-50% mildly hostile / >50% hostile. VIX <=15 supportive / 15-20 neutral / 20-28 mildly hostile / >28 hostile. 2s10s inverted = hostile; clearly positive = mildly supportive. Elevated financial stress = hostile.
> State the net risk-on/risk-off tilt. Then — the part that matters — pick only the 3-5 variables MATERIAL to this name and trace each through a specific channel: REVENUE (demand sensitivity), MARGINS (input costs, interest expense, FX), or THE MULTIPLE (discount rate, risk appetite). "Rates up" is not analysis; "rates up compresses this unprofitable long-duration name's multiple and raises its interest expense" is.
> No indicator dumps. If a series isn't retrieved, say so rather than assume its level.

## 12. Market Pulse
> Read the market-wide risk pulse. Scale: +1 = stressed/risk-off, -1 = calm/risk-on (positive = stress, matching how financial-stress indices are signed).
> Six subcomponents: (1) Volatility — weighted realized vol z-scored vs. baseline. (2) Breadth — participation; broad = healthy/negative, narrow = positive/stress. Narrowing breadth deteriorates before the index does. (3) Correlation — median pairwise; when everything moves together, diversification fails; rising correlation is a late-cycle tell. (4) Concentration — a tape carried by a few mega-caps is fragile. (5) Dispersion. (6) Macro regime (VIX/curve/recession/stress).
> Give the headline, then ATTRIBUTE it to specific subcomponents. Watch the CHANGING component — a pulse rising on correlation and concentration (fragility building) is more concerning than the same level from an already-known high VIX.
> Use as backdrop that raises the bar for new longs — never as a standalone buy/sell.

## 13. Quant Signal Interpretation
> Interpret these quant signals for a decision — each is a LEAN, not a certainty, and each needs its caveat.
> **Hurst**: ~0.5 random (no edge); >0.5 trending/persistent (favors momentum, mean-reversion entries are dangerous); <0.5 mean-reverting (favors fading extremes, trend-chasing gets whipsawed). Regime-dependent and unstable.
> **OU process**: half-life is the key input — a 5-day half-life is tradeable; a 300-day half-life means "cheap" can stay cheap for a year. Distance from long-run mean (in SD) sizes the opportunity. ALWAYS pair with a fundamental check — statistical cheapness + a broken business is a value trap, and the mean itself will re-rate lower.
> **Regime/HMM**: use for CONDITIONING, not prediction. High-vol regime → deeper drawdowns, correlations rise, mean-reversion fails more; raise the bar and cut size. Models lag transitions — never assume the regime holds.
> **Realized vs. implied vol**: implied >> realized = options expensive (market sees a catalyst); implied << realized = optionality cheap. The spread is information about the event calendar.
> **VaR/CVaR**: VaR says nothing about how bad the tail is — prefer CVaR (average loss in the tail). Assume FAT tails; normal models understate real market moves badly.
> Every signal is estimated from past data over a finite window and breaks exactly when it matters (regime change). Where a signal and the fundamentals disagree, that TENSION is information — don't resolve it by trusting the model.

---

# FILINGS & SIGNALS

## 14. Filing & 8-K Signal
> Read [TICKER]'s filings for SIGNAL, not completeness.
> Material 8-K items: 5.02 (officer/director change — CEO/CFO turnover is first-order), 5.03 (bylaws), 8.01 (other material — buybacks, ratings), 1.01 (material agreement), 2.02 (results), 3.02 (unregistered sale — dilution). Also detect buybacks (capital return, often bullish) vs. secondaries/ATM (dilution, usually a headwind) — always state the DIRECTION.
> In the 10-K/10-Q: risk-factor density (Item 1A) — and a NEW risk factor year-over-year is itself a signal; management tone in MD&A (cite from the filing, never assert); notable disclosures (litigation, customer concentration, going concern, impairments, segment/accounting changes); guidance and results.
> Watch for distressed language: going concern, covenant waivers, liquidity discussions, material weakness, restatements — these escalate risk sharply and belong in the memo with the specific disclosure quoted.
> Weight by materiality. A missing filing is absence, not a signal.

## 15. Insider & Ownership
> Read [TICKER]'s insider and institutional activity.
> Form 4 codes: **P** = open-market purchase (STRONGEST — insider spending own money); **A** = grant (compensation-driven, weaker); **S** = open-market sale; **F** = tax withholding on vesting (MECHANICAL — weight lightly); **D** = disposition.
> Aggregate to NET USD over rolling 30-day windows, not single transactions. A CLUSTER of P buys by multiple insiders is far stronger than one buy.
> DISCOUNT 10b5-1 sales — pre-scheduled, not a reaction to information. Don't read them as bearish.
> Ownership: **13D** = activist >5% (intent to influence — potential catalyst, look for what change is being pushed) vs. **13G** = passive >5% (index/mechanical, low signal). Same threshold, very different meaning. **13F** = quarterly institutional holdings — track CHANGES, but it's a quarter stale, so it's confirmation, not a leading signal.
> Buying is stronger evidence than selling (insiders sell for diversification, taxes, liquidity). Missing data = absence.

## 16. Sentiment & News
> Weigh sentiment for [TICKER] as the SOFT signal it is.
> Read the ENSEMBLE (retail mention volume, social tone, news tone, coverage volume) — not one flaky feed.
> Weight it LOW: a tie-breaker and context flag, never a driver. Don't let bullish buzz upgrade a weak fundamental case.
> EXTREMES ARE CONTRARIAN: euphoric retail sentiment and mention spikes often mark crowding and late-stage moves; capitulation can mark washouts. Lean AGAINST extremes.
> Trust VOLUME over DIRECTION — rising mentions reliably says "attention here"; the crowd's implied view is far less reliable.
> Use news tone to CONFIRM a fundamental change, not to originate a thesis.
> If bias-scored news is available, ask whether a bearish story is real signal or an outlet's house slant. Keep sentiment labeled as soft — never cite it like a fundamental.

## 17. Earnings & Guidance
> Analyze [TICKER]'s earnings.
> (1) **The print vs. the bar** — the headline is meaningless alone. Surprise MAGNITUDE (1% EPS beat is noise; 10% is real) and COMPOSITION: a revenue-driven beat (demand) is higher quality than a beat from a low tax rate, buyback-shrunk share count, or one-offs. Check earnings quality: did margins expand, or did EPS beat on financial engineering? Is cash flow tracking net income? Note the surprise HISTORY — a company that always beats by a penny is managing the bar; a first miss after a beat streak is a big signal.
> (2) **The guide usually moves the stock more than the print.** Direction vs. prior; **vs. CONSENSUS — a "raise" still below the Street is a DE FACTO CUT** (the most common place readers get the reaction backwards); quality (widening range = rising uncertainty; a kitchen-sink reset by a new CEO can be bullish; pulling guidance is a red flag); composition (pricing vs. volume vs. easy comps).
> (3) **The transcript** — prepared remarks are scripted; the signal is in the Q&A. Tone shift vs. prior calls; EVASIVENESS (analysts asking twice, "we don't guide on that," filibustering — repeated dodging marks the risk); what they volunteer vs. what's extracted; language on demand/pricing/backlog/churn. Cite tone from the transcript, never infer it from numbers.
> (4) **The forward view** — did the trajectory change? That's the output, not the beat/miss. Was the price reaction right? Re-anchor valuation; reset the catalyst path.

## 18. Catalyst Map
> Build the catalyst path for [TICKER]. A catalyst is only real if it has a DATE WINDOW and a MEASURABLE TRIGGER — "sentiment could improve" is not a catalyst.
> For each: (1) Event, (2) Date/window (as precise as sourceable — never invent a date; flag unconfirmed ones), (3) TRIGGER — the measurable thing defining success vs. failure ("net adds >2.0M", "gross margin guide >=34%"), (4) Direction & magnitude if hit vs. missed, (5) Probability/conviction.
> Types: scheduled (earnings, investor days, votes, index rebalances, lockups), semi-scheduled (launches, regulatory decisions, contract awards), conditional/thesis-specific ("if segment turns FCF-positive, re-rate"), recurring monitorables.
> Sequence into a timeline. Read the SHAPE: near-term (1-3mo, drives timing/sizing) vs. long-term; binary (argues for smaller size) vs. continuous; density (clustered = event-rich window, higher implied vol).
> Output the single NEXT DECISION POINT and what result would confirm or break the thesis. Check whether the options market has already priced a known catalyst (rich IV).

## 19. Special Situations
> Analyze this event-driven situation for [TICKER]. The return driver is the EVENT MECHANICS, not business growth.
> **Spinoff**: the structural edge is that new shares land in portfolios that never chose them (index funds, funds for whom it's too small) → indiscriminate early selling. Work the Form 10: why is the parent shedding it? How much DEBT was allocated to spinco? Stranded costs vs. claimed standalone economics? Management incentives (equity struck at post-spin prices? insiders buying?) Which side does the CEO run? Check BOTH sides — the parent sometimes re-rates too.
> **Merger arb**: the spread is a PROBABILITY, not free money. Compute implied odds from target price / offer / downside (pre-announcement price). Inventory break risk: antitrust (the big one), financing, votes, MAC clauses — read the actual merger agreement and termination fee. Cash (clean spread) vs. stock (measure against the acquirer's shares). The shape: small capped upside, fat-tailed downside. Size very small or study rather than hold.
> **Forced selling**: index deletions, fund liquidations, margin cascades, downgrade thresholds. Identify WHO must sell, HOW MUCH, BY WHEN; confirm the business is unimpaired; size for overshoot.
> **Post-bankruptcy**: read the plan of reorganization. Fresh-start accounting makes screens misleading; new share count and debt; who owns it now; is the BUSINESS problem fixed, or just the balance sheet?
> **Tenders/Dutch auctions/rights**: read the offer doc for proration and odd-lot provisions before assuming the arithmetic. For rights: subscription discount, insider backstop (strong signal), what the raise funds.
> Universal: read the ACTUAL documents (that's the edge — most participants don't); map the timeline as catalysts; UNDERWRITE THE DOWNSIDE FIRST (what do you hold if the event fails?); check the business after the mechanics; size for discontinuous outcomes.

---

# REACHING A CALL

## 20. Buy/Hold/Sell Verdict
> Reach a verdict on [TICKER].
> (1) Score three biases in ~[-1,+1]: **fundamental** (profitability, growth, leverage, cash, valuation — dominates on a 1-year horizon); **macro** (recession prob <=10% → +0.5, <=30% → +0.2, <=50% → -0.3, else -1.0; VIX <=15 → +0.5, <=20 → +0.1, <=28 → -0.3, else -0.9; inverted curve → -0.4, clearly positive → +0.2; average them); **filing tone** (soft, resolves ties).
> (2) Weight: raw_bias = 0.55(fundamental) + 0.30(macro) + 0.15(filing tone), clipped to [-1,+1].
> (3) Dampen by combined risk: HIGH x0.5, MEDIUM x0.8, LOW x1.0. High risk compresses conviction toward neutral.
> (4) Map: >=+0.50 strong up/bullish | +0.15 to +0.50 moderate up/bullish | -0.15 to +0.15 flat/neutral | -0.50 to -0.15 moderate down/bearish | <=-0.50 strong down/bearish.
> (5) State: rating (**BUY >15% upside | HOLD 0-15% | SELL >15% downside**), the 2-3 numbers justifying it, a **0-100 conviction score with its main driver** (80-100 high conviction: signals align, risk LOW-MED, clear variant view; 60-79 constructive; 40-59 balanced/mixed — most HOLDs; 20-39 negative lean; 0-19 strong SELL. Move DOWN for conflicting signals, HIGH risk, crowded consensus, thin data, open bias flags; UP for aligned signals, a genuine variant view, insider/catalyst confirmation, margin of safety), the CHANGE-OF-VIEW TRIGGERS (specific numbers that would flip it), and one sentence separating business quality from stock attractiveness.
> Confidence rises when the three biases AGREE and falls when they conflict — say which.

## 21. Stock Screening
> Screen for candidates. Screening is TRIAGE — it produces a shortlist, not a verdict.
> Combine filters to express a THESIS, not a single-metric sort: sector; market-cap range; valuation (P/E, fwd P/E, PEG, P/S, P/B, EV/EBITDA); growth (revenue, EPS); profitability (margins, ROE floor); leverage (debt/equity, current ratio); cash (FCF positive, FCF yield); yield; beta; analyst rating.
> Examples: "quality at a reasonable price" = ROE floor + FCF positive + P/E and PEG ranges + debt/equity cap. "Beaten-down cyclical" = sector + drawdown/valuation floor + still solvent.
> RANK toward composite quality (profitable, growing, cash-generative, not over-levered, at a multiple its growth supports) — not by any single metric.
> Real data only: missing = "—", never a fabricated value. Don't rank on invented numbers.
> Route survivors into the full workup: fundamentals/valuation/peers → distress/risk → macro → filings/insider/sentiment → verdict. Plenty of screened names SHOULD fail the deeper work.

---

# POSITION & PORTFOLIO

## 22. Position Sizing
> Size a position in [TICKER]. Direction is half a decision; SIZE is where the risk lives.
> (1) Start from the verdict and conviction.
> (2) Compute BOTH: **fractional Kelly** — f* = (b·p - q)/b where p = win probability, b = win/loss ratio. NEVER full Kelly (wildly volatile, unforgiving of mis-estimated inputs) — use 1/2 or 1/4 as a CEILING and sanity check. Your p and b are estimates from your own thesis; haircut them. **Volatility target** — size so each position contributes similar RISK, not similar dollars (a 60%-vol name at the same weight as a 15%-vol name carries 4x the risk).
> (3) Take the MORE CONSERVATIVE of the two.
> (4) Construction checks: single-name cap (~5-10% diversified); CORRELATION to existing holdings (two 0.9-correlated 5% positions behave like one 10% position); aggregate sector/factor exposure (a fifth semi name may breach the sector limit even if each is small); marginal contribution to portfolio risk (a hedge justifies more, piling onto the biggest bet justifies less); liquidity (size down illiquid names regardless of conviction).
> (5) Stage it — starter now, adds on confirmation — rather than full size into uncertainty.
> State the size WITH its logic and add/trim triggers: "3% starter: 1/4-Kelly ceiling ~5% but trimmed for high correlation to [X] and a full sector sleeve; room to 5% on a confirmed guide."

## 23. Portfolio Review
> Review the whole book (read the latest dated portfolio block; if it's stale, say so first).
> (1) **Exposure map** — aggregate by sector, factor (growth/value, size, momentum, quality, rate-sensitivity, cyclical/defensive), geography. Find the exposures NOBODY DECIDED ON PURPOSE — unintended concentration is the most common problem. A book of "different" names can secretly be one factor bet.
> (2) **Concentration & correlation** — single-name weights vs. cap; correlation CLUSTERS; the EFFECTIVE NUMBER OF INDEPENDENT BETS (10 names might be 3 real bets).
> (3) **Contribution** — which positions drive portfolio CVaR (prefer CVaR over VaR — it captures the tail)? Often a small weight in a volatile correlated name contributes outsized risk. Plus return attribution: is the risk being rewarded?
> (4) **Cash & gross/net** — dry powder vs. drag.
> (5) **Drift** — winners grow into oversized weights; a 3% starter at 9% is a different decision than the one made.
> Actions, prioritized by IMPACT: trim (drifted past cap, outsized risk contribution, weakened thesis — trimming a winner back to target is discipline); add (below target, intact thesis, diversifying); hedge/reduce an unintended aggregate exposure; raise cash. **"Do nothing" is legitimate** — rebalance for a reason, never for activity. Weigh turnover cost.

## 24. Thesis Pre-Mortem
> Attack this thesis before the market does.
> (1) **PRE-MORTEM**: "It's 12 months from now and this position is down 40%. What happened?" Assuming failure has ALREADY occurred frees you to generate concrete causes. Force a full list (demand, margins, balance sheet, competition, management, macro, valuation, the thing nobody's modeling), then rank by likelihood × severity. Be specific and causal: "the top customer (28% of sales) in-sourced," not "customer concentration risk."
> (2) **STEELMAN THE BEAR CASE** — make it good enough that it scares you. A strawman manufactures false confidence. Cover: the valuation bear (what must go right to justify the multiple? what's it worth if not?); the fundamental bear (decelerating growth, margin pressure, cash-flow quality); the competitive bear (share loss, disruption, secular headwind); the EXPECTATIONS bear (even if the business is fine — is everyone already positioned? is the good news priced?); the governance bear (incentives, aggressive accounting, insider selling).
> (3) **LOAD-BEARING ASSUMPTIONS** — what MUST be true? Rank by fragility × damage. Is the thesis single-threaded (works only if one thing happens) or over-determined?
> (4) **KILL-CRITERIA** — pre-committed, MEASURABLE conditions decided NOW while calm: "two consecutive quarters of negative organic revenue growth," not "if fundamentals deteriorate." Each tied to a load-bearing assumption AND to an action (trim/exit/re-underwrite).
> Separate "I'm wrong" from "I'm early": a falling price isn't proof you're wrong, but "I'm early" is the most expensive sentence in investing. The thesis is intact only if the FUNDAMENTAL kill-criteria are un-fired. If one fired and you're explaining why it doesn't count, that's the trap closing.

## 25. Industry Primer
> Build a working primer on [INDUSTRY] before we pick names in it. 1-2 pages, decision-useful, not an encyclopedia.
> (1) What it sells and how money is made — one paragraph a freshman could follow. (2) **The value chain — and WHERE THE PROFIT POOL SITS, and why** (in most chains one layer earns most of the economics; the "why" is usually a moat mechanism at that layer). (3) Demand drivers; classify: secular grower / GDP-tracker / cyclical / declining — this label sets valuation expectations for every name in it. (4) Supply side: capacity lead times (long = boom-bust), entry barriers. (5) **The 3-5 KPIs insiders actually watch** — they differ by sector (same-store sales & inventory turns; ARR/NRR & CAC payback; loan growth & NIM; RevPAR; utilization & day-rates; book-to-bill) — name them, define them, give healthy ranges. (6) **Cycle position** and historical peak/trough margins — this is what stops you capitalizing peak earnings at a full multiple. (7) Competitive structure: consolidated or fragmented, rational or price-warring, share trends. (8) Regulatory frame and live policy risks. (9) The player map by chain layer, one line each. (10) **Conclusion: where in this industry would you want to own a business, and why** — plus the 2-3 things to monitor that would change the answer.
> Best free sources: the 10-K Item 1/1A of the 2-3 leaders, earnings transcripts, sector fundamentals across the cohort. Label understanding vs. sourced data.

---

# FUND OPERATIONS

## 26. Portfolio Memory Format
> Maintain the fund's book in this fixed block. Restate the WHOLE block on every change with a new as-of date — never append fragments.
> ```
> === ARY FUND PORTFOLIO (as of YYYY-MM-DD) ===
> Cash: XX.X% | Benchmark: [S&P 500]
> Single-name cap: XX% | Sector cap: XX%
>
> --- POSITION: TICKER ---
> Name: [Company]
> Weight: X.X% | Shares: N | Cost basis: $XX.XX | Entered: YYYY-MM-DD
> Rating: BUY/HOLD/SELL | Conviction: NN/100
> Thesis (1-2 lines): [why we own it]
> Key catalyst: [next dated event + trigger]
> Kill-criteria: [pre-committed exit/trim conditions]
> Last reviewed: YYYY-MM-DD
>
> --- WATCHLIST: TICKER ---
> Why watching: [1 line] | Entry trigger: [price/level/event]
> === END PORTFOLIO ===
> ```
> Rules: every position carries its thesis AND kill-criteria (a ticker + weight is a holding, not a decision). Weights + cash ≈ 100% — flag if not, don't silently normalize. NEVER store live prices or P&L — compute fresh (they go stale instantly). Dates everywhere; stale as-of dates get surfaced before any analysis. Closed positions move to the decision journal, not deleted silently. If a ticker isn't in the block, it isn't in the book — never invent a position.

## 27. Decision Journal
> Record this decision BEFORE the outcome is known. Memory rewrites itself after results (hindsight bias) — the pre-commitment is the whole point.
> ```
> === DECISION: TICKER — ACTION (YYYY-MM-DD) ===
> Decision: [Open 3% / Add to 5% / Trim half / Close / Pass]
> Price at decision: $XX.XX | Rating: [X] | Conviction: NN/100
> Thesis (2-3 lines): [the VARIANT view — what we believe the market doesn't]
> Expected outcome: [+25-40% over 12mo] | Probability: [our odds we're right]
> Base rates considered: [what usually happens in situations like this]
> Kill-criteria: [pre-committed conditions meaning we're wrong]
> What would prove us wrong: [specific evidence, not just price]
> Key risks accepted: [top 2 from the pre-mortem]
> Dissent: [who disagreed and why — record the minority view]
> Emotional state/context: [FOMO? post-loss? consensus pressure? be honest]
> === END DECISION ===
> ```
> The uncomfortable fields are the valuable ones. Record trims, adds, closes AND passes — they're all decisions.

## 28. Post-Mortem
> Grade this closed position. **Decision quality is NOT outcome quality** — a fund that grades only P&L learns to be lucky, not good.
> ```
> === POST-MORTEM: TICKER (closed YYYY-MM-DD) ===
> Result: [+/-XX% vs. expected +XX-XX%] | Holding period: [X months]
> vs. benchmark same period: [+/-XX%]
> Thesis verdict: RIGHT / WRONG / RIGHT-FOR-WRONG-REASONS / UNRESOLVED
> What actually happened: [2-3 lines, the causal story]
> Decision grade (PROCESS): GOOD / MIXED / POOR — independent of result
> Luck component: [what broke our way or against us that we didn't predict]
> Kill-criteria performance: [did they fire? did we OBEY them?]
> Calibration: [our stated odds vs. what happened]
> Lesson (one sentence): [the transferable rule, if any]
> === END POST-MORTEM ===
> ```
> The 2x2: good process + good outcome = deserved win. Good process + bad outcome = BAD LUCK, don't change the process over one loss. **Bad process + good outcome = DUMB LUCK — the most dangerous cell**, it teaches the wrong lesson and gets repeated until it doesn't work. Bad process + bad outcome = the clearest lesson.
> Grade against what was KNOWABLE AT THE TIME — the journal entry is the evidence.
> Quarterly, read ACROSS entries: calibration (when we said 70%, were we right ~70%?); recurring bias patterns; kill-criteria obedience vs. rationalization (the rationalization rate is the fund's real risk number); which THESIS TYPES work for this team; and grade the PASSES too.

---

## Quick index

| # | Prompt | Space |
|---|---|---|
| 1-3 | Memo, Review, Bias audit | 1 |
| 4-9 | Metrics, Valuation, Distress, Peers, Moat, DCF | 1 |
| 10-13 | Risk flags, Macro, Pulse, Quant signals | 3 (10 in 1) |
| 14-19 | Filings, Insider, Sentiment, Earnings, Catalysts, Special sits | 4 |
| 20-21 | Verdict, Screening | 1 |
| 22-25 | Sizing, Portfolio review, Pre-mortem, Industry primer | 2 (25 in 1) |
| 26-28 | Memory format, Journal, Post-mortem | 2 |
