"""
diag_distressed.py — verify the troubled-filing path end to end.

All 8 watchlist tickers are clean blue-chips, so the defensive / red-flag
branch of the filing analyzer never fires in normal runs. This feeds
realistic DISTRESSED 10-K language through the actual production chain:

    summarize_filings_by_year(...)  ->  _score_filings_bias(...)

and prints the resulting tone / red_flags / risk count / filings bias, so
we can confirm the negative case behaves (defensive tone, red flags fire,
bias drives strongly negative) — the mirror image of the healthy MSFT case.

For contrast it also runs a clean filing through the same path.

    python diag_distressed.py
"""

from agent.filing_analyzer import summarize_filings_by_year
from agent.thesis_generator import _score_filings_bias


# --- Realistic distressed 10-K excerpts (affirmative declarations) ----------
# These mirror the language an actually-troubled company uses: declared
# going concern, an identified material weakness, an actual restatement,
# plus the usual routine risk-factor boilerplate around them.
DISTRESSED_TEXT = """
ITEM 1A. RISK FACTORS
Our recent operating losses and negative cash flows raise substantial doubt
about our ability to continue as a going concern. Our auditors have included
an explanatory paragraph in their report expressing this substantial doubt.

During fiscal 2025, management identified a material weakness in our internal
control over financial reporting related to revenue recognition. As a result,
we concluded that our disclosure controls and procedures were not effective.

We have restated our previously issued consolidated financial statements for
fiscal 2023 and 2024 to correct errors in the timing of revenue.

We recorded an impairment of goodwill of $412 million during the period. We
were notified by our lenders of a covenant breach under our senior credit
facility and are currently in discussions regarding a waiver.

We face intense competition and our results may be adversely affected by
macroeconomic conditions, foreign currency fluctuations, and litigation in
the ordinary course of business.
"""

# --- Clean filing (healthy mega-cap language) for contrast ------------------
CLEAN_TEXT = """
ITEM 1A. RISK FACTORS
We face intense competition across all markets for our products and services.
A material weakness is a deficiency, or combination of deficiencies, such that
there is a reasonable possibility that a material misstatement will not be
prevented or detected. A material weakness exists when such a deficiency is
present. Based on management's assessment, our internal control over financial
reporting was effective as of year end, and no material weakness was identified.

We delivered record revenue with strong demand, robust growth, expanding
operating margins, and continued market leadership across our cloud and
productivity segments. From time to time we are subject to litigation and
regulatory investigation in the ordinary course of business.
"""


def _run(label: str, text: str) -> None:
    filings = [{
        "accession_number": "TEST-0001",
        "filing_type": "10-K",
        "filed_date": "2025-07-30",
        "text": text,
    }]
    summary = summarize_filings_by_year("TEST", filings, max_filings=10)
    tone = summary.get("management_tone")
    red_flags = summary.get("red_flags") or []
    risk_factors = summary.get("risk_factors") or []
    bias = _score_filings_bias(summary)

    print(f"\n=== {label} ===")
    print(f"  tone          : {tone}")
    print(f"  red_flags     : {len(red_flags)}")
    for rf in red_flags:
        print(f"      - {rf[:90]}")
    print(f"  risk_factors  : {len(risk_factors)}")
    print(f"  filings_bias  : {bias:+.2f}")


if __name__ == "__main__":
    _run("DISTRESSED filing (expect defensive, red flags > 0, bias strongly negative)",
         DISTRESSED_TEXT)
    _run("CLEAN filing (expect confident/neutral, red_flags=0, bias >= 0)",
         CLEAN_TEXT)
    print()
