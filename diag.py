"""
diag.py — one CLI for every pipeline/data diagnostic.

Consolidates the diag_*.py family into a single argparse tool. Each subcommand
preserves the exact reads, SQL, and output of the script it replaces. Imports
are done lazily inside each command so a broken optional dependency (e.g.
chromadb) only affects that one subcommand, and the ordering of printed lines
still works as a native-crash tracer where that matters.

Run from the project root, venv active:

    python diag.py chroma
    python diag.py ragsmoke MSFT
    python diag.py filings MSFT
    python diag.py money-graph
    python diag.py metrics-drop
    python diag.py providers
    python diag.py rag-learning
    python diag.py risk                     # optional: risk AAPL COST MSFT ...
    python diag.py sector-coverage
    python diag.py peer-risk
    python diag.py universe-coverage
    python diag.py distressed

Subcommand -> replaced script
    chroma             diag_chroma.py
    ragsmoke           diag_crash.py                (staged RAG read/chunk/embed/store smoke)
    filings            diag_filings.py
    money-graph        diag_money_graph.py
    metrics-drop       diag_probe_metrics.py
    providers          diag_providers.py
    rag-learning       diag_rag_learning.py
    risk               diag_risk5.py                (supersedes diag_risk[1-4])
    sector-coverage    diag_sector_coverage2.py     (supersedes diag_sector_coverage)
    peer-risk          diag_stage2_peer_risk.py
    universe-coverage  diag_universe_coverage.py
    distressed         diag_distressed.py
"""
from __future__ import annotations

import argparse
import sys


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def _out(msg):
    print(msg, flush=True)   # flush so nothing is lost on a native crash


def _cfg_market_db():
    """(config_module_or_None, market_db_path, portfolio_db_path)."""
    try:
        from data import config as cfg  # type: ignore
    except Exception:
        try:
            import config as cfg        # type: ignore
        except Exception:
            cfg = None
    market = (getattr(cfg, "MARKET_DB_PATH", None)
              or getattr(cfg, "HEDGEFUND_DB_PATH", None)
              or "data/hedgefund.db")
    portfolio = getattr(cfg, "PORTFOLIO_DB_PATH", None) or "data/portfolio.db"
    return cfg, market, portfolio


def _universe():
    try:
        from data.universe import US_UNIVERSE
    except Exception:
        from universe import US_UNIVERSE  # type: ignore
    return US_UNIVERSE


def _sec_fetcher(db_path=None):
    from data.sec_fetcher import SECFetcher
    return SECFetcher(db_path=db_path) if db_path else SECFetcher()


def _money_graph():
    try:
        from data import money_graph as mg
    except Exception:
        import money_graph as mg  # type: ignore
    return mg


# ======================================================================
# chroma  (was diag_chroma.py)
# ======================================================================
def cmd_chroma(args: argparse.Namespace) -> int:
    _out("Chroma versions:")
    try:
        import chromadb
        _out(f"  chromadb = {getattr(chromadb, '__version__', '?')}")
    except Exception as e:
        _out(f"  chromadb import FAILED: {e}")
        return 1
    try:
        import numpy as np
        _out(f"  numpy = {np.__version__}")
    except Exception as e:
        _out(f"  numpy import FAILED: {e}")

    import config
    _out(f"\nOpening PersistentClient at {config.RAG_VECTOR_STORE_PATH} ...")
    client = chromadb.PersistentClient(path=config.RAG_VECTOR_STORE_PATH)
    _out("  ok")

    _out("Getting/creating a TEST collection ...")
    col = client.get_or_create_collection(name="diag_test")
    _out("  ok")

    _out("\nWrite 1 row (dim 768) ...")
    col.upsert(
        ids=["diag_1"],
        embeddings=[[0.01] * 768],
        documents=["hello world"],
        metadatas=[{"doc_id": "diag", "ticker": "TEST"}],
    )
    _out("  ok — 1 row written")

    _out("\nWrite 10 rows ...")
    col.upsert(
        ids=[f"diag_{i}" for i in range(10)],
        embeddings=[[0.01 * (i + 1)] * 768 for i in range(10)],
        documents=[f"doc {i}" for i in range(10)],
        metadatas=[{"doc_id": "diag", "ticker": "TEST", "n": i} for i in range(10)],
    )
    _out("  ok — 10 rows written")

    _out("\nCount in test collection:")
    _out(f"  {col.count()}")

    _out("\nCleaning up test collection ...")
    client.delete_collection("diag_test")
    _out("  ok")

    _out("\nALL CHROMA WRITES PASSED.")
    return 0


# ======================================================================
# ragsmoke  (was diag_crash.py)
# ======================================================================
def cmd_ragsmoke(args: argparse.Namespace) -> int:
    """Runs the RAG index path stage by stage with flush-after-print so the
    last line before any native crash names the culprit layer."""
    ticker = args.ticker.upper()

    _out("STAGE 0: importing config + sec_fetcher ...")
    import config
    from data.sec_fetcher import SECFetcher
    _out("  ok")

    _out("STAGE 0b: importing rag.embedder ...")
    from rag.embedder import Embedder
    _out("  ok")

    _out("STAGE 0c: importing rag.vector_store ...")
    from rag.vector_store import VectorStore
    _out("  ok")

    _out("STAGE 0d: importing chunker ...")
    from rag.chunker import chunk_document
    _out("  ok")

    _out("\nSTAGE 1: reading one 10-K text from DB ...")
    sec = SECFetcher(db_path=config.PORTFOLIO_DB_PATH)
    cached = sec._get_cached_filings(ticker, "10-K", 1, None, None)
    _out(f"  cached rows: {len(cached)}")
    acc = cached[0]["accession_number"]
    _out(f"  accession: {acc}")
    text = sec.get_filing_text(acc)
    _out(f"  text length: {len(text):,} chars")

    _out("\nSTAGE 2: chunking ...")
    chunks = chunk_document(
        raw_text=text, doc_id=f"{ticker}_TEST", doc_type="filing",
        base_metadata={"ticker": ticker, "doc_type": "filing", "doc_id": f"{ticker}_TEST"},
        chunk_tokens=getattr(config, "RAG_CHUNK_TOKENS", 500),
        overlap_tokens=getattr(config, "RAG_OVERLAP_TOKENS", 50),
    )
    _out(f"  produced {len(chunks)} chunks")

    _out("\nSTAGE 3: building Embedder (forces ollama) ...")
    import os
    os.environ.setdefault("ARY_EMBED_BACKEND", "ollama")
    emb = Embedder(cache_db_path=getattr(config, "RAG_EMBEDDING_CACHE_DB", None))
    _out(f"  backend={emb.backend_name} dim={emb.dimension}")

    _out("\nSTAGE 4: embedding first 3 chunks via Ollama ...")
    sample = [c.text for c in chunks[:3]]
    vecs = emb.embed(sample, role="document")
    _out(f"  got embeddings shape: {vecs.shape}")

    _out(f"\nSTAGE 5: embedding ALL {len(chunks)} chunks ...")
    all_inputs = [c.text for c in chunks]
    all_vecs = emb.embed(all_inputs, role="document")
    _out(f"  got embeddings shape: {all_vecs.shape}")

    _out("\nSTAGE 6: opening VectorStore ...")
    store = VectorStore(
        persist_path=config.RAG_VECTOR_STORE_PATH,
        embedding_dim=768,
    )
    _out("  ok")

    _out(f"\nSTAGE 7: upserting {len(chunks)} chunks ...")
    store.upsert_chunks(chunks, all_vecs)
    _out("  ok — WROTE SUCCESSFULLY")

    _out("\nALL STAGES PASSED. No crash.")
    return 0


# ======================================================================
# filings  (was diag_filings.py)
# ======================================================================
def cmd_filings(args: argparse.Namespace) -> int:
    import traceback
    import config

    ticker = args.ticker.upper()
    print(f"Testing FilingsLoader for {ticker}\n")

    sec = _sec_fetcher(config.PORTFOLIO_DB_PATH)

    for ft, count in [("10-K", 3), ("10-Q", 2), ("8-K", 10), ("DEF 14A", 2)]:
        try:
            rows = sec.get_filings(ticker, ft, count=count) or []
            print(f"get_filings({ticker!r}, {ft!r}, count={count}) -> {len(rows)} rows")
            for r in rows[:1]:
                print(f"    sample keys: {sorted(r.keys())}")
                print(f"    accession: {r.get('accession_number')}  "
                      f"date: {r.get('filed_date') or r.get('filing_date')}")
        except Exception as e:  # noqa: BLE001
            print(f"get_filings({ft}) FAILED: {e}")
            traceback.print_exc()

    print("\n--- FilingsLoader.load_for_ticker ---")
    try:
        from rag.document_loaders.filings import FilingsLoader
    except Exception as e:  # noqa: BLE001
        print(f"IMPORT FAILED: {e}")
        traceback.print_exc()
        return 1

    fl = FilingsLoader(sec_fetcher=sec)
    docs = list(fl.load_for_ticker(ticker))
    print(f"load_for_ticker yielded {len(docs)} document(s)")
    for d in docs:
        text_len = len(getattr(d, "text", "") or "")
        print(f"  doc_id={getattr(d, 'doc_id', '?')}  "
              f"type={getattr(d, 'doc_type', '?')}  text={text_len:,} chars")

    if not docs:
        print("\n>>> Loader yielded NOTHING. The per-type get_filings calls "
              "above tell us whether rows came back; if they did but the "
              "loader yielded 0, the break is inside _build_document "
              "(accession/text fetch).")
    return 0


# ======================================================================
# money-graph  (was diag_money_graph.py)
# ======================================================================
def cmd_money_graph(args: argparse.Namespace) -> int:
    import sqlite3
    _, market_db, portfolio_db = _cfg_market_db()
    print(f"market DB   = {market_db}")
    print(f"portfolio DB= {portfolio_db}\n")

    def q(db, sql, params=()):
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
                return c.execute(sql, params).fetchall()
        except Exception as e:  # noqa: BLE001
            print(f"  query error: {e}")
            return []

    mg = _money_graph()
    uni = mg.universe_tickers()
    print(f"[1] universe_tickers() -> {len(uni)} names "
          f"(expect ~560).  first few: {uni[:8]}\n")

    total = q(market_db, "SELECT COUNT(*) FROM ownership_filings")
    total = total[0][0] if total else 0
    nonempty = q(market_db,
                 "SELECT COUNT(*) FROM ownership_filings "
                 "WHERE filer_name IS NOT NULL AND TRIM(filer_name) <> ''")
    nonempty = nonempty[0][0] if nonempty else 0
    distinct = q(market_db,
                 "SELECT COUNT(DISTINCT filer_name) FROM ownership_filings "
                 "WHERE filer_name IS NOT NULL AND TRIM(filer_name) <> ''")
    distinct = distinct[0][0] if distinct else 0
    tickers = q(market_db, "SELECT COUNT(DISTINCT ticker) FROM ownership_filings")
    tickers = tickers[0][0] if tickers else 0
    print("[2] ownership_filings:")
    print(f"      total rows            = {total}")
    print(f"      rows WITH filer_name  = {nonempty}   <-- edges can only come from these")
    print(f"      distinct filer_names  = {distinct}")
    print(f"      distinct tickers      = {tickers}")
    if total:
        print(f"      -> {100*nonempty/total:.1f}% of ownership rows are usable")
    print("      top filer_name values (by count):")
    for name, n in q(market_db,
                     "SELECT filer_name, COUNT(*) c FROM ownership_filings "
                     "GROUP BY filer_name ORDER BY c DESC LIMIT 12"):
        shown = repr(name)[:50]
        print(f"        {n:>5}  {shown}")

    f13f = q(market_db, "SELECT COUNT(*) FROM f13f_holdings")
    f13f = f13f[0][0] if f13f else 0
    print(f"\n[3] f13f_holdings rows = {f13f}   "
          f"(0 = you haven't ingested 13F yet; this is the real density source)")

    g = mg.build_money_graph(tickers=uni or None, portfolio_db_path=portfolio_db,
                             market_db_path=market_db, market_data=None,
                             drop_isolated=True, max_nodes=600)
    m = g["meta"]
    print("\n[4] build_money_graph(full universe, offline):")
    print(f"      source        = {m.get('source')}")
    print(f"      scope_size    = {m.get('scope_size')}")
    print(f"      rendered nodes= {m.get('rendered_nodes', len(g['nodes']))}")
    print(f"      rendered edges= {m.get('rendered_edges', len(g['edges']))}")
    print(f"      provenance    = {m.get('provenance')}")

    print("\nDIAGNOSIS:")
    if nonempty < total * 0.2:
        print("  -> Most ownership rows have EMPTY filer_name (filer lives in the")
        print("     filing header, not the subject feed). Run: backfill.py ownership-filers")
    if f13f == 0:
        print("  -> No 13F data. For a dense who-owns-whom graph, ingest 13F for the")
        print("     big institutions: backfill.py 13f")
    return 0


# ======================================================================
# metrics-drop  (was diag_probe_metrics.py)
# ======================================================================
def cmd_metrics_drop(args: argparse.Namespace) -> int:
    import config
    from data.market_data import MarketData
    from agent.filing_analyzer import summarize_filings_by_year
    from data import peer_stats as ps

    db_path = config.PORTFOLIO_DB_PATH
    md = MarketData(db_path=db_path)
    sec = _sec_fetcher(db_path)

    uni = ps.get_universe_tickers(config)
    print(f"get_universe_tickers -> {len(uni)} tickers; first 10: {uni[:10]}\n")

    print(f"{'ticker':6} {'sector (get_fundamentals)':28} {'risk_count':>10}")
    sample = uni[:15]
    sector_ok = 0
    for tk in sample:
        try:
            f = md.get_fundamentals(tk, use_cache=True)
            sec_val = (f.get("sector") if isinstance(f, dict) else None) or "(none)"
        except Exception as e:
            sec_val = f"ERR: {e}"

        rc = 0
        try:
            filings = []
            for kind in ("10-K", "10-Q"):
                cached = sec._get_cached_filings(tk, kind, 3, None, None)
                for fl in cached:
                    acc = fl.get("accession_number")
                    if not acc:
                        continue
                    txt = sec.get_filing_text(acc)
                    if txt:
                        filings.append({**fl, "text": txt})
            if filings:
                summ = summarize_filings_by_year(tk, filings)
                rc = int(summ.get("risk_factor_count") or 0)
        except Exception as e:
            rc = f"ERR: {e}"

        if sec_val not in ("(none)",) and not str(sec_val).startswith("ERR"):
            sector_ok += 1
        print(f"{tk:6} {str(sec_val):28} {str(rc):>10}")

    print(f"\n{sector_ok}/{len(sample)} sample tickers returned a sector.")

    print("\nRunning full compute_all_sector_peer_stats over the universe...")

    def _metrics_for(tk):
        try:
            f = md.get_fundamentals(tk, use_cache=True)
        except Exception:
            return None
        if not isinstance(f, dict):
            return None
        s = f.get("sector")
        s = s.strip() if isinstance(s, str) and s.strip() else None
        if s is None:
            return None
        rc = 0
        try:
            filings = []
            for kind in ("10-K", "10-Q"):
                for fl in sec._get_cached_filings(tk, kind, 3, None, None):
                    acc = fl.get("accession_number")
                    if not acc:
                        continue
                    txt = sec.get_filing_text(acc)
                    if txt:
                        filings.append({**fl, "text": txt})
            if filings:
                rc = int(summarize_filings_by_year(tk, filings).get("risk_factor_count") or 0)
        except Exception:
            rc = 0
        return {"sector": s, "risk_factor_count": rc}

    stats = ps.compute_all_sector_peer_stats(_metrics_for, uni)
    print(f"\nRESULT: {len(stats)} sector(s)")
    for s in sorted(stats):
        rc = stats[s].get("risk_factor_count")
        if rc:
            print(f"  {s:26} mean={rc['mean']:6.1f} std={rc['std']:6.1f} n={rc['n']}")
    return 0


# ======================================================================
# providers  (was diag_providers.py)
# ======================================================================
def cmd_providers(args: argparse.Namespace) -> int:
    from datetime import date, timedelta
    try:
        from data import providers
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: cannot import data.providers: {e}")
        return 1

    TICKER = "MSFT"
    today = date.today()
    start = (today - timedelta(days=30)).isoformat()
    end = today.isoformat()
    cal_from = today.isoformat()
    cal_to = (today + timedelta(days=14)).isoformat()

    def _peek(obj):
        try:
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                return f"DataFrame rows={len(obj)} cols={list(obj.columns)[:8]}"
        except Exception:
            pass
        if isinstance(obj, dict):
            return f"dict keys={list(obj.keys())[:8]}"
        if isinstance(obj, list):
            head = obj[0] if obj else None
            if isinstance(head, dict):
                return f"list len={len(obj)} first_keys={list(head.keys())[:8]}"
            return f"list len={len(obj)}"
        if obj is None:
            return "None"
        return f"{type(obj).__name__}: {str(obj)[:80]}"

    def check(name, fn):
        print(f"\n--- {name} ---")
        try:
            rv = fn()
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL ({type(e).__name__}): {str(e)[:200]}")
            return False
        desc = _peek(rv)
        empty = (rv is None or (hasattr(rv, "__len__") and len(rv) == 0))
        flag = "EMPTY (check key/tier/endpoint)" if empty else "OK"
        print(f"  {flag}: {desc}")
        return not empty

    results = {}
    results["tiingo.get_prices"] = check(
        "Tiingo get_prices (adjusted OHLCV)",
        lambda: providers.get_prices(TICKER, start, end))
    results["tiingo.get_fundamentals"] = check(
        "Tiingo get_fundamentals (note: paid tier on free plan)",
        lambda: providers.get_fundamentals(TICKER))
    results["fmp.get_analyst_data"] = check(
        "FMP get_analyst_data (estimates/ratios/dcf)",
        lambda: providers.get_analyst_data(TICKER))
    results["fmp.get_transcripts"] = check(
        "FMP get_transcripts (earnings calls)",
        lambda: providers.get_transcripts(TICKER))
    results["finnhub.get_earnings_events"] = check(
        "Finnhub get_earnings_events (calendar)",
        lambda: providers.get_earnings_events(cal_from, cal_to))
    results["finnhub.get_ownership_data"] = check(
        "Finnhub get_ownership_data (institutional holders)",
        lambda: providers.get_ownership_data(TICKER))

    print("\n" + "=" * 60)
    ok = sum(1 for v in results.values() if v)
    print(f"SUMMARY: {ok}/{len(results)} returned non-empty data")
    for k, v in results.items():
        print(f"  {'OK ' if v else '-- '} {k}")
    print("\nNote: an EMPTY result isn't necessarily broken — Tiingo "
          "fundamentals and some FMP endpoints are gated on paid tiers. "
          "What matters is which ones return data you actually want to use.")
    return 0


# ======================================================================
# rag-learning  (was diag_rag_learning.py)
# ======================================================================
def cmd_rag_learning(args: argparse.Namespace) -> int:
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from datetime import datetime, timezone, timedelta
    import importlib

    def _imp(*candidates):
        last = None
        for c in candidates:
            try:
                return importlib.import_module(c)
            except Exception as e:  # noqa: BLE001
                last = e
        raise ImportError(f"none of {candidates} importable; last error: {last}")

    print("=" * 64)
    print("RAG LEARNING DIAGNOSTIC — feeding one winning closed thesis")
    print("=" * 64)

    loop_mod = _imp("rag.learning.loop", "loop")
    curator_mod = _imp("rag.learning.curator", "curator")
    indexer_mod = _imp("rag.indexer", "indexer")

    LearningLoop = loop_mod.LearningLoop
    Curator = curator_mod.Curator
    Indexer = indexer_mod.Indexer

    print("\n[1] Building Curator + Indexer + LearningLoop...")
    curator = Curator()
    try:
        indexer = Indexer()
        emb = getattr(indexer, "embedder", None)
        dim = getattr(emb, "dimension", "?")
        name = type(emb).__name__ if emb else "?"
        print(f"    embedder = {name}  (dim={dim})")
    except Exception as e:  # noqa: BLE001
        print(f"    ERROR building Indexer (embedder/store): {e}")
        print("    If this is an Ollama/embedder issue, start `ollama serve` "
              "and/or set ARY_EMBED_BACKEND=ollama, then retry.")
        return 1

    def _fake_pnl(thesis: dict):
        return {"return_pct": 0.35, "days_held": 60, "benchmark_return_pct": 0.05}

    loop = LearningLoop(curator=curator, auditor=None, indexer=indexer,
                        pnl_lookup_fn=_fake_pnl)

    now = datetime.now(timezone.utc)
    thesis = {
        "id": f"DIAG-RAG-{now.strftime('%Y%m%d%H%M%S')}",
        "ticker": "NVDA",
        "created_at": (now - timedelta(days=60)).isoformat(),
        "score": 0.9,
        "thesis_text": (
            "NVIDIA (NVDA) — closed long. Thesis: the Blackwell data-center "
            "ramp plus CUDA lock-in would drive durable revenue growth and "
            "margin expansion. Realized: position held ~60 days for a +35% "
            "return versus +5% for the benchmark, validating the AI-demand "
            "thesis. Key driver played out as expected; risk from hyperscaler "
            "capex digestion did not materialize in the window."
        ),
        "outlook": "bullish",
    }

    def _stats():
        try:
            return curator.stats()
        except Exception:
            return {}

    before = _stats()
    print(f"\n[2] Corpus stats BEFORE: active="
          f"{before.get('active', '?')}  demoted={before.get('demoted', '?')}")

    print("\n[3] Running process_closed_theses on the winning thesis...")
    result = loop.process_closed_theses([thesis])
    print(f"    indexed={result.get('indexed')}  blocked={result.get('blocked')}"
          f"  error={result.get('error')}")
    for d in result.get("decisions", []):
        print(f"    decision: {d.get('decision')}  thesis={d.get('thesis_id')}"
              f"  reasons={d.get('reasons', d.get('block_reasons', ''))}")

    try:
        decision = curator.decide_indexable(thesis, realized_pnl=_fake_pnl(thesis))
        q = getattr(decision, "quality", None)
        print(f"\n[4] Curator: should_index={getattr(decision, 'should_index', '?')}"
              f"  composite={getattr(q, 'composite', '?')}"
              f"  threshold={curator.index_threshold}")
        if getattr(q, "warnings", None):
            print(f"    warnings: {q.warnings}")
    except Exception as e:  # noqa: BLE001
        print(f"\n[4] (couldn't re-score for detail: {e})")

    after = _stats()
    print(f"\n[5] Corpus stats AFTER:  active="
          f"{after.get('active', '?')}  demoted={after.get('demoted', '?')}")

    print("\n" + "=" * 64)
    if result.get("indexed", 0) > 0:
        print("RESULT: The loop INDEXED the winning thesis — learning path is "
              "proven. The curator scored it, the indexer pushed it into the "
              "vector store, and the corpus grew.")
    else:
        print("RESULT: The thesis was NOT indexed. See the block reasons / "
              "composite above. Common causes: composite below threshold "
              f"({curator.index_threshold}), or an embedder/store issue. This "
              "still shows the pipeline RAN end-to-end; it just declined to "
              "index this input.")
    print("=" * 64)
    print("\nNote: this wrote a test thesis (id starts with DIAG-RAG-) into the "
          "tracking DB / vector store. It does not affect your portfolio.")
    return 0


# ======================================================================
# risk  (was diag_risk5.py — supersedes diag_risk[1-4])
# ======================================================================
def cmd_risk(args: argparse.Namespace) -> int:
    import config
    from agent import filing_analyzer as fa

    sec = _sec_fetcher(config.PORTFOLIO_DB_PATH)
    tickers = ([t.upper() for t in args.tickers]
               if args.tickers else
               ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "COST", "PEP", "AXP"])

    for tkr in tickers:
        rows = sec._get_cached_filings(tkr, "10-K", 1, None, None)
        if not rows:
            print(f"\n{tkr}: no cached 10-K")
            continue
        t = sec.get_filing_text(rows[0]["accession_number"])
        region = fa._select_risk_region(t)
        risks = fa._extract_risk_sentences(t)
        used = "Item1A section" if region is not None else "WHOLE-DOC fallback"
        rlen = len(region) if region is not None else 0
        print(f"\n===== {tkr} =====")
        print(f"  path           : {used}"
              + (f" ({rlen:,} chars)" if region is not None else ""))
        print(f"  risk sentences : {len(risks)}")
        for r in risks[:4]:
            print(f"      - {r[:100]}")
    return 0


# ======================================================================
# sector-coverage  (was diag_sector_coverage2.py — supersedes v1)
# ======================================================================
def cmd_sector_coverage(args: argparse.Namespace) -> int:
    import json
    import sqlite3
    from collections import Counter
    import config

    US_UNIVERSE = _universe()
    conn = sqlite3.connect(config.PORTFOLIO_DB_PATH)

    try:
        have_text = {r[0] for r in conn.execute(
            "SELECT DISTINCT ticker FROM sec_filings "
            "WHERE full_text IS NOT NULL AND full_text != ''"
        ).fetchall()}
    except sqlite3.OperationalError:
        have_text = set()

    # Sector per ticker from fundamentals_cache JSON blobs (the real location).
    sector_of = {}
    try:
        for tk, blob in conn.execute(
            "SELECT ticker, data_json FROM fundamentals_cache"
        ).fetchall():
            try:
                d = json.loads(blob)
                s = (d.get("sector") or "").strip()
                if s:
                    sector_of[tk] = s
            except Exception:
                continue
    except sqlite3.OperationalError:
        print("(no fundamentals_cache table found)")

    uni = set(US_UNIVERSE)
    counts = Counter()
    text_no_sector = 0
    for t in uni:
        if t not in have_text:
            continue
        s = sector_of.get(t)
        if not s:
            text_no_sector += 1
            continue
        counts[s] += 1

    print(f"universe size                  : {len(uni)}")
    print(f"hydrated (filing text)         : {len(have_text & uni)}")
    print(f"  ... with a resolvable sector : {sum(counts.values())}")
    print(f"  ... text but NO sector       : {text_no_sector}")
    print("\nper-sector usable peers (>=2 needed for peer stats):")
    usable = 0
    for sector, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        flag = "OK" if n >= 2 else "THIN"
        if n >= 2:
            usable += 1
        print(f"  {sector:26} {n:3}  {flag}")
    print(f"\nsectors with usable peer stats : {usable}")
    return 0


# ======================================================================
# peer-risk  (was diag_stage2_peer_risk.py)
# ======================================================================
def cmd_peer_risk(args: argparse.Namespace) -> int:
    import logging
    import config
    from data import pipeline as pl
    from data import peer_stats as ps
    from agent.thesis_generator import _risk_count_penalty

    logging.basicConfig(level=logging.WARNING)

    WATCHLIST = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "COST", "PEP", "AXP"]
    KNOWN = {"MSFT": 119, "AAPL": 67, "NVDA": 52, "GOOGL": 25,
             "AMZN": 51, "COST": 67, "PEP": 20, "AXP": 103}

    db_path = config.PORTFOLIO_DB_PATH
    print("Forcing full-universe peer-stats recompute (this is the slow part)...\n")
    stats = pl.get_sector_peer_stats(db_path, config, force=True)

    print(f"computed stats for {len(stats)} sector(s)\n")

    print("=== risk_factor_count distribution by sector ===")
    for sector in sorted(stats):
        rc = stats[sector].get("risk_factor_count")
        if not rc:
            continue
        print(f"  {sector:26} mean={rc['mean']:6.1f}  std={rc['std']:6.1f}  n={rc['n']}")

    print("\n=== watchlist: risk count vs sector peers ===")
    print(f"  {'ticker':6} {'sector':24} {'count':>5} {'mean':>6} {'std':>6} {'z':>5} {'penalty':>8}")
    for tk in WATCHLIST:
        try:
            ctx = pl.build_agent_context(tk, db_path, config)
            metrics_raw = ctx.get("metrics") or {}
            sector = (metrics_raw.get("sector") or "").strip().lower()
        except Exception as e:
            print(f"  {tk:6} (context failed: {e})")
            continue

        peer = ps.peer_stats_for_sector(stats, sector) if sector else None
        rc = (peer or {}).get("risk_factor_count") if peer else None
        count = KNOWN.get(tk, 0)
        if rc and rc.get("std", 0) > 0:
            z = (count - rc["mean"]) / rc["std"]
            pen = _risk_count_penalty(count, peer)
            print(f"  {tk:6} {sector:24} {count:5} {rc['mean']:6.1f} {rc['std']:6.1f} "
                  f"{z:5.2f} {-pen:8.2f}")
        else:
            print(f"  {tk:6} {sector:24} {count:5}    (no usable sector distribution)")

    print("\nDone. A correct cache is now written; the next main.py run will use it.")
    return 0


# ======================================================================
# universe-coverage  (was diag_universe_coverage.py)
# ======================================================================
def cmd_universe_coverage(args: argparse.Namespace) -> int:
    import sqlite3
    import config

    US_UNIVERSE = _universe()
    c = sqlite3.connect(config.PORTFOLIO_DB_PATH)
    uni = set(US_UNIVERSE)

    def _distinct(where=""):
        try:
            qy = "SELECT DISTINCT ticker FROM sec_filings"
            if where:
                qy += " WHERE " + where
            return set(r[0] for r in c.execute(qy).fetchall())
        except sqlite3.OperationalError as e:
            print("  (query failed:", e, ")")
            return set()

    have = _distinct()
    withtext = _distinct("full_text IS NOT NULL AND full_text != ''")
    have_uni = have & uni

    print("universe size              :", len(uni))
    print("universe w/ ANY filing row :", len(have_uni))
    print("universe w/ full_text      :", len(have_uni & withtext))
    print("universe needing FETCH     :", len(uni - have))
    print()
    need = len(uni - have)
    calls = need * 5
    print(f"~{need} tickers to fetch, ~{calls} EDGAR calls")
    print(f"at ~5 req/s polite rate: ~{calls/5/60:.0f} min of fetching (rough)")
    return 0


# ======================================================================
# distressed  (was diag_distressed.py)
# ======================================================================
_DISTRESSED_TEXT = """
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

_CLEAN_TEXT = """
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


def cmd_distressed(args: argparse.Namespace) -> int:
    from agent.filing_analyzer import summarize_filings_by_year
    from agent.thesis_generator import _score_filings_bias

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

    _run("DISTRESSED filing (expect defensive, red flags > 0, bias strongly negative)",
         _DISTRESSED_TEXT)
    _run("CLEAN filing (expect confident/neutral, red_flags=0, bias >= 0)",
         _CLEAN_TEXT)
    print()
    return 0


# ======================================================================
# Parser
# ======================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified pipeline/data diagnostics.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("chroma", help="minimal ChromaDB write test").set_defaults(func=cmd_chroma)

    rs = sub.add_parser("ragsmoke", help="staged RAG read/chunk/embed/store smoke test")
    rs.add_argument("ticker", nargs="?", default="MSFT")
    rs.set_defaults(func=cmd_ragsmoke)

    fi = sub.add_parser("filings", help="test FilingsLoader for a ticker")
    fi.add_argument("ticker", nargs="?", default="MSFT")
    fi.set_defaults(func=cmd_filings)

    sub.add_parser("money-graph", help="diagnose an empty Money Flow Graph").set_defaults(func=cmd_money_graph)
    sub.add_parser("metrics-drop", help="find where universe tickers get dropped").set_defaults(func=cmd_metrics_drop)
    sub.add_parser("providers", help="live smoke test of Tiingo/FMP/Finnhub").set_defaults(func=cmd_providers)
    sub.add_parser("rag-learning", help="prove the RAG learning loop indexes a winner").set_defaults(func=cmd_rag_learning)

    rk = sub.add_parser("risk", help="verify risk-factor extraction against cached 10-Ks")
    rk.add_argument("tickers", nargs="*", help="default: the 8-name watchlist")
    rk.set_defaults(func=cmd_risk)

    sub.add_parser("sector-coverage", help="per-sector hydrated peer coverage").set_defaults(func=cmd_sector_coverage)
    sub.add_parser("peer-risk", help="validate sector-relative risk-count scoring").set_defaults(func=cmd_peer_risk)
    sub.add_parser("universe-coverage", help="how much of the universe has filings").set_defaults(func=cmd_universe_coverage)
    sub.add_parser("distressed", help="verify the troubled-filing path end to end").set_defaults(func=cmd_distressed)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
