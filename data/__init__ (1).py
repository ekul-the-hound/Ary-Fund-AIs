"""
data — Ary Quant data layer package.

Provides the full ingestion, normalization, and derived-signal pipeline.

Quick start::

    from data.data_registry import DataRegistry
    from data.market_data import MarketData
    from data.refresh_scheduler import RefreshScheduler

    sch = RefreshScheduler(tickers=["AAPL", "MSFT"])
    sch.run_daily()
"""
