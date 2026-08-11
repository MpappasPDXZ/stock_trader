"""Production market-ML pipeline (parallel multi-ticker runner).

Notebook ``market_ml_analysis.ipynb`` remains the ideation surface.
This package runs the same canonical stages headlessly for many tickers.
"""

from market_ml.pipeline import RunConfig, run_ticker, run_tickers

__all__ = ["RunConfig", "run_ticker", "run_tickers"]
