from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True, slots=True)
class Coverage:
    """Data coverage metadata for a single ticker."""

    ticker: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    observations: int


def _normalize_adj_close(
    df: pd.DataFrame | pd.Series, tickers: list[str]
) -> pd.DataFrame:
    """Ensure we return a DataFrame with columns as tickers containing Adj Close values."""
    match df:
        case pd.Series():
            return df.to_frame(name=tickers[0])
        case pd.DataFrame() if df.columns.nlevels == 1:
            return df
        case pd.DataFrame():
            if "Adj Close" in df.columns.get_level_values(0):
                adj = df["Adj Close"].copy()
                adj.columns.name = None
                return adj
            if "Close" in df.columns.get_level_values(0):
                close = df["Close"].copy()
                close.columns.name = None
                return close
            raise ValueError(
                "Expected 'Adj Close' or 'Close' in yfinance download result"
            )
        case _:
            raise TypeError("Expected a pandas Series or DataFrame")


def fetch_adjusted_close(
    tickers: Iterable[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Fetch adjusted close prices for all tickers in one call when possible.

    Parameters
    ----------
    tickers: iterable of str
            Ticker symbols resolvable by yfinance.
    start, end: optional
            Date range for the fetch. If omitted, yfinance defaults are used.
    progress: bool
            Show yfinance progress bar.

    Returns
    -------
    DataFrame
            Adjusted close price panel, columns are tickers.
    """
    unique: list[str] = sorted(
        {str(t).strip().upper() for t in tickers if t and str(t).strip()}
    )
    if not unique:
        return pd.DataFrame()
    data = yf.download(
        tickers=unique,
        start=start,
        end=end,
        progress=progress,
        auto_adjust=False,
        actions=False,
        group_by="column",
    )
    if data is None or len(data) == 0:
        return pd.DataFrame(columns=unique)
    adj = _normalize_adj_close(data, unique)
    for t in unique:
        if t not in adj.columns:
            adj[t] = np.nan
    adj = adj[unique]
    adj.index = pd.to_datetime(adj.index)
    adj.sort_index(inplace=True)
    return adj


def align_on_common_dates(
    prices: pd.DataFrame, min_obs: int = 60
) -> tuple[pd.DataFrame, list[str]]:
    """Align series on common dates and drop assets with insufficient obs.

    Returns aligned frame and a list of dropped tickers.
    """
    if prices.empty:
        return prices, []
    dropped: list[str] = []
    counts = prices.notna().sum()
    for t in prices.columns:
        if counts.get(t, 0) < min_obs:
            dropped.append(t)
    prices2 = prices.drop(columns=dropped, errors="ignore")
    aligned = prices2.dropna(axis=0, how="any").copy()
    return aligned, dropped


def coverage_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a table with coverage info per ticker."""
    rows: list[dict[str, object]] = []
    for t in prices.columns:
        col = prices[t].dropna()
        start = col.index.min() if not col.empty else None
        end = col.index.max() if not col.empty else None
        rows.append(
            {
                "Ticker": t,
                "Start": start,
                "End": end,
                "Observations": int(col.shape[0]),
            }
        )
    return pd.DataFrame(rows)
