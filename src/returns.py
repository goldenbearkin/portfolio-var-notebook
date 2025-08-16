from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


Timeframe = Literal["daily", "weekly", "monthly"]


def resample_prices(prices: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Resample prices to the selected timeframe.

    - daily: business-day frequency, last observation
    - weekly/monthly: period-end, last observation
    """
    if prices.empty:
        return prices
    tf = timeframe.lower()
    if tf == "daily":
        resampled = prices.resample("B").last().dropna(how="all")
    elif tf == "weekly":
        resampled = prices.resample("W-FRI").last().dropna(how="all")
    elif tf == "monthly":
        resampled = prices.resample("M").last().dropna(how="all")
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return resampled


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price panel."""
    if prices.empty:
        return prices
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return returns


def latest_prices_from_frame(prices: pd.DataFrame) -> pd.Series:
    """Return the last available price for each column."""
    if prices.empty:
        return pd.Series(dtype=float)
    return prices.tail(1).T.iloc[:, 0]


def portfolio_pnl_series(
    asset_returns: pd.DataFrame, prices: pd.DataFrame, positions: pd.Series
) -> pd.Series:
    """Compute portfolio P&L time series in currency terms.

    Per-asset P&L_t ≈ position_units * price_{t-1} * asset_log_return_t.
    Sum across assets to obtain portfolio P&L per date.
    """
    if asset_returns.empty or prices.empty or positions.empty:
        return pd.Series(dtype=float)
    common = asset_returns.columns.intersection(prices.columns).intersection(
        positions.index
    )
    if common.empty:
        return pd.Series(dtype=float)
    aligned_returns = asset_returns[common].copy()
    price_lag = prices[common].shift(1).reindex(aligned_returns.index)
    per_asset_pnl = aligned_returns * price_lag
    per_asset_pnl = per_asset_pnl.mul(positions[common], axis=1)
    pnl = per_asset_pnl.sum(axis=1)
    return pnl.dropna()


def portfolio_return_series(
    asset_returns: pd.DataFrame, exposures: pd.Series
) -> pd.Series:
    """Compute portfolio percent return series from asset returns and currency exposures.

    Weights are exposures normalized by their sum (ignoring non-positive total leads to zeros).
    """
    common = asset_returns.columns.intersection(exposures.index)
    if common.empty or asset_returns.empty:
        return pd.Series(dtype=float)
    exp = exposures[common]
    total = float(exp.sum())
    if total == 0:
        return pd.Series(0.0, index=asset_returns.index)
    weights = exp / total
    portfolio_ret = asset_returns[common].dot(weights)
    return portfolio_ret.dropna()


def returns_statistics(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute per-asset statistics: mean, std, skew, kurtosis.

    Values are per-period; interpret according to selected timeframe.
    """
    if asset_returns.empty:
        return pd.DataFrame(columns=["mu", "sigma", "skew", "kurtosis"])
    stats = pd.DataFrame(
        {
            "mu": asset_returns.mean(),
            "sigma": asset_returns.std(ddof=1),
            "skew": asset_returns.skew(),
            "kurtosis": asset_returns.kurtosis(),
        }
    )
    return stats
