from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class Asset:
    """Represents a single portfolio asset.

    Attributes
    ----------
    ticker: str
            Ticker symbol resolvable by yfinance (e.g., "AAPL").
    name: str
            Friendly display name.
    position: float
            Position size in units (positive for long, negative for short).
    """

    ticker: str
    name: str
    position: float


class Portfolio:
    """A simple portfolio model holding assets and providing helpers.

    Parameters
    ----------
    assets:
            List of `Asset` instances.
    """

    def __init__(self, assets: Sequence[Asset]):
        self.assets: List[Asset] = list(assets)

    def tickers(self) -> List[str]:
        """Return tickers for all assets."""
        return [a.ticker for a in self.assets]

    def positions_series(self) -> pd.Series:
        """Return a Series of positions indexed by ticker."""
        return pd.Series({a.ticker: a.position for a in self.assets}, dtype=float)

    def names_map(self) -> Dict[str, str]:
        """Map ticker to friendly name."""
        return {a.ticker: a.name for a in self.assets}

    def holdings_table(self, latest_prices: Optional[pd.Series] = None) -> pd.DataFrame:
        """Return a holdings summary table.

        If `latest_prices` is provided, includes market value in currency.
        """
        data = []
        for a in self.assets:
            row: Dict[str, object] = {
                "Ticker": a.ticker,
                "Name": a.name,
                "Position": a.position,
            }
            if latest_prices is not None and a.ticker in latest_prices.index:
                px = latest_prices[a.ticker]
                row["Last Price"] = float(px)
                row["Market Value"] = float(px * a.position)
            data.append(row)
        df = pd.DataFrame(data)
        if "Market Value" in df.columns:
            df["Weight"] = df["Market Value"] / df["Market Value"].sum()
        return df

    def current_weights(self, latest_prices: pd.Series) -> pd.Series:
        """Compute current weights by market value given latest prices.

        Returns
        -------
        pd.Series
                Weights per ticker summing to 1.0. If total market value is 0, returns zeros.
        """
        positions = self.positions_series()
        common = positions.index.intersection(latest_prices.index)
        if common.empty:
            return pd.Series(dtype=float)
        market_values = positions.loc[common] * latest_prices.loc[common]
        total = float(market_values.sum())
        if total == 0:
            return market_values * 0.0
        return market_values / total

    def current_value(self, latest_prices: pd.Series) -> float:
        """Compute current total market value of the portfolio given latest prices."""
        positions = self.positions_series()
        common = positions.index.intersection(latest_prices.index)
        if common.empty:
            return 0.0
        return float((positions.loc[common] * latest_prices.loc[common]).sum())

    def recreate_snippet(self) -> str:
        """Return a Python code snippet to recreate this portfolio."""
        lines = [
            "from src.portfolio import Asset, Portfolio",
            "portfolio = Portfolio([",
        ]
        for a in self.assets:
            lines.append(
                f'    Asset(ticker="{a.ticker}", name="{a.name}", position={a.position}),'
            )
        lines.append("])")
        return "\n".join(lines)


def validate_confidence_level(confidence: float) -> None:
    """Validate that the confidence level is within [0.90, 0.9999]."""
    if not (0.90 <= float(confidence) <= 0.9999):
        raise ValueError(
            f"Invalid confidence level {confidence}. Must be between 0.90 and 0.9999."
        )


def validate_timeframe(timeframe: str) -> str:
    """Normalize and validate timeframe string (daily/weekly/monthly)."""
    t = timeframe.strip().lower()
    valid = {"daily", "weekly", "monthly"}
    if t not in valid:
        raise ValueError(
            f"Invalid timeframe '{timeframe}'. Choose one of {sorted(valid)}"
        )
    return t
