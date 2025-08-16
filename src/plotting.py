from __future__ import annotations

from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-whitegrid")


def plot_prices(prices: pd.DataFrame, names_map: Optional[dict] = None) -> None:
    """Plot price history per ticker."""
    if prices.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in prices.columns:
        label = names_map.get(t, t) if names_map else t
        ax.plot(prices.index, prices[t], label=label)
    ax.set_title("Adjusted Close Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_portfolio_pnl(pnl: pd.Series) -> None:
    """Plot portfolio cumulative P&L."""
    if pnl is None or pnl.empty:
        return
    cum = pnl.cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cum.index, cum.values, label="Cumulative P&L")
    ax.set_title("Portfolio Cumulative P&L")
    ax.set_xlabel("Date")
    ax.set_ylabel("Currency")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_return_histogram(pnl: pd.Series, var_abs: float, confidence: float) -> None:
    """Plot histogram of portfolio P&L with VaR threshold marked."""
    if pnl is None or pnl.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax.hist(pnl.values, bins=50, color="#69b3a2", alpha=0.7)
    threshold = -var_abs
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Historical VaR @ {confidence:.2%}",
    )
    ax.set_title("Distribution of Portfolio P&L")
    ax.set_xlabel("P&L per period")
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
