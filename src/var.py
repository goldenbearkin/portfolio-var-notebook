from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class VarResult:
    """Container for VaR results."""

    confidence: float
    timeframe: str
    historical_var_abs: float
    historical_var_pct: float
    parametric_var_abs: float | None = None
    parametric_var_pct: float | None = None


def _inv_norm_cdf(p: float) -> float:
    """Acklam's approximation for inverse CDF of standard normal (no scipy)."""
    a = [
        -39.6968302866538,
        220.946098424521,
        -275.928510446969,
        138.357751867269,
        -30.6647980661472,
        2.50662827745924,
    ]
    b = [
        -54.4760987982241,
        161.585836858041,
        -155.698979859887,
        66.8013118877197,
        -13.2806815528857,
    ]
    c = [
        -0.00778489400243029,
        -0.322396458041136,
        -2.40075827716184,
        -2.54973253934373,
        4.37466414146497,
        2.93816398269878,
    ]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if phigh < p:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def historical_var(pnl_series: pd.Series, confidence: float) -> float:
    """Historical VaR as positive currency loss at the given confidence.

    We compute the left-tail quantile q = np.quantile(PnL, 1 - confidence). VaR is -q (positive).
    """
    if pnl_series is None or pnl_series.empty:
        return 0.0
    q = float(np.quantile(pnl_series.values, 1.0 - float(confidence)))
    return float(max(0.0, -q))


def parametric_var_from_series(pnl_series: pd.Series, confidence: float) -> float:
    """Parametric VaR assuming normal distribution of P&L: VaR = z * sigma."""
    if pnl_series is None or pnl_series.empty:
        return 0.0
    sigma = float(pnl_series.std(ddof=1))
    alpha = 1.0 - float(confidence)
    z_left = -_inv_norm_cdf(alpha)
    return float(max(0.0, z_left * sigma))


def contributions_parametric(
    asset_returns: pd.DataFrame,
    exposures: pd.Series,
    portfolio_value: float,
    confidence: float,
) -> pd.Series:
    """Approximate asset contributions to VaR via Euler allocation (variance-covariance model).

    - Compute covariance of asset returns (log returns).
    - Convert currency exposures to value weights w = exposures / portfolio_value.
    - Portfolio sigma: sqrt(w' Σ w). Marginal contribution: (Σ w)_i / sigma.
    - Currency contribution to VaR: z * portfolio_value * w_i * (Σ w)_i / sigma.
    """
    if asset_returns.empty or exposures.empty or portfolio_value == 0:
        return pd.Series(dtype=float)
    common = asset_returns.columns.intersection(exposures.index)
    if common.empty:
        return pd.Series(dtype=float)
    R = asset_returns[common].dropna()
    if R.empty:
        return pd.Series(dtype=float)
    Sigma = R.cov().values
    w = (exposures[common] / float(portfolio_value)).values.reshape(-1, 1)
    sigma_p = float(np.sqrt(np.maximum(1e-18, (w.T @ Sigma @ w)[0, 0])))
    if sigma_p == 0:
        return pd.Series(0.0, index=common)
    alpha = 1.0 - float(confidence)
    z_left = -_inv_norm_cdf(alpha)
    marginal_sigma = (Sigma @ w) / sigma_p  # shape (n,1)
    contrib_currency = (z_left * float(portfolio_value)) * (
        w * marginal_sigma
    )  # elementwise
    contrib = pd.Series(contrib_currency.flatten(), index=common)
    return contrib


def summarize_var(
    pnl_series: pd.Series,
    portfolio_value: float,
    confidence: float,
    timeframe: str,
    asset_returns: pd.DataFrame | None = None,
    positions: pd.Series | None = None,
    latest_prices: pd.Series | None = None,
) -> tuple[VarResult, pd.Series | None]:
    """Compute historical VaR and optional parametric VaR plus contributions.

    If `positions` and `latest_prices` are provided, contributions are computed using value exposures.
    """
    h_var = historical_var(pnl_series, confidence)
    h_pct = (h_var / portfolio_value) if portfolio_value else 0.0
    p_var = (
        parametric_var_from_series(pnl_series, confidence)
        if not pnl_series.empty
        else 0.0
    )
    p_pct = (p_var / portfolio_value) if portfolio_value else 0.0
    result = VarResult(
        confidence=float(confidence),
        timeframe=str(timeframe),
        historical_var_abs=float(h_var),
        historical_var_pct=float(h_pct),
        parametric_var_abs=float(p_var),
        parametric_var_pct=float(p_pct),
    )
    contrib = None
    if (
        asset_returns is not None
        and positions is not None
        and latest_prices is not None
        and portfolio_value
    ):
        exposures = positions.reindex(asset_returns.columns).fillna(
            0.0
        ) * latest_prices.reindex(asset_returns.columns).fillna(0.0)
        contrib = contributions_parametric(
            asset_returns, exposures, portfolio_value, confidence
        )
    return result, contrib
