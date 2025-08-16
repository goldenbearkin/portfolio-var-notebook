# Portfolio VaR Notebook

An interactive Jupyter Notebook application to calculate portfolio Value at Risk (VaR) using historical prices from yfinance. Configure tickers, positions, confidence level, and return frequency, then view tables and charts.

## Prerequisites

- Python 3.10–3.12 (3.11 recommended)
- uv installed (`https://github.com/astral-sh/uv`)

## Setup (quickstart)

```bash
# Inside project root
uv python install 3.11
uv sync
```

This creates a virtual env and installs dependencies declared in `pyproject.toml`.

Alternatively, editable install style:

```bash
uv python install 3.11
uv venv
uv pip install -e .
```

## Run the notebook

```bash
uv run jupyter notebook | cat
```

Then open `portfolio_var.ipynb`. Follow the ordered sections:

- Setup → Portfolio → Data → Returns → VaR → Contributions → Plots → Summary

## Usage

- Edit the configuration cell to adjust:
  - Portfolio tickers, friendly names, and positions (positive long, negative short) in units
  - Confidence level (e.g. 0.95, 0.975, 0.99) within [0.90, 0.9999]
  - Timeframe: "daily", "weekly", or "monthly"
  - Start/end dates
- Re-run cells below to refresh tables and charts.

## Methodology

- Prices: Adjusted close via yfinance; tickers fetched in a single call when possible
- Alignment: All series aligned on common dates; assets with insufficient history are dropped with a warning
- Returns: Log returns; daily uses business-day resampling; weekly/monthly use period-end resampling
- Portfolio: Positions interpreted as units; current market value uses latest available prices
- Historical VaR: Left-tail quantile of portfolio return distribution at level (1 - confidence). Reported as positive loss in currency and percent of portfolio value.
- Contributions: Approximated using Euler allocation from the variance-covariance model applied to the historical VaR magnitude (documented in code). Optional parametric VaR also computed for comparison.

## Troubleshooting

- Ticker errors or delistings: The data fetch step will warn and drop tickers with no/insufficient history
- Too few observations: Increase lookback window or switch to daily frequency
- Empty results: Verify tickers and date range; ensure internet access for yfinance

## Project Layout

```
project-root/
  README.md
  pyproject.toml
  portfolio_var.ipynb
  /src/
    __init__.py
    data.py
    portfolio.py
    returns.py
    var.py
    plotting.py
  /data/           # optional cache
```

## License

MIT
