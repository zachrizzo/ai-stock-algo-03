# Micro-CTA Algorithmic Trading Strategy

A modular implementation of a micro-CTA (Commodity Trading Advisor) strategy designed for small accounts. This strategy uses a weekly trend gate with volatility targeting to allocate between SPY, TLT, and cash.

## Features

- **Weekly Trend Gate**: Decides between risk-on or risk-off assets every Monday based on 12-week momentum.
- **Volatility Budget**: Scales position size to target approximately 2% weekly portfolio volatility.
- **Crash Insurance**: Automatically adds a sleeve of inverse ETF when S&P 500 falls >6% in a week.
- **Capital Preservation**: Kill-switch to move to cash during significant drawdowns.
- **Cash Yield**: Idle cash is swept into short-term Treasury ETF for yield.

## Project Structure

```
stock_trader_o3_algo/
├── stock_trader_o3_algo/       # Main package
│   ├── core/                   # Core strategy logic
│   ├── data/                   # Data fetching and processing
│   ├── backtest/               # Backtesting engine
│   ├── execution/              # Trade execution with Alpaca API
│   ├── utils/                  # Utility functions
│   └── config/                 # Configuration settings
├── backtest.py                 # Main backtesting script
├── trade.py                    # Main trading script
├── tests/                      # Unit tests
├── backtest_results/           # Directory for backtest results
├── .env                        # Environment variables (not in repo)
├── pyproject.toml              # Poetry dependency management
└── README.md                   # This file
```

## Installation

This project uses Poetry for dependency management.

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stock_trader_o3_algo.git
   cd stock_trader_o3_algo
   ```

2. Install dependencies:
   ```
   poetry install
   ```

3. Create a `.env` file with your Alpaca API credentials (see `.env.example`).

## Usage

### Backtesting

Run a backtest over a specified period:

```bash
poetry run python backtest.py --start 2020-01-01 --end 2023-12-31 --capital 100 --save
```

Run a rolling window analysis to evaluate consistency:

```bash
poetry run python backtest.py --start 2018-01-01 --end 2023-12-31 --window 2 --step 6 --save
```

### Live Trading

Run the strategy in paper trading mode:

```bash
poetry run python trade.py --paper
```

Run in dry-run mode (no actual trades):

```bash
poetry run python trade.py --dry-run
```

Deploy to a cloud environment (AWS Lambda, GitHub Actions, etc.) and schedule to run every Monday at 9:35 AM ET.

## Configuration

Strategy parameters can be modified in `stock_trader_o3_algo/config/settings.py`:

- `WEEKLY_VOL_TARGET`: Target weekly volatility (default: 2%)
- `CRASH_THRESHOLD`: Weekly return threshold for crash protection (default: -6%)
- `HEDGE_WEIGHT`: Weight of hedge position when crash protection is active (default: 15%)
- `KILL_DD`: Drawdown threshold for kill switch (default: 20%)
- `COOLDOWN_WEEKS`: Number of weeks to remain in cash after kill switch (default: 4)

## License

MIT

## Disclaimer

This software is for educational purposes only. Use at your own risk. Past performance is not indicative of future results.
# ai-stock-algo-03
