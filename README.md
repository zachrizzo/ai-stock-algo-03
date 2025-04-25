# Algorithmic Trading Strategy Collection

A collection of modular algorithmic trading strategies designed for different market conditions. This repo includes multiple strategies with different risk profiles and trading frequencies.

## Available Strategies

- **Tri-Shot**: A three-times-weekly strategy with Monday (directional view), Wednesday (volatility rebalancing), and Friday (risk gate) checkpoints.
- **DMT (Differentiable Market Twin)**: A deep learning approach using PyTorch for differentiable strategy optimization.
- **Turbo QT**: An aggressive rotational strategy using technical indicators and volatility-based sizing.

## Project Structure

```shell
stock_trader_o3_algo/
├── bin/                        # Command-line scripts
│   ├── trader_cli.py           # Unified CLI for all strategies
│   ├── cli.py                  # Legacy CLI (for backward compatibility)
│   └── tri_shot_cli.py         # Tri-Shot specific CLI
├── stock_trader_o3_algo/       # Main package
│   ├── strategies/             # Strategy implementations
│   │   ├── tri_shot/           # Tri-Shot strategy
│   │   ├── dmt/                # Differentiable Market Twin strategy
│   │   └── turbo_qt/           # Turbo QT rotational strategy
│   ├── backtest/               # Backtesting engines
│   ├── data/                   # Data fetching and processing
│   ├── utils/                  # Utility functions
│   ├── execution/              # Trade execution with Alpaca API
│   └── config/                 # Configuration settings
├── data/                       # Cached data files
├── tri_shot_data/              # Tri-Shot state and model files
├── backtest_results/           # Directory for backtest results
├── tri_shot                    # Shortcut script for Tri-Shot strategy
├── dmt                         # Shortcut script for DMT strategy
├── turbo_qt                    # Shortcut script for Turbo QT strategy
├── trader                      # Unified CLI shortcut
├── .env                        # Environment variables (not in repo)
├── pyproject.toml              # Poetry dependency management
└── README.md                   # This file
```

## Installation

This project uses Poetry for dependency management.

1. Clone the repository:

```shell
git clone https://github.com/yourusername/stock_trader_o3_algo.git
cd stock_trader_o3_algo
```

2. Install dependencies:

```shell
poetry install
```

3. Create a `.env` file with your Alpaca API credentials (see `.env.example`).

## Usage

The project includes convenient shortcut scripts for each strategy:

### Tri-Shot Strategy

```shell
# Run strategy based on current day
./tri_shot run

# Run a backtest
./tri_shot backtest --days 365 --capital 1000 --plot

# Train model
./tri_shot train --force

# Set up paper trading
./tri_shot paper --capital 1000
```

### DMT Strategy

```shell
# Train the model
./dmt train --epochs 50 --lookback 30

# Run a backtest
./dmt backtest --days 365 --capital 10000 --plot
```

### Turbo QT Strategy

```shell
# Run rebalancing
./turbo_qt rebalance

# Check if stops are hit
./turbo_qt check_stops

# Run a backtest
./turbo_qt backtest --days 365 --capital 10000 --plot
```

### Unified CLI

You can also use the unified CLI for any strategy:

```shell
# General format
./trader [strategy] [command] [options]

# Examples
./trader tri_shot run
./trader dmt backtest --days 365
./trader turbo_qt rebalance
```

## Configuration

Strategy parameters are configured in their respective modules.

## License

MIT

## Disclaimer

This software is for educational purposes only. Use at your own risk. Past performance is not indicative of future results.
