# Stock Trader O3 Algorithm

Welcome to the documentation for the Stock Trader O3 Algorithm project - a comprehensive suite of algorithmic trading strategies designed for various market conditions.

## Overview

This project provides a collection of advanced trading strategies with different risk profiles, timeframes, and methodologies:

- **[DMT v2](strategies/dmt-v2.md)**: Differentiable Market Twin with neural networks
- **[TurboDMT v2](strategies/turbo-dmt-v2.md)**: Enhanced DMT with market regime detection and dynamic position sizing
- **[Tri-Shot](strategies/tri-shot.md)**: Three-times-weekly strategy with scheduled checkpoints
- **[Turbo QT](strategies/turbo-qt.md)**: Fast rotational strategy focused on momentum

## Key Features

- **Modular Design**: Each strategy is implemented in a modular way for easy customization and extension
- **Unified Backtesting**: Consistent backtesting framework across all strategies
- **Performance Analytics**: Comprehensive performance metrics and visualizations
- **Paper Trading**: Support for paper trading via Binance testnet
- **Regime Detection**: Market regime analysis for adaptive trading

## Getting Started

1. First, check out the [Installation Guide](getting-started/installation.md)
2. Then follow the [Quick Start](getting-started/quick-start.md) tutorial

## Documentation Structure

- **Getting Started**: Installation and basic usage
- **Strategies**: Detailed overview of each trading strategy
- **Modules**: Documentation of the core modules
- **API Reference**: Comprehensive API documentation
- **Examples**: Practical examples and tutorials

## Recent Performance

TurboDMT v2, our flagship strategy, has shown exceptional performance in recent backtests:

| Metric | TurboDMT v2 | Buy & Hold |
|--------|-------------|------------|
| Total Return | 350.99% | 11.21% |
| Sharpe Ratio | 5.77 | 0.89 |
| Max Drawdown | -9.43% | -23.12% |
| Win Rate | 68.2% | N/A |

## License

This project is licensed under the MIT License - see the LICENSE file for details.
