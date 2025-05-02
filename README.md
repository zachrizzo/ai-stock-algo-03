# Algorithmic Trading Strategy Collection

A collection of modular algorithmic trading strategies designed for different market conditions. This repo includes multiple strategies with different risk profiles and trading frequencies.

## Available Strategies

- **Tri-Shot**: A three-times-weekly strategy with Monday (directional view), Wednesday (volatility rebalancing), and Friday (risk gate) checkpoints.
- **DMT (Differentiable Market Twin)**: A deep learning approach using PyTorch for differentiable strategy optimization.
- **TurboDMT v2**: Enhanced DMT strategy with market regime detection and dynamic position sizing, achieving significantly improved risk-adjusted returns.
- **Turbo QT**: An aggressive rotational strategy using technical indicators and volatility-based sizing.

## Project Structure

```shell
ai-stock-algo-03/
├── scripts/                    # Command-line tools and utilities
│   ├── trade.py                # Unified CLI for all strategies
│   ├── paper_trade.py          # Paper trading implementation (Binance)
│   └── run_tests.py            # Test runner script
│
├── stock_trader_o3_algo/       # Main package
│   ├── strategies/             # Strategy implementations
│   │   ├── dmt_v2_strategy.py  # Core DMT v2 implementation
│   │   ├── market_regime.py    # Market regime detection
│   │   ├── tri_shot/           # Tri-Shot strategy
│   │   ├── dmt/                # Original DMT strategy
│   │   ├── dmt_v2/             # DMT v2 components
│   │   ├── turbo_dmt_v2/       # TurboDMT v2 enhancements
│   │   └── turbo_qt/           # Turbo QT rotational strategy
│   │
│   ├── backtester/             # Unified backtesting framework
│   │   ├── core.py             # Backtester base classes
│   │   ├── performance.py      # Performance metrics
│   │   └── visualization.py    # Plotting and visualization
│   │
│   ├── data_utils/             # Data utilities
│   │   └── market_simulator.py # Market data simulation
│   │
│   ├── execution/              # Trade execution
│   └── utils/                  # Utility functions
│
├── tests/                      # Test suite
│   ├── test_dmt_v2.py          # DMT v2 tests
│   ├── test_turbo_dmt_v2.py    # TurboDMT v2 tests
│   └── ...                     # Other test modules
│
├── data/                       # Version-controlled sample data
├── data_cache/                 # Cached market data (gitignored)
└── backtest_results/           # Backtest outputs and visualizations
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-stock-algo-03.git
cd ai-stock-algo-03

# Install dependencies
pip install -e .
```

### Running Backtests

The new unified CLI makes it easy to run any strategy:

```bash
# Run TurboDMT v2 backtest on SPY
python scripts/trade.py turbo_dmt_v2 backtest --symbol SPY --start-date 2023-01-01

# Run enhanced DMT v2 backtest on QQQ with custom parameters
python scripts/trade.py enhanced_dmt_v2 backtest --symbol QQQ --capital 100000 --lookback 126

# Compare multiple strategies
python scripts/trade.py compare --strategies dmt_v2,enhanced_dmt_v2,turbo_dmt_v2 --symbol SPY
```

### Paper Trading

For live paper trading, you can use the Binance testnet support:

```bash
# Set up your API keys first
export BINANCE_API_KEY_TEST="your_testnet_api_key"
export BINANCE_API_SECRET_TEST="your_testnet_api_secret"

# Run TurboDMT v2 on Bitcoin/USDT
python scripts/paper_trade.py --symbol BTCUSDT --interval 1d --version turbo
```

### Legacy Command Support

For backwards compatibility, the original command structure is maintained:

```bash
# These commands will use the new unified system behind the scenes
./dmt backtest --symbol SPY
./tri_shot run
./turbo_qt backtest
```

## Strategy Details

### TurboDMT v2

The TurboDMT v2 strategy extends the original DMT (Differentiable Market Twin) with several key enhancements:

1. **Market Regime Detection**: Identifies Bull, Bear, and Neutral regimes using multiple technical indicators
2. **Dynamic Position Sizing**: Adapts position sizes based on current market regime and volatility
3. **Improved Signal Generation**: Enhanced transformer-based model with 96 dimensions, 6 attention heads, and 5 layers
4. **Risk Management**: Intelligent drawdown control and volatility targeting

In backtests, TurboDMT v2 has significantly outperformed both Buy & Hold and the original DMT strategy, with a Sharpe ratio of 5.77 and returns of 350.99% over the test period.

### DMT (Differentiable Market Twin)

DMT uses PyTorch-based neural networks to create a differentiable strategy that can be optimized end-to-end. The strategy employs:

1. **Feature Engineering**: Technical indicators and price patterns
2. **Model Architecture**: Hybrid LSTM-Transformer models
3. **Ensemble Methods**: Multiple models combined for robust predictions

### Tri-Shot

A structured weekly approach with three checkpoints:
- **Monday**: Establish directional view for the week
- **Wednesday**: Volatility-based position rebalancing
- **Friday**: Risk assessment and weekend position sizing

### Turbo QT

A high-frequency rotational strategy focused on the QQQ ETF and its components, using:
1. **Technical Indicators**: RSI, MACD, and momentum signals
2. **Volatility-Based Sizing**: Position sizing based on current market volatility
3. **Rotational Logic**: Selecting highest momentum components

## Documentation

For detailed documentation on each strategy and the API reference, see the [full documentation](./docs/).

## License

MIT
