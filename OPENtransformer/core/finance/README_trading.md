# Real-Time Trading Execution Strategy

This project implements a real-time trading execution strategy for cryptocurrency assets using technical indicators and risk management techniques.

## Features

### Technical Analysis
- **RSI (Relative Strength Index)**: Momentum oscillator that measures the speed and change of price movements
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **Bollinger Bands**: Volatility bands placed above and below a moving average

### Risk Management
- **Value at Risk (VaR)**: Statistical measure of potential loss
- **Expected Shortfall (ES)**: Average of losses beyond VaR
- **Stop Loss**: Automatic position closure to limit losses
- **Take Profit**: Automatic position closure to secure profits
- **Position Sizing**: Dynamic position sizing based on capital percentage

### Performance Metrics
- Win rate calculation
- Total return tracking
- Execution time benchmarking
- Equity curve visualization

## Components

- **trading_strategy.py**: Main trading strategy implementation
- **benchmark_strategy.py**: Benchmarking and comparison tools
- **run_trading_strategy.py**: Command-line interface to run the strategy

## Usage

Run the trading strategy with default parameters:
```bash
python run_trading_strategy.py
```

Run with specific assets and parameters:
```bash
python run_trading_strategy.py --mode run --assets BTC ETH --iterations 5 --sleep 10
```

Run benchmarks:
```bash
python run_trading_strategy.py --mode benchmark --assets BTC ETH --iterations 3
```

Compare different strategy parameters:
```bash
python run_trading_strategy.py --mode compare --assets BTC --iterations 2
```

Benchmark execution speed:
```bash
python run_trading_strategy.py --mode speed --assets BTC ETH
```

## Command-line Options

- `--mode`: Mode to run (run, benchmark, compare, speed)
- `--assets`: List of assets to trade
- `--iterations`: Number of iterations to run
- `--sleep`: Sleep time between iterations in seconds
- `--capital`: Initial capital
- `--position-size`: Position size as percentage of capital
- `--stop-loss`: Stop loss percentage
- `--take-profit`: Take profit percentage
- `--max-positions`: Maximum number of concurrent positions

## Implementation Details

The strategy uses the StocksAPI from the finlib library, which provides optimized implementations of financial calculations using assembly kernels. The real-time data is fetched from CoinAPI.

### Signal Generation

The strategy combines signals from multiple technical indicators:
1. RSI: Buy when < 30, Sell when > 70
2. MACD: Buy when MACD line crosses above signal line, Sell when below
3. Bollinger Bands: Buy when price is near lower band, Sell when near upper band

The final signal is determined by majority voting among the indicators.

### Risk Management

The strategy implements risk management through:
1. Position sizing based on a percentage of available capital
2. Stop loss orders to limit potential losses
3. Take profit orders to secure gains
4. Maximum number of concurrent positions
5. VaR and ES calculations to monitor portfolio risk

## Performance Benchmarking

The strategy includes tools to benchmark:
1. Execution time of different components
2. Comparison of different parameter sets (Conservative, Balanced, Aggressive)
3. Overall strategy performance metrics

## Visualization

The strategy generates:
1. Equity curve plots
2. Strategy comparison plots

## Dependencies

- NumPy
- Pandas
- Matplotlib
- Requests
- finlib (internal library with optimized financial calculations)

## Future Improvements

- Add more technical indicators (Stochastic, ADX, etc.)
- Implement portfolio optimization
- Add machine learning-based signal generation
- Implement backtesting functionality
- Add support for more asset classes 