# Alpaca Paper Trading Integration

This integration allows you to test your cryptocurrency trading signals with Alpaca's paper trading API, which provides a simulated trading environment with no real money at risk.

## Prerequisites

1. **Alpaca Account**: Sign up for a free Alpaca account at [https://app.alpaca.markets/signup](https://app.alpaca.markets/signup)
2. **API Keys**: Generate API keys from your Alpaca dashboard (Paper Trading section)
3. **Python Packages**: Install the required packages:

```bash
pip install alpaca-py pandas numpy
```

## Getting Started

### 1. Set up your Alpaca account

1. Sign up for a free Alpaca account
2. Navigate to the Paper Trading section
3. Generate API keys (Key ID and Secret Key)
4. Note that your paper trading account starts with $100,000 in virtual funds

### 2. Configure the integration

You can either:

- Set environment variables:
  ```bash
  export ALPACA_API_KEY="your_api_key"
  export ALPACA_API_SECRET="your_api_secret"
  ```

- Or pass the keys directly as command-line arguments (see below)

### 3. Run the paper trading system

Basic usage:

```bash
python alpaca_paper_trader.py
```

This will run a single trading cycle using your default settings.

## Command-Line Options

The script supports several command-line options:

```
--api-key TEXT           Alpaca API key (if not provided, will use ALPACA_API_KEY env var)
--api-secret TEXT        Alpaca API secret (if not provided, will use ALPACA_API_SECRET env var)
--max-position-value FLOAT  Maximum value per position in USD (default: 1000.0)
--max-positions INT      Maximum number of positions to hold at once (default: 3)
--symbols TEXT           Specific symbols to trade (e.g., BTCUSD ETHUSD)
--interval INT           Interval between trading cycles in seconds (default: 3600)
--cycles INT             Number of trading cycles to run (default: 1, 0 for infinite)
```

## Examples

### Run with specific API keys:

```bash
python alpaca_paper_trader.py --api-key YOUR_API_KEY --api-secret YOUR_API_SECRET
```

### Trade specific cryptocurrencies:

```bash
python alpaca_paper_trader.py --symbols BTCUSD ETHUSD SOLUSD
```

### Run continuously with 30-minute intervals:

```bash
python alpaca_paper_trader.py --cycles 0 --interval 1800
```

### Allocate more funds per position:

```bash
python alpaca_paper_trader.py --max-position-value 5000 --max-positions 5
```

## How It Works

1. The system fetches historical price data from Alpaca's API
2. It uses your existing signal generation logic to analyze the data
3. Based on the signals, it executes paper trades on your Alpaca account
4. It logs all activities and saves the signals to a JSON file

## Monitoring Your Paper Trades

You can monitor your paper trades in several ways:

1. **Alpaca Dashboard**: Log in to your Alpaca account and view the Paper Trading section
2. **Log Files**: Check the `alpaca_paper_trading.log` file for detailed logs
3. **JSON Output**: Review the `alpaca_trading_signals.json` file for the latest signals

## Supported Cryptocurrencies

The integration supports trading the following cryptocurrencies on Alpaca:

- Bitcoin (BTCUSD)
- Ethereum (ETHUSD)
- Solana (SOLUSD)
- Cardano (ADAUSD)
- Dogecoin (DOGEUSD)
- Polkadot (DOTUSD)
- Avalanche (AVAXUSD)
- Chainlink (LINKUSD)
- Polygon (MATICUSD)
- Litecoin (LTCUSD)
- Uniswap (UNIUSD)
- Stellar (XLMUSD)
- Cosmos (ATOMUSD)
- Algorand (ALGOUSD)
- Tezos (XTZUSD)

## Limitations

- Alpaca's paper trading environment is a simulation and may not perfectly reflect real market conditions
- Paper trading does not account for factors like market impact, slippage, or order queue position
- The integration currently only supports market orders

## Troubleshooting

If you encounter issues:

1. Verify your API keys are correct
2. Check that you're using the paper trading API keys, not live trading keys
3. Review the log file for detailed error messages
4. Ensure you have the latest version of the alpaca-py package 