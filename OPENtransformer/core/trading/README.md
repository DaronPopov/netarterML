# Cryptocurrency Trading Signal Generator

This tool generates trading signals for cryptocurrencies based on technical indicators and risk metrics. It uses optimized assembly kernels from the `finlib` library for fast computation.

## Features

- Fetches cryptocurrency data from CoinGecko API
- Analyzes price data using technical indicators:
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
- Calculates risk metrics:
  - Value at Risk (VaR)
  - Expected Shortfall (ES)
  - Black-Scholes VaR
- Generates trading signals (BUY, SELL, HOLD) based on combined indicators
- Supports caching of historical data to avoid API rate limits
- Live mode for real-time updates without requerying historical data
- Optimized for CoinGecko's free API tier with rate limiting protection

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the `finlib` library is installed and properly configured

## Usage

### Basic Usage

To generate trading signals for the top 3 cryptocurrencies:

```bash
python run_trading_signals.py
```

This will:
1. Fetch the top 3 cryptocurrencies by market cap
2. Get historical data for the past 30 days
3. Generate trading signals
4. Save the results to `crypto_signals.json`

### Command Line Options

```
usage: run_trading_signals.py [-h] [--num-coins NUM_COINS] [--days DAYS]
                             [--output OUTPUT] [--confidence CONFIDENCE]
                             [--simulations SIMULATIONS]
                             [--specific-coins SPECIFIC_COINS [SPECIFIC_COINS ...]]
                             [--max-retries MAX_RETRIES]
                             [--retry-delay RETRY_DELAY] [--test]
                             [--force-refresh] [--live]
                             [--cache-dir CACHE_DIR] [--cache-ttl CACHE_TTL]

Cryptocurrency Trading Signal Generator

optional arguments:
  -h, --help            show this help message and exit
  --num-coins NUM_COINS
                        Number of top cryptocurrencies to analyze (default: 3)
  --days DAYS           Number of days of historical data to fetch (default: 30)
  --output OUTPUT       Output JSON file path (default: crypto_signals.json)
  --confidence CONFIDENCE
                        Confidence level for risk metrics (default: 0.95)
  --simulations SIMULATIONS
                        Number of Monte Carlo simulations for Black-Scholes VaR (default: 10000)
  --specific-coins SPECIFIC_COINS [SPECIFIC_COINS ...]
                        Specific cryptocurrencies to analyze (e.g., bitcoin ethereum)
  --max-retries MAX_RETRIES
                        Maximum number of retries for API requests (default: 5)
  --retry-delay RETRY_DELAY
                        Delay in seconds between retries (default: 120)
  --test                Run in test mode with synthetic data
  --force-refresh       Force refresh of historical data (ignore cache)
  --live                Use live data updates instead of full historical data fetch
  --cache-dir CACHE_DIR
                        Directory to store cached data (default: ./data_cache)
  --cache-ttl CACHE_TTL
                        Cache time-to-live in hours (default: 24)
```

### Examples

#### Analyze specific cryptocurrencies:

```bash
python run_trading_signals.py --specific-coins bitcoin ethereum
```

#### Use more historical data:

```bash
python run_trading_signals.py --days 90
```

#### Run in test mode with synthetic data:

```bash
python run_trading_signals.py --test
```

### Live Mode

The live mode uses cached historical data and only fetches the latest price updates, which is much more efficient and avoids API rate limits. This is useful for running the tool periodically to get updated signals.

#### Run once in live mode:

```bash
python run_trading_signals.py --live
```

#### Run periodically using the live signals script:

```bash
python run_live_signals.py --interval 15
```

This will run the signal generator every 15 minutes, using cached data and only fetching the latest updates.

#### Live signals options:

```
usage: run_live_signals.py [-h] [--num-coins NUM_COINS] [--output OUTPUT]
                          [--specific-coins SPECIFIC_COINS [SPECIFIC_COINS ...]]
                          [--cache-dir CACHE_DIR] [--interval INTERVAL]
                          [--cache-ttl CACHE_TTL] [--simulations SIMULATIONS]

Live Cryptocurrency Trading Signal Generator

optional arguments:
  -h, --help            show this help message and exit
  --num-coins NUM_COINS
                        Number of top cryptocurrencies to analyze (default: 3)
  --output OUTPUT       Output JSON file path (default: live_signals.json)
  --specific-coins SPECIFIC_COINS [SPECIFIC_COINS ...]
                        Specific cryptocurrencies to analyze (e.g., bitcoin ethereum)
  --cache-dir CACHE_DIR
                        Directory to store cached data (default: ./data_cache)
  --interval INTERVAL   Interval in minutes to run the signal generator (0 for once, minimum 15)
  --cache-ttl CACHE_TTL
                        Cache time-to-live in hours (default: 24)
  --simulations SIMULATIONS
                        Number of Monte Carlo simulations for Black-Scholes VaR (default: 10000)
```

## Caching

The tool uses a caching mechanism to avoid hitting API rate limits. Historical data is cached in the `data_cache` directory (configurable with `--cache-dir`). The cache includes:

- Historical price data for each cryptocurrency
- Black-Scholes VaR results
- Timestamps for when each piece of data was last updated
- Top coins list (cached for 6 hours)
- Latest price data (cached for 15 minutes)

By default, cached data is used if it's less than 24 hours old (configurable with `--cache-ttl`). You can force a refresh of the data with the `--force-refresh` flag.

## API Rate Limiting

This tool is designed to work with CoinGecko's free API tier, which has rate limits. To avoid hitting these limits, the tool:

1. Limits the number of coins analyzed to 3 by default
2. Implements extensive caching for all API responses
3. Uses exponential backoff for retries when rate limited
4. Enforces a minimum interval between API requests (1.5 seconds)
5. Limits the live mode update interval to a minimum of 15 minutes
6. Reduces the number of Monte Carlo simulations to 10,000 by default

## Output Format

The tool generates a JSON file with the following structure:

```json
{
  "timestamp": "2025-03-13T08:39:05.929989",
  "signals": {
    "bitcoin": {
      "price": 65432.12,
      "rsi": 68.5,
      "macd": 123.45,
      "signal_line": 120.67,
      "upper_band": 66000.0,
      "middle_band": 65000.0,
      "lower_band": 64000.0,
      "var_95": 0.03,
      "es_95": 0.04,
      "bs_var_95": 0.035,
      "rsi_signal": "HOLD",
      "macd_signal": "BUY",
      "bb_signal": "HOLD",
      "risk_signal": "HOLD",
      "final_signal": "HOLD",
      "confidence": 0.75
    },
    // More cryptocurrencies...
  },
  "metadata": {
    "num_coins": 3,
    "days_analyzed": 30,
    "confidence_level": 0.95,
    "simulations": 10000,
    "test_mode": false,
    "live_mode": true,
    "force_refresh": false,
    "cache_ttl": 24
  }
}
```

## Setting Up as a Service

To run the live signal generator as a service, you can use cron (Linux/macOS) or Task Scheduler (Windows).

### Using cron (Linux/macOS):

```bash
# Edit crontab
crontab -e

# Add this line to run every 15 minutes
*/15 * * * * cd /path/to/project && python trading/run_live_signals.py --specific-coins bitcoin ethereum
```

### Using systemd (Linux):

Create a service file:

```bash
sudo nano /etc/systemd/system/crypto-signals.service
```

Add the following content:

```
[Unit]
Description=Cryptocurrency Trading Signal Generator
After=network.target

[Service]
User=yourusername
WorkingDirectory=/path/to/project
ExecStart=/usr/bin/python trading/run_live_signals.py --interval 15 --specific-coins bitcoin ethereum
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable crypto-signals
sudo systemctl start crypto-signals
```

## Troubleshooting

### API Rate Limits

If you're still hitting API rate limits, try:

1. Reducing the number of coins you're analyzing (use `--specific-coins` with just 1-2 coins)
2. Increasing the `--cache-ttl` parameter to use cached data for longer
3. Increasing the `--interval` parameter to run less frequently (30 minutes or more)
4. Running in test mode (`--test`) to use synthetic data for development/testing

### Missing Data

If you're missing data for some coins, check:

1. That the coin IDs are correct (they should match the CoinGecko API IDs)
2. The log file for any API errors
3. Try running with `--force-refresh` to ignore the cache

## License

This project is licensed under the MIT License - see the LICENSE file for details. 