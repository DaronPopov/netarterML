# Historical Data Caching System

## Overview

The historical data caching system provides a robust way to store and retrieve historical financial data, reducing the need for repeated API calls. This improves performance, reduces API usage, and makes the trading strategy execution more reliable.

## Features

- **Persistent Disk-Based Cache**: Stores historical data on disk for reuse across multiple runs
- **Automatic Cache Invalidation**: Cached data expires after a configurable time period (default: 24 hours)
- **Asset Type Segregation**: Separate cache directories for cryptocurrencies and stocks
- **Cache Statistics**: Tracks cache hits, misses, and other metrics
- **Preloading Capability**: Ability to preload data for multiple assets before strategy execution

## Usage

### Command Line Arguments

The following command line arguments have been added to `run_trading_strategy.py`:

```
--preload-data        Preload historical data for all assets before starting
--cache-dir PATH      Directory to store cached data (default: ~/.finlib/cache)
--cache-expiry HOURS  Number of hours before cached data expires (default: 24)
--clear-cache         Clear expired cache entries before starting
```

### Example Usage

```bash
# Run with default cache settings
python -m finlib.finance.run_trading_strategy --assets BTC ETH AAPL MSFT --iterations 10

# Run with preloading and custom cache directory
python -m finlib.finance.run_trading_strategy --assets BTC ETH AAPL MSFT --preload-data --cache-dir /tmp/finlib_cache

# Clear expired cache entries before running
python -m finlib.finance.run_trading_strategy --assets BTC ETH AAPL MSFT --clear-cache
```

## Implementation Details

### Cache Directory Structure

```
~/.finlib/cache/
├── crypto/
│   ├── BTC_USD_1MIN.json
│   ├── ETH_USD_1MIN.json
│   └── ...
└── stocks/
    ├── AAPL_USD_1d.json
    ├── MSFT_USD_1d.json
    └── ...
```

### Cache File Format

Each cache file contains the historical data in JSON format, with the filename pattern:
`{asset_symbol}_{quote_currency}_{period_id}.json`

### Cache Expiry

By default, cached data expires after 24 hours. This can be configured using the `--cache-expiry` command line argument.

## API

### Main Classes and Functions

#### `HistoricalDataCache` Class

The main cache implementation class with methods for storing and retrieving data.

```python
# Get the cache instance
from finlib.finance.data_cache import get_cache_instance
cache = get_cache_instance()

# Get cached data
data = cache.get_cached_data(asset_symbol, quote_currency, period_id, is_crypto)

# Save data to cache
cache.save_to_cache(asset_symbol, quote_currency, period_id, is_crypto, data)

# Preload data
data = cache.preload_data(asset_symbol, quote_currency, period_id, is_crypto, data_fetcher)

# Get cache statistics
stats = cache.get_cache_stats()

# Clear expired cache entries
cleared_count = cache.clear_expired_cache()
```

#### `preload_historical_data` Function

A utility function in `data_provider.py` to preload data for multiple assets.

```python
from finlib.finance.data_provider import preload_historical_data

# Preload data for multiple assets
results = preload_historical_data(assets, quote_currency="USD")
```

## Benefits

1. **Reduced API Calls**: Minimizes the number of API calls to external data providers
2. **Improved Performance**: Faster strategy execution by using cached data
3. **Increased Reliability**: Less dependency on external API availability
4. **Offline Capability**: Can run with cached data when internet connectivity is limited
5. **Reduced Rate Limiting**: Helps avoid hitting API rate limits

## Future Improvements

- Add compression for cached data to reduce disk usage
- Implement a memory cache layer for even faster access
- Add support for more granular cache invalidation based on asset volatility
- Implement automatic cache pruning to limit total disk usage 