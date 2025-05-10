import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

class YFinanceAPI:
    def __init__(self):
        self._cache = {}
        self._cache_timeout = 60  # seconds
        self._last_update = {}

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid."""
        if symbol not in self._last_update:
            return False
        return (datetime.now() - self._last_update[symbol]).total_seconds() < self._cache_timeout

    def get_realtime_data(self, asset_names: List[str]) -> Dict[str, Any]:
        """
        Get real-time data for multiple assets.
        
        Args:
            asset_names: List of asset symbols
            
        Returns:
            Dictionary containing real-time data for each asset
        """
        try:
            results = {}
            for symbol in asset_names:
                if self._is_cache_valid(symbol):
                    results[symbol] = self._cache[symbol]
                    continue

                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Extract relevant data
                quote_data = {
                    'price': info.get('regularMarketPrice', 0.0),
                    'change': info.get('regularMarketChange', 0.0),
                    'change_percent': info.get('regularMarketChangePercent', 0.0),
                    'volume': info.get('regularMarketVolume', 0),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Update cache
                self._cache[symbol] = quote_data
                self._last_update[symbol] = datetime.now()
                results[symbol] = quote_data

            return {
                'status': 'success',
                'data': results,
                'message': None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'data': None,
                'message': f'Error fetching real-time data: {str(e)}'
            }

    def get_historical_data(self, symbol: str, period: str = '1mo', interval: str = '1d') -> Dict[str, Any]:
        """
        Get historical data for an asset.
        
        Args:
            symbol: Asset symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            Dictionary containing historical data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            # Convert DataFrame to list of dictionaries
            data = []
            for index, row in df.iterrows():
                data.append({
                    'timestamp': index.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            return {
                'status': 'success',
                'data': data,
                'message': None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'data': None,
                'message': f'Error fetching historical data: {str(e)}'
            }

    def search_stocks(self, query: str) -> Dict[str, Any]:
        """
        Search for stocks matching the query.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Use yfinance's search functionality
            tickers = yf.Tickers(query)
            results = []
            
            for symbol in tickers.tickers:
                info = tickers.tickers[symbol].info
                results.append({
                    'symbol': symbol,
                    'name': info.get('longName', ''),
                    'exchange': info.get('exchange', ''),
                    'type': info.get('quoteType', '')
                })
            
            return {
                'status': 'success',
                'data': results,
                'message': None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'data': None,
                'message': f'Error searching stocks: {str(e)}'
            }

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            stock_info = {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'volume': info.get('volume', 0),
                'average_volume': info.get('averageVolume', 0)
            }
            
            return {
                'status': 'success',
                'data': stock_info,
                'message': None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'data': None,
                'message': f'Error fetching stock info: {str(e)}'
            } 