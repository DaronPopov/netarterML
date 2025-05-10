#!/usr/bin/env python
"""
Test script to verify that the Alpaca imports work correctly.
"""

print("Testing Alpaca imports...")

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    
    print("✅ All Alpaca imports successful!")
    print("The following modules were imported successfully:")
    print("- alpaca.trading.client.TradingClient")
    print("- alpaca.trading.requests.MarketOrderRequest, LimitOrderRequest")
    print("- alpaca.trading.enums.OrderSide, TimeInForce")
    print("- alpaca.data.historical.CryptoHistoricalDataClient")
    print("- alpaca.data.requests.CryptoBarsRequest")
    print("- alpaca.data.timeframe.TimeFrame")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure you have installed the alpaca-py package:")
    print("pip install alpaca-py") 