from finlib.APIS.stocks_api import StocksAPI
import yfinance as yf
import numpy as np
import datetime
import pandas as pd
from collections import Counter

tickers = ["PRUAX", "CHTTX", "MBDFX", "ICBMX"]

def analyze_ticker(ticker):
    # Define time range for fetching data
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=5 * 365)  # 5 years

    # Fetch historical data using yf.download()
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    # Check if data was fetched successfully
    if data.empty:
        print(f"Error: No data found for ticker {ticker}. Skipping analysis.")
        return {
            "ticker": ticker,
            "error": "No data found."
        }

    # Instantiate StocksAPI without passing data
    api = StocksAPI()
    
    # Convert data to numpy array format for StocksAPI methods
    prices = data['Close'].values.reshape(-1, 1).astype(np.float32)  # Reshape for (days, assets) format

    # Calculate stock indicators
    moving_average = api.moving_average(prices, window=50)
    relative_strength_index = api.relative_strength_index(prices, window=14)
    macd_line, signal_line, histogram = api.macd(prices, fast_period=12, slow_period=26, signal_period=9)
    
    # Convert results back to pandas DataFrames for easier handling
    moving_average_df = pd.DataFrame(moving_average, index=data.index, columns=['MA'])
    rsi_df = pd.DataFrame(relative_strength_index, index=data.index, columns=['RSI'])
    macd_df = pd.DataFrame(macd_line, index=data.index, columns=['MACD'])
    macd_signal_df = pd.DataFrame(signal_line, index=data.index, columns=['Signal'])
    
    # For Bollinger Bands, we need to implement this separately as it's not shown in the StocksAPI
    # This is a placeholder - you may need to adjust this based on actual implementation
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        # Calculate rolling mean and standard deviation
        rolling_mean = np.zeros_like(prices)
        rolling_std = np.zeros_like(prices)
        
        for i in range(window - 1, len(prices)):
            window_slice = prices[i-window+1:i+1]
            rolling_mean[i] = np.mean(window_slice)
            rolling_std[i] = np.std(window_slice)
            
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band, lower_band
    
    upper_band, lower_band = calculate_bollinger_bands(prices, window=20, num_std=2)
    upper_band_df = pd.DataFrame(upper_band, index=data.index, columns=['Upper'])
    lower_band_df = pd.DataFrame(lower_band, index=data.index, columns=['Lower'])

    # Generate buy/sell/hold signals based on moving average crossover
    signals = []
    for i in range(1, len(data)):
        try:
            if i < len(moving_average_df) and i < len(rsi_df) and i < len(macd_df) and i < len(upper_band_df):
                ma_current = float(moving_average_df['MA'].iloc[i])
                ma_previous = float(moving_average_df['MA'].iloc[i - 1])
                close_current = float(data['Close'].iloc[i])
                close_previous = float(data['Close'].iloc[i - 1])

                if (not np.isnan(ma_current) and not np.isnan(ma_previous) and 
                    not np.isnan(close_current) and not np.isnan(close_previous)):
                    if ma_current > close_current and ma_previous <= close_previous:
                        signals.append('Buy')
                    elif ma_current < close_current and ma_previous >= close_previous:
                        signals.append('Sell')
                    else:
                        signals.append('Hold')
                else:
                    signals.append('Hold')  # Handle NaN values
            else:
                signals.append('Hold')  # Default to hold if data is missing
        except (ValueError, TypeError):
            signals.append('Hold')

    # Create a DataFrame for signals
    signal_df = pd.DataFrame(signals, index=data.index[1:], columns=['Signal'])

    # Count the buy/sell/hold signals
    signal_counts = Counter(signals)
    buy_count = signal_counts['Buy']
    sell_count = signal_counts['Sell']
    hold_count = signal_counts['Hold']

    # Get the latest signal
    latest_signal = signals[-1] if signals else "Hold"

    recommendation_reasoning = {
        "Buy": "The 50-day moving average crossed above the closing price, indicating a potential upward trend.",
        "Sell": "The 50-day moving average crossed below the closing price, indicating a potential downward trend.",
        "Hold": "The moving average and closing price did not cross, resulting in a hold recommendation."
    }.get(latest_signal, "Hold recommendation due to lack of sufficient data.")

    return {
        "ticker": ticker,
        "moving_average": moving_average_df.to_string(),
        "relative_strength_index": rsi_df.to_string(),
        "macd": macd_df.to_string(),
        "bollinger_bands": {"upper": upper_band_df.to_string(), "lower": lower_band_df.to_string()},
        "signals": signals,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "latest_signal": latest_signal,
        "recommendation_reasoning": recommendation_reasoning
    }

if __name__ == '__main__':
    for ticker in tickers:
        analysis_results = analyze_ticker(ticker)
        print(analysis_results)