from finlib.APIS.stocks_api import StocksAPI
import yfinance as yf
import numpy as np
import time
import datetime
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


tickers = ["PRUAX", "CHTTX", "MBDFX", "ICBMX"]

def analyze_ticker(ticker):
    # Fetch historical data for the current ticker from yfinance
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=5*365) # 5 years
    data = yf.download(ticker, start=start_date, end=end_date)

    # Instantiate StocksAPI with the historical data
    api = StocksAPI()

    # Calculate all the stock analysis functions
    moving_average = api.moving_average(data, window=50)
    relative_strength_index = api.relative_strength_index(data, window=14)
    macd_df, macd_signal_df = api.macd(data, fast_period=12, slow_period=26, signal_period=9)
    upper_band, lower_band = api.bollinger_bands(data, window=20, num_std=2)

    # Generate buy/sell/hold signals based on moving average crossover
    signals = []
    for i in range(1, len(data)):
        try:
            if i < len(moving_average) and i < len(relative_strength_index) and i < len(macd_df) and i < len(upper_band):
                # Extract scalar values
                ma_current = float(moving_average['MA'].iloc[i])
                ma_previous = float(moving_average['MA'].iloc[i-1])
                close_current = float(data['Close'].iloc[i])
                close_previous = float(data['Close'].iloc[i-1])
                
                # Check for NaN values using Python's math.isnan
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
                signals.append('Hold') # Default to hold if data is not available
        except (ValueError, TypeError):
            signals.append('Hold')  # Handle any extraction errors

    # Create a DataFrame for signals
    signal_df = pd.DataFrame(signals, index=data.index[1:], columns=['Signal'])

    # Count the buy/sell/hold signals
    signal_counts = Counter(signals)
    buy_count = signal_counts['Buy']
    sell_count = signal_counts['Sell']
    hold_count = signal_counts['Hold']

    # Current recommendation based on most recent data
    latest_signal = signals[-1] if len(signals) > 0 else "Hold"
    
    recommendation_reasoning = ""
    
    # Create subplots for visualization
    fig, axs = plt.subplots(4, 1, figsize=(12, 18))
    
    if latest_signal == 'Buy':
        # Plot 1: Price and Moving Average
        axs[0].plot(data.index, data['Close'], label='Close Price', color='blue')
        axs[0].plot(moving_average.index, moving_average['MA'], label='50-day MA', color='red')
        axs[0].set_title(f'{ticker} Stock Price and Moving Average')
        axs[0].set_ylabel('Price')
        axs[0].legend()
        axs[0].grid(True)

    # Plot 2: RSI
    axs[1].plot(relative_strength_index.index, relative_strength_index['RSI'], color='purple')
    axs[1].axhline(70, color='red', linestyle='--')  # Overbought line
    axs[1].axhline(30, color='green', linestyle='--')  # Oversold line
    axs[1].set_title('Relative Strength Index (RSI)')
    axs[1].set_ylabel('RSI')
    axs[1].grid(True)

    # Plot 3: MACD
    axs[2].plot(macd_df.index, macd_df['MACD'], label='MACD', color='blue')
    axs[2].plot(macd_signal_df.index, macd_signal_df['Signal'], label='Signal Line', color='red')
    axs[2].set_title('Moving Average Convergence Divergence (MACD)')
    axs[2].set_ylabel('MACD')
    axs[2].legend()
    axs[2].grid(True)

    # Plot 4: Bollinger Bands
    axs[3].plot(data.index, data['Close'], label='Close Price', color='blue')
    axs[3].plot(upper_band.index, upper_band['Upper'], label='Upper Band', color='red')
    axs[3].plot(lower_band.index, lower_band['Lower'], label='Lower Band', color='green')
    axs[3].set_title('Bollinger Bands')
    axs[3].set_ylabel('Price')
    axs[3].set_xlabel('Date')
    axs[3].legend()
    axs[3].grid(True)

    # Mark buy and sell signals on the price chart
    for idx, row in signal_df.iterrows():
        if row['Signal'] == 'Buy':
            axs[0].scatter(idx, data.loc[idx]['Close'], color='green', s=100, marker='^')
        elif row['Signal'] == 'Sell':
            axs[0].scatter(idx, data.loc[idx]['Close'], color='red', s=100, marker='v')

    # Format the date on the x-axis
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3 months
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(f'{ticker}_analysis.png')
    plt.show()

    # Summary
    print("\nSIGNAL SUMMARY:")
    print(f"Total periods analyzed: {len(signals)}")
    print(f"Buy signals: {buy_count} ({buy_count/len(signals)*100:.1f}%)")
    print(f"Sell signals: {sell_count} ({sell_count/len(signals)*100:.1f}%)")
    print(f"Hold signals: {hold_count} ({hold_count/len(signals)*100:.1f}%)")

    # Current recommendation based on most recent data
    if len(signals) > 0:
        latest_signal = signals[-1]
        print(f"\nCURRENT RECOMMENDATION for {ticker}: {latest_signal.upper()}")

        # Explain the recommendation
        print("Reasoning:")
        if latest_signal == 'Buy':
            print("The 50-day moving average crossed above the closing price, indicating a potential upward trend.")
        elif latest_signal == 'Sell':
            print("The 50-day moving average crossed below the closing price, indicating a potential downward trend.")
        else:
            print("The moving average and closing price did not cross, resulting in a hold recommendation.")

    # Print only the final recommendation and reasoning
    # print(f"Ticker: {ticker}")
    # print(f"Moving Average (50 days):\n{moving_average}")
    # print(f"Relative Strength Index (14 days):\n{relative_strength_index}")
    # print(f"MACD:\n{macd_df}")
    # print(f"Bollinger Bands:\n{upper_band}, {lower_band}")
    # print(f"Signals:\n{signals}")

