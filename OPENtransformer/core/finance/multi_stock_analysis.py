from finlib.APIS.stocks_api import StocksAPI
import yfinance as yf
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
import os

def analyze_stock(ticker, start_date, end_date):
    """Analyze a single stock and return the analysis results"""
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if len(data) < 50:
        print(f"Insufficient data for {ticker}, skipping")
        return None
        
    # Instantiate StocksAPI with the historical data
    api = StocksAPI(data)

    # Calculate all the stock analysis functions
    moving_average = api.moving_average(window=50)
    relative_strength_index = api.relative_strength_index(window=14)
    macd_df, macd_signal_df = api.macd(fast_period=12, slow_period=26, signal_period=9)
    upper_band, lower_band = api.bollinger_bands(window=20, num_std=2)

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
                
                # Check for NaN values using numpy's isnan
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

    # Count the buy/sell/hold signals
    signal_counts = Counter(signals)
    buy_count = signal_counts['Buy']
    sell_count = signal_counts['Sell']
    hold_count = signal_counts['Hold']

    # Calculate returns
    data['Returns'] = data['Close'].pct_change()
    
    # Create a plot with multiple subplots
    fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)

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
    signal_df = pd.DataFrame(signals, index=data.index[1:], columns=['Signal'])
    for idx, row in signal_df.iterrows():
        if row['Signal'] == 'Buy':
            axs[0].scatter(idx, data.loc[idx]['Close'], color='green', s=100, marker='^')
        elif row['Signal'] == 'Sell':
            axs[0].scatter(idx, data.loc[idx]['Close'], color='red', s=100, marker='v')

    # Format the date on the x-axis
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show every 6 months
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(f'{ticker}_analysis.png')
    plt.close()  # Close the plot to avoid displaying

    # Summary
    summary = {
        'ticker': ticker,
        'data': data,
        'signals': signals,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'hold_count': hold_count,
        'signal_percentage': {
            'buy': buy_count/len(signals)*100 if signals else 0,
            'sell': sell_count/len(signals)*100 if signals else 0,
            'hold': hold_count/len(signals)*100 if signals else 0
        },
        'latest_signal': signals[-1] if signals else 'Hold',
        'returns': data['Returns'].mean() * 252,  # Annualized returns
        'volatility': data['Returns'].std() * np.sqrt(252)  # Annualized volatility
    }
    
    return summary

def compare_stocks(stock_list, start_date, end_date):
    """Compare multiple stocks"""
    # Create a directory for plots if it doesn't exist
    os.makedirs('stock_plots', exist_ok=True)
    
    results = {}
    stock_returns = {}
    
    for ticker in stock_list:
        result = analyze_stock(ticker, start_date, end_date)
        if result:
            results[ticker] = result
            stock_returns[ticker] = result['data']['Close'] / result['data']['Close'].iloc[0]  # Normalized returns
    
    # Plot comparative returns
    plt.figure(figsize=(15, 10))
    for ticker, returns in stock_returns.items():
        plt.plot(returns.index, returns, label=ticker)
    plt.xlabel('Date')
    plt.ylabel('Normalized Returns')
    plt.title('Comparative Returns')
    plt.legend()
    plt.grid(True)
    plt.savefig('stock_plots/comparative_returns.png')
    plt.close()
    
    # Create summary comparison table
    summary_data = []
    for ticker, result in results.items():
        summary_data.append({
            'Ticker': ticker,
            'Latest Price': result['data']['Close'].iloc[-1],
            'Annualized Return': result['returns'],
            'Annualized Volatility': result['volatility'],
            'Buy Signals': result['buy_count'],
            'Sell Signals': result['sell_count'],
            'Hold Signals': result['hold_count'],
            'Current Recommendation': result['latest_signal'].upper()
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nSTOCK COMPARISON SUMMARY:")
    print(summary_df.to_string(index=False))
    
    # Plot a bar chart of buy/sell/hold recommendations
    recommendations = [result['latest_signal'].upper() for result in results.values()]
    rec_counts = Counter(recommendations)
    
    plt.figure(figsize=(10, 6))
    plt.bar(rec_counts.keys(), rec_counts.values())
    plt.title('Current Recommendations Distribution')
    plt.xlabel('Recommendation')
    plt.ylabel('Number of Stocks')
    plt.savefig('stock_plots/recommendations.png')
    plt.close()
    
    # Plot annualized returns vs volatility (risk-return plot)
    returns = [result['returns'] for result in results.values()]
    volatilities = [result['volatility'] for result in results.values()]
    tickers = list(results.keys())
    
    plt.figure(figsize=(12, 8))
    plt.scatter(volatilities, returns)
    
    # Add ticker labels to each point
    for i, ticker in enumerate(tickers):
        plt.annotate(ticker, (volatilities[i], returns[i]))
    
    plt.title('Risk vs Return')
    plt.xlabel('Risk (Annualized Volatility)')
    plt.ylabel('Return (Annualized)')
    plt.grid(True)
    plt.savefig('stock_plots/risk_return.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    # Define list of stocks to analyze
    stock_list = [
        "BPTRX",    # Original stock
        "SPY",      # S&P 500 ETF
        "QQQ",      # Nasdaq ETF
        "AAPL",     # Apple
        "MSFT",     # Microsoft
        "GOOGL",    # Google
        "AMZN",     # Amazon
        "TSLA",     # Tesla
        "NVDA",     # NVIDIA
        "JPM",      # JPMorgan Chase
        "JNJ",      # Johnson & Johnson
        "V"         # Visa
    ]
    
    # Define date range for analysis
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=5*365)  # 5 years
    
    # Run the analysis
    results = compare_stocks(stock_list, start_date, end_date)
    
    print("\nAnalysis complete! Check the 'stock_plots' directory for visualizations.")
