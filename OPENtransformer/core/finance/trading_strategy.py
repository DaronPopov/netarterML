import numpy as np
import pandas as pd
import time
import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from finlib.APIS.stocks_api import StocksAPI
from finlib.finance.data_provider import fetch_realtime_data, fetch_historical_data, is_crypto
from finlib.finance.advanced_analytics import AdvancedAnalytics
from finlib.finance.data_cache import get_cache_instance

class TradingStrategy:
    """
    A real-time trading execution strategy that combines multiple technical indicators
    and risk management techniques.
    """
    
    def __init__(self, 
                 assets: List[str], 
                 quote_currency: str = "USD", 
                 initial_capital: float = 100000.0,
                 position_size_pct: float = 0.1,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.1,
                 max_positions: int = 5):
        """
        Initialize the trading strategy.
        
        Args:
            assets: List of asset symbols to trade
            quote_currency: Quote currency for all assets
            initial_capital: Initial capital for the strategy
            position_size_pct: Percentage of capital to allocate per position
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_positions: Maximum number of concurrent positions
        """
        self.assets = assets
        self.quote_currency = quote_currency
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        
        # Initialize StocksAPI
        self.stocks_api = StocksAPI()
        
        # Initialize Advanced Analytics
        self.advanced_analytics = AdvancedAnalytics(simulation_runs=10)
        
        # Portfolio state
        self.positions: Dict[str, Dict] = {}  # Current open positions
        self.closed_positions: List[Dict] = []  # Historical closed positions
        
        # Performance tracking
        self.equity_curve = [initial_capital]
        self.timestamps = [datetime.datetime.now()]
        
        # Trading signals history
        self.signals_history: Dict[str, List[Dict]] = {asset: [] for asset in assets}
        
        # Risk metrics
        self.portfolio_var = 0.0
        self.portfolio_es = 0.0
        
        # Execution metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.execution_times = []
        
        print(f"Trading strategy initialized with {len(assets)} assets and {initial_capital} {quote_currency} capital")
    
    def get_position_size(self, asset: str, price: float) -> float:
        """
        Calculate the position size for a given asset based on current capital.
        
        Args:
            asset: Asset symbol
            price: Current price of the asset
            
        Returns:
            float: Position size in units of the asset
        """
        # Calculate position size based on percentage of available capital
        available_capital = self.capital * self.position_size_pct
        
        # Adjust for number of current positions
        if len(self.positions) >= self.max_positions:
            return 0.0
        
        # Calculate units to buy with minimum size check
        units = available_capital / price
        
        # Ensure minimum position size (0.001 units)
        if units < 0.001:
            return 0.0
            
        # Round to 6 decimal places for crypto and 2 for stocks
        if is_crypto(asset):
            units = round(units, 6)
        else:
            units = round(units, 2)
        
        return units
    
    def analyze_technical_indicators(self, asset: str, historical_data: List[Dict], verbose: bool = True) -> Dict:
        """
        Analyze technical indicators for an asset.
        
        Args:
            asset: The asset symbol
            historical_data: Historical data for the asset (if None, will be fetched)
            verbose: Whether to print verbose output
            
        Returns:
            dict: Technical indicators and signals
        """
        start_time = time.time()
        
        # Fetch historical data if not provided
        if historical_data is None:
            period_id = '1MIN' if is_crypto(asset) else '1d'
            # Use appropriate limit based on asset type
            limit = 500 if is_crypto(asset) else 200
            historical_data = fetch_historical_data(asset, self.quote_currency, period_id=period_id, limit=limit)
            
            if not historical_data:
                if verbose:
                    print(f"Error: Failed to fetch historical data for {asset}")
                return self._get_neutral_signals(start_time, verbose)
        
        # Check if we have enough data points
        if len(historical_data) < 50:
            if verbose:
                print(f"Warning: Not enough historical data points for {asset} ({len(historical_data)} available, 50 required)")
                print(f"Fetching more data for {asset}...")
            
            # Try to fetch more data
            period_id = '1MIN' if is_crypto(asset) else '1d'
            limit = 500 if is_crypto(asset) else 200
            historical_data = fetch_historical_data(asset, self.quote_currency, period_id=period_id, limit=limit)
            
            # Check again if we have enough data
            if not historical_data or len(historical_data) < 50:
                if verbose:
                    print(f"Error: Still not enough historical data points for {asset}")
                return self._get_neutral_signals(start_time, verbose)
        
        # Extract price data from historical data
        try:
            # Try to get price from either 'rate_close' or 'close' field
            prices = []
            for item in historical_data:
                if 'rate_close' in item:
                    prices.append(float(item['rate_close']))
                elif 'close' in item:
                    prices.append(float(item['close']))
                else:
                    print(f"Error: No price data found in item: {item}")
                    return self._get_neutral_signals(start_time, verbose)
            
            prices = np.array(prices).reshape(-1, 1)
        except (KeyError, TypeError) as e:
            print(f"Error extracting price data for {asset}: {e}")
            return self._get_neutral_signals(start_time, verbose)
        
        # Check if we have enough data
        min_required_points = 50  # Increased from 30 to ensure enough data for all indicators
        if prices.shape[0] < min_required_points:
            if verbose:
                print(f"Warning: Not enough historical data points for {asset} ({prices.shape[0]} available, {min_required_points} recommended)")
                print(f"Generating synthetic data for {asset} advanced analytics demonstration...")
            
            # If we have at least some data points, use the last price for reference
            last_price = prices[-1, 0] if prices.size > 0 else 100.0
            
            # Generate synthetic price data (random walk with drift)
            np.random.seed(42)  # For reproducibility
            synthetic_points = 250  # Increased from 200 to ensure enough data
            synthetic_prices = np.zeros((synthetic_points, 1), dtype=np.float32)
            synthetic_prices[0, 0] = last_price
            
            # Use a more realistic price model with mean reversion and volatility clustering
            volatility = 0.015  # Base volatility
            mean_reversion = 0.005  # Mean reversion strength
            trend = 0.0005
            
            for i in range(1, synthetic_points):
                # Mean reversion component
                mean_reversion_component = mean_reversion * (last_price - synthetic_prices[i-1, 0])
                
                # Trend component
                trend_component = trend
                
                # Volatility clustering (GARCH-like effect)
                vol = volatility * (1 + 0.5 * abs(synthetic_prices[i-1, 0] - synthetic_prices[max(0, i-2), 0]) / last_price)
                
                # Random component
                random_component = vol * np.random.randn()
                
                # Combine components
                synthetic_prices[i, 0] = synthetic_prices[i-1, 0] + mean_reversion_component + trend_component + random_component
            
            # Use synthetic data for analysis
            prices = synthetic_prices
        
        # Calculate technical indicators
        # Use StocksAPI for efficient calculation
        api = StocksAPI()
        
        # Calculate RSI
        rsi = api.relative_strength_index(prices, window=14)
        current_rsi = rsi[-1, 0] if rsi.size > 0 else 50.0
        
        # Calculate MACD
        macd_line, signal_line, histogram = api.macd(prices, fast_period=12, slow_period=26, signal_period=9)
        current_macd = macd_line[-1, 0] if macd_line.size > 0 else 0.0
        current_signal = signal_line[-1, 0] if signal_line.size > 0 else 0.0
        
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = api.bollinger_bands(prices, window=20, num_std=2)
        current_price = prices[-1, 0]
        
        # Calculate BB position (0 = at lower band, 1 = at upper band)
        if upper_band.size > 0 and lower_band.size > 0:
            current_upper = upper_band[-1, 0]
            current_lower = lower_band[-1, 0]
            bb_range = current_upper - current_lower
            if bb_range > 0:
                bb_position = (current_price - current_lower) / bb_range
            else:
                bb_position = 0.5
        else:
            bb_position = 0.5
        
        # Calculate advanced indicators using AdvancedAnalytics
        advanced_analytics = AdvancedAnalytics()
        
        # Calculate advanced indicators
        advanced_indicators = advanced_analytics.calculate_advanced_indicators(prices)
        
        # Calculate Value at Risk (VaR)
        # Simplify the returns calculation and ensure we're working with flattened arrays
        prices_flat = prices.flatten()
        returns = np.diff(prices_flat) / prices_flat[:-1] if len(prices_flat) > 1 else np.array([0.0])
        var_result, es_result, _ = advanced_analytics.monte_carlo_var(
            returns,
            confidence_level=0.95
        )
        
        # Predict price movement
        prediction = advanced_analytics.predict_price_movement(prices.flatten(), returns)
        
        # Generate trading signal
        signal = self._generate_signal(
            current_rsi, 
            current_macd, 
            current_signal, 
            bb_position,
            advanced_indicators['hurst_exponent'],
            prediction['up_probability']
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Return the results
        return {
            'rsi': current_rsi,
            'macd': current_macd,
            'signal_line': current_signal,
            'bb_position': bb_position,
            'hurst_exponent': advanced_indicators['hurst_exponent'],
            'fractal_dimension': advanced_indicators['fractal_dimension'],
            'entropy': advanced_indicators['sample_entropy'],
            'var': var_result * 100,  # Convert to percentage
            'es': es_result * 100,  # Convert to percentage
            'up_probability': prediction['up_probability'],
            'prediction_confidence': prediction['confidence'],
            'signal': signal,
            'execution_time': execution_time
        }

    def _get_neutral_signals(self, start_time, verbose=True):
        """
        Return neutral signals for all indicators.
        
        Args:
            start_time: Start time for execution time calculation
            verbose: Whether to print verbose output
            
        Returns:
            Dict: Dictionary of neutral signals
        """
        execution_time = time.time() - start_time
        
        if verbose:
            print("Returning neutral signals due to insufficient data")
        
        return {
            "rsi": 50.0,
            "macd": 0.0,
            "signal_line": 0.0,
            "histogram": 0.0,
            "bb_upper": 0.0,
            "bb_middle": 0.0,
            "bb_lower": 0.0,
            "bb_position": 0.5,
            "hurst_exponent": 0.5,
            "fractal_dimension": 1.5,
            "entropy": 0.0,
            "var": 0.0,
            "es": 0.0,
            "up_probability": 0.5,
            "prediction_confidence": 0.0,
            "rsi_signal": "Neutral",
            "macd_signal": "Neutral",
            "bb_signal": "Neutral",
            "combined_signal": "Neutral",
            "signal": "neutral",
            "execution_time": execution_time
        }
    
    def calculate_risk_metrics(self, historical_data: List[Dict]) -> Tuple[float, float]:
        """
        Calculate risk metrics for the portfolio.
        
        Args:
            historical_data: Historical price data
            
        Returns:
            Tuple[float, float]: VaR and Expected Shortfall
        """
        try:
            # Extract prices from historical data
            if not historical_data or len(historical_data) < 10:
                print(f"Warning: Not enough historical data points for risk metrics ({len(historical_data) if historical_data else 0} available, 10 minimum)")
                return 0.05, 0.08  # Return default values
            
            # Extract close prices
            close_prices = []
            for item in historical_data:
                if 'rate_close' in item:
                    close_prices.append(item['rate_close'])
            
            if len(close_prices) < 10:
                print(f"Warning: Not enough valid price points for risk metrics ({len(close_prices)} available, 10 minimum)")
                return 0.05, 0.08  # Return default values
            
            # Calculate returns
            returns = []
            for i in range(1, len(close_prices)):
                returns.append((close_prices[i] / close_prices[i-1]) - 1)
            
            # Convert to numpy array
            returns_array = np.array([returns], dtype=np.float32)
            
            # Calculate VaR and ES
            try:
                var, es = self.stocks_api.calculate_var_and_es(returns_array, confidence_level=0.95)
                
                # Handle array outputs
                if isinstance(var, np.ndarray):
                    if var.size > 0:
                        var = float(var[0])
                    else:
                        var = 0.05
                
                if isinstance(es, np.ndarray):
                    if es.size > 0:
                        es = float(es[0])
                    else:
                        es = 0.08
                
                # Check for NaN or inf
                if np.isnan(var) or np.isinf(var):
                    var = 0.05
                if np.isnan(es) or np.isinf(es):
                    es = 0.08
                
                return float(var), float(es)
            except Exception as e:
                print(f"Error in VaR calculation: {e}")
                return 0.05, 0.08
                
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return 0.05, 0.08  # Return default values on error
    
    def execute_trade(self, asset: str, signal: str, price: float, verbose: bool = True) -> Optional[Dict]:
        """
        Execute a trade based on the signal.
        
        Args:
            asset: Asset symbol
            signal: Buy or Sell signal
            price: Current price
            verbose: Whether to print verbose output
            
        Returns:
            Optional[Dict]: Trade details if a trade was executed, None otherwise
        """
        if signal not in ["Buy", "Sell"]:
            if verbose:
                print(f"Invalid signal: {signal}")
            return None
        
        # Check if we have enough capital for a buy
        if signal == "Buy":
            # Calculate position size (5% of capital)
            position_value = self.capital * self.position_size_pct
            units = position_value / price
            
            # Round units appropriately
            if is_crypto(asset):
                units = round(units, 6)
            else:
                units = round(units, 2)
            
            # Ensure minimum position size
            if units < 0.001:
                if verbose:
                    print(f"Position size too small for {asset}: {units} units")
                return None
            
            # Calculate actual cost
            cost = units * price
            
            # Check if we have enough capital
            if cost > self.capital:
                if verbose:
                    print(f"Not enough capital to buy {units} units of {asset} at {price}")
                return None
            
            # Check if we already have a position
            if asset in self.positions:
                if verbose:
                    print(f"Already have a position in {asset}")
                return None
            
            # Check if we have reached the maximum number of positions
            if len(self.positions) >= self.max_positions:
                if verbose:
                    print(f"Maximum number of positions reached ({self.max_positions})")
                return None
            
            # Execute the buy
            self.capital -= cost
            
            # Calculate stop loss and take profit levels
            stop_loss = price * (1 - self.stop_loss_pct)
            take_profit = price * (1 + self.take_profit_pct)
            
            # Record the position
            self.positions[asset] = {
                "units": units,
                "entry_price": price,
                "entry_time": datetime.datetime.now(),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "cost": cost
            }
            
            # Update trade count
            self.total_trades += 1
            
            if verbose:
                print(f"Executed BUY for {asset}:")
                print(f"  Units: {units}")
                print(f"  Price: {price}")
                print(f"  Cost: {cost}")
                print(f"  Stop Loss: {stop_loss}")
                print(f"  Take Profit: {take_profit}")
            
            return {
                "asset": asset,
                "signal": signal,
                "units": units,
                "price": price,
                "cost": cost,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "time": datetime.datetime.now()
            }
            
        elif signal == "Sell":
            # Check if we have a position to sell
            if asset not in self.positions:
                if verbose:
                    print(f"No position in {asset} to sell")
                return None
            
            # Get position details
            position = self.positions[asset]
            units = position["units"]
            entry_price = position["entry_price"]
            cost = position["cost"]
            
            # Calculate profit/loss
            proceeds = units * price
            profit_loss = proceeds - cost
            profit_loss_pct = (price / entry_price - 1) * 100
            
            # Update capital
            self.capital += proceeds
            
            # Update trade statistics
            if profit_loss > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Remove the position
            del self.positions[asset]
            
            if verbose:
                print(f"Executed SELL for {asset}:")
                print(f"  Units: {units}")
                print(f"  Entry Price: {entry_price} {self.quote_currency}")
                print(f"  Exit Price: {price} {self.quote_currency}")
                print(f"  Profit/Loss: {profit_loss} {self.quote_currency} ({profit_loss_pct:.2f}%)")
            
            return {
                "asset": asset,
                "signal": signal,
                "units": units,
                "entry_price": entry_price,
                "exit_price": price,
                "profit_loss": profit_loss,
                "profit_loss_pct": profit_loss_pct,
                "time": datetime.datetime.now()
            }
        
        return None
    
    def check_stop_loss_take_profit(self, asset: str, current_price: float, verbose: bool = True) -> Optional[Dict]:
        """
        Check if stop loss or take profit has been triggered for a position.
        
        Args:
            asset: Asset symbol
            current_price: Current price
            verbose: Whether to print verbose output
            
        Returns:
            Optional[Dict]: Trade details if a trade was executed, None otherwise
        """
        # Check if we have a position in this asset
        if asset not in self.positions:
            return None
        
        position = self.positions[asset]
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        
        # Check if stop loss has been triggered
        if current_price <= stop_loss:
            if verbose:
                print(f"Stop loss triggered for {asset} at {current_price} {self.quote_currency}")
            return self.execute_trade(asset, "Sell", current_price, verbose)
        
        # Check if take profit has been triggered
        if current_price >= take_profit:
            if verbose:
                print(f"Take profit triggered for {asset} at {current_price} {self.quote_currency}")
            return self.execute_trade(asset, "Sell", current_price, verbose)
        
        return None
    
    def update_portfolio_value(self):
        """
        Update the portfolio value and equity curve.
        """
        # Calculate current portfolio value
        portfolio_value = self.capital
        
        # Add value of open positions
        for asset, position in self.positions.items():
            # Fetch current price
            data = fetch_realtime_data(asset, self.quote_currency)
            if data:
                current_price = data['rate']
                position_value = position["units"] * current_price
                portfolio_value += position_value
        
        # Update equity curve
        self.equity_curve.append(portfolio_value)
        self.timestamps.append(datetime.datetime.now())
    
    def print_performance_metrics(self, verbose: bool = True):
        """
        Print performance metrics for the strategy.
        
        Args:
            verbose: Whether to print verbose output
        """
        # Calculate win rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Calculate portfolio value
        portfolio_value = self.capital
        for asset, position in self.positions.items():
            portfolio_value += position["units"] * position["entry_price"]
        
        # Calculate total return
        total_return = ((portfolio_value / self.equity_curve[0]) - 1) * 100
        
        # Calculate average execution time
        avg_execution_time = sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0
        
        # Update equity curve
        self.equity_curve.append(portfolio_value)
        self.timestamps.append(datetime.datetime.now())
        
        # Print metrics
        if verbose:
            print("\n" + "="*50)
            print("TRADING STRATEGY PERFORMANCE METRICS")
            print("="*50)
            print(f"Initial Capital: {self.equity_curve[0]:.2f} {self.quote_currency}")
            print(f"Current Portfolio Value: {portfolio_value:.2f} {self.quote_currency}")
            print(f"Total Return: {total_return:.2f}%")
            print(f"Total Trades: {self.total_trades}")
            print(f"Winning Trades: {self.winning_trades}")
            print(f"Losing Trades: {self.losing_trades}")
            print(f"Win Rate: {win_rate:.2f}%")
            
            # Print risk metrics if available
            if hasattr(self, 'portfolio_var') and hasattr(self, 'portfolio_es'):
                print(f"Portfolio VaR (95%): {self.portfolio_var:.2f}%")
                print(f"Portfolio ES (95%): {self.portfolio_es:.2f}%")
            else:
                print(f"Portfolio VaR (95%): 0.00%")
                print(f"Portfolio ES (95%): 0.00%")
            
            print(f"Average Execution Time: {avg_execution_time*1000:.2f} ms")
            print(f"Open Positions: {len(self.positions)}")
            
            # Print open positions
            if self.positions:
                print("\nOpen Positions:")
                print("-"*50)
                for asset, position in self.positions.items():
                    print(f"Asset: {asset}")
                    print(f"Units: {position['units']}")
                    print(f"Entry Price: {position['entry_price']:.2f} {self.quote_currency}")
                    print(f"Current Value: {position['units'] * position['entry_price']:.2f} {self.quote_currency}")
                    print(f"Stop Loss: {position['stop_loss']:.2f} {self.quote_currency}")
                    print(f"Take Profit: {position['take_profit']:.2f} {self.quote_currency}")
                    print("-"*50)
        else:
            # Check if we have enough historical data for risk metrics
            if not hasattr(self, 'portfolio_var') or not hasattr(self, 'portfolio_es'):
                if len(self.equity_curve) < 10:
                    print(f"Warning: Not enough historical data points for risk metrics ({len(self.equity_curve)} available, 10 minimum)")
            
            print(f"Total trades: {self.total_trades}")
            print(f"Win rate: {win_rate:.2f}%")
            print(f"Final portfolio value: {portfolio_value:.2f} {self.quote_currency}")
            print(f"Total return: {total_return:.2f}%")
            print(f"Average execution time: {avg_execution_time*1000:.2f} ms")
    
    def plot_equity_curve(self):
        """
        Plot the equity curve.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.equity_curve)
        plt.title('Strategy Equity Curve')
        plt.xlabel('Time')
        plt.ylabel(f'Portfolio Value ({self.quote_currency})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('equity_curve.png')
        print("Equity curve saved to 'equity_curve.png'")
    
    def run_strategy(self, iterations: int = 5, sleep_time: float = 0.1, verbose: bool = True, quiet: bool = False, use_cached_data: bool = True):
        """
        Run the trading strategy for a specified number of iterations.
        
        Args:
            iterations: Number of iterations to run
            sleep_time: Sleep time between iterations in seconds
            verbose: Whether to print verbose output
            quiet: Whether to suppress all output except final summary
            use_cached_data: Whether to use cached data when available
        """
        # Initialize the cache
        cache = get_cache_instance()
        
        # Initialize the equity curve and timestamps
        self.equity_curve = [self.initial_capital]
        self.timestamps = [datetime.datetime.now()]
        
        # Initialize the signals history
        self.signals_history = {asset: [] for asset in self.assets}
        
        # Initialize the positions
        self.positions = []
        
        # Initialize trade statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Preload historical data for all assets if using cached data
        if use_cached_data:
            if verbose:
                print("Preloading historical data for all assets...")
            
            # Preload data for each asset
            for asset in self.assets:
                period_id = '1MIN' if is_crypto(asset) else '1d'
                
                # Check if data is already in cache
                cached_data = cache.get_cached_data(
                    asset, 
                    self.quote_currency, 
                    period_id, 
                    is_crypto(asset)
                )
                
                if cached_data is None:
                    if verbose:
                        print(f"Preloading data for {asset}...")
                    
                    # Fetch and cache data - use appropriate limit based on asset type
                    limit = 500 if is_crypto(asset) else 200
                    historical_data = fetch_historical_data(asset, self.quote_currency, period_id=period_id, limit=limit)
                    
                    if historical_data:
                        if verbose:
                            print(f"Successfully preloaded data for {asset}")
                    else:
                        if verbose:
                            print(f"Failed to preload data for {asset}")
                else:
                    if verbose:
                        print(f"Using cached data for {asset}")
                    # Check if we have enough data points
                    if len(cached_data) < 50:
                        if verbose:
                            print(f"Cached data for {asset} has only {len(cached_data)} data points, fetching more...")
                        # Fetch and cache data with increased limit
                        limit = 500 if is_crypto(asset) else 200
                        historical_data = fetch_historical_data(asset, self.quote_currency, period_id=period_id, limit=limit)
                        if historical_data and len(historical_data) >= 50:
                            if verbose:
                                print(f"Successfully fetched {len(historical_data)} data points for {asset}")
        
        # Run the strategy for the specified number of iterations
        for i in range(iterations):
            if verbose and not quiet:
                print(f"\nIteration {i+1}/{iterations}")
                print("-" * 50)
            
            # Update the portfolio value
            self.update_portfolio_value()
            
            # Add the current portfolio value to the equity curve
            self.equity_curve.append(self.portfolio_value)
            # Add the current timestamp
            self.timestamps.append(datetime.datetime.now())
            
            # Analyze each asset
            for asset in self.assets:
                # Skip if we already have the maximum number of positions
                if len(self.positions) >= self.max_positions:
                    if verbose and not quiet:
                        print(f"Maximum number of positions reached ({self.max_positions}). Skipping {asset}.")
                    continue
                
                # Check if we already have a position for this asset
                if any(p['asset'] == asset for p in self.positions):
                    if verbose and not quiet:
                        print(f"Already have a position for {asset}. Skipping.")
                    continue
                
                # Analyze the asset
                signals = self.analyze_technical_indicators(asset, None, verbose and not quiet)
                
                # Store the signals in the history
                self.signals_history[asset].append(signals)
                
                # Check if we should enter a position
                if signals['signal'] == 'buy':
                    # Enter a position
                    self.enter_position(asset, signals, verbose and not quiet)
            
            # Update existing positions
            self.update_positions(verbose and not quiet)
            
            # Sleep between iterations
            if i < iterations - 1:
                time.sleep(sleep_time)
        
        # Print the final summary
        if not quiet:
            self.print_summary()
        
        # Return the final portfolio value
        return self.portfolio_value
    
    def save_equity_curve(self, filename='equity_curve.csv', quiet=False):
        """
        Save the equity curve to a CSV file.
        
        Args:
            filename: Name of the CSV file
            quiet: Whether to suppress output
        """
        try:
            # Create a DataFrame with timestamps and equity curve
            df = pd.DataFrame({
                'timestamp': self.timestamps,
                'equity': self.equity_curve
            })
            
            # Save to CSV
            df.to_csv(filename, index=False)
            
            if not quiet:
                print(f"Equity curve saved to '{filename}'")
        except Exception as e:
            print(f"Error saving equity curve: {e}")

    def _generate_signal(self, rsi, macd, signal_line, bb_position, hurst_exponent, up_probability):
        """
        Generate a trading signal based on technical indicators and advanced analytics.
        More aggressive signal generation for testing purposes.
        
        Args:
            rsi: Relative Strength Index value
            macd: MACD line value
            signal_line: MACD signal line value
            bb_position: Bollinger Band position (0 = at lower band, 1 = at upper band)
            hurst_exponent: Hurst exponent value
            up_probability: Probability of price going up
            
        Returns:
            str: Trading signal ('buy', 'sell', or 'neutral')
        """
        # More aggressive RSI thresholds
        rsi_signal = "buy" if rsi < 40 else "sell" if rsi > 60 else "neutral"
        
        # More sensitive MACD crossover
        macd_signal = "buy" if macd > signal_line * 0.95 else "sell" if macd < signal_line * 1.05 else "neutral"
        
        # More aggressive Bollinger Band signals
        bb_signal = "buy" if bb_position < 0.3 else "sell" if bb_position > 0.7 else "neutral"
        
        # Combine signals with advanced indicators - more aggressive thresholds
        if hurst_exponent > 0.5:  # Trending market - lower threshold
            if up_probability > 0.55:  # More aggressive up probability threshold
                final_signal = "buy"
            elif up_probability < 0.45:  # More aggressive down probability threshold
                final_signal = "sell"
            else:
                # Use technical indicators in trending market
                if rsi_signal == "buy" or macd_signal == "buy":  # Changed from AND to OR
                    final_signal = "buy"
                elif rsi_signal == "sell" or macd_signal == "sell":  # Changed from AND to OR
                    final_signal = "sell"
                else:
                    final_signal = "neutral"
        elif hurst_exponent < 0.45:  # Mean-reverting market - adjusted threshold
            if bb_position > 0.7 and rsi > 60:  # More aggressive overbought
                final_signal = "sell"
            elif bb_position < 0.3 and rsi < 40:  # More aggressive oversold
                final_signal = "buy"
            else:
                final_signal = "neutral"
        else:  # Random market
            # Use majority vote with more weight on probability
            signals = [rsi_signal, macd_signal, bb_signal]
            buy_count = signals.count("buy")
            sell_count = signals.count("sell")
            
            if (buy_count > 0 and up_probability > 0.52) or buy_count > sell_count:  # More aggressive probability threshold
                final_signal = "buy"
            elif (sell_count > 0 and up_probability < 0.48) or sell_count > buy_count:
                final_signal = "sell"
            else:
                final_signal = "neutral"
        
        return final_signal

    def enter_position(self, asset, signals, verbose=True):
        """
        Enter a new position for an asset.
        
        Args:
            asset: The asset symbol
            signals: The signals dictionary
            verbose: Whether to print verbose output
        """
        # Get the current price
        realtime_data = fetch_realtime_data(asset, self.quote_currency)
        if not realtime_data:
            if verbose:
                print(f"Error: Failed to fetch real-time data for {asset}")
            return
        
        current_price = realtime_data['rate']
        
        # Calculate position size
        position_size = self.capital * self.position_size_pct
        
        # Calculate number of units
        units = position_size / current_price
        
        # Calculate stop loss and take profit levels
        stop_loss = current_price * (1 - self.stop_loss_pct)
        take_profit = current_price * (1 + self.take_profit_pct)
        
        # Create the position
        position = {
            'asset': asset,
            'entry_price': current_price,
            'units': units,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.datetime.now(),
            'signals': signals
        }
        
        # Add the position to the list
        self.positions.append(position)
        
        # Increment the total trades counter
        self.total_trades += 1
        
        if verbose:
            print(f"Entered position for {asset} at {current_price:.2f} {self.quote_currency}")
            print(f"Position size: {position_size:.2f} {self.quote_currency}")
            print(f"Units: {units:.6f}")
            print(f"Stop loss: {stop_loss:.2f} {self.quote_currency}")
            print(f"Take profit: {take_profit:.2f} {self.quote_currency}")
    
    def update_positions(self, verbose=True):
        """
        Update all open positions.
        
        Args:
            verbose: Whether to print verbose output
        """
        if verbose:
            print(f"\nUpdating {len(self.positions)} open positions...")
        
        positions_to_remove = []
        
        for position in self.positions:
            asset = position['asset']
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            # Get the current price
            realtime_data = fetch_realtime_data(asset, self.quote_currency)
            if not realtime_data:
                if verbose:
                    print(f"Error: Failed to fetch real-time data for {asset}")
                continue
            
            current_price = realtime_data['rate']
            
            # Calculate profit/loss
            pnl_pct = (current_price - entry_price) / entry_price * 100
            
            if verbose:
                print(f"{asset}: Entry: {entry_price:.2f}, Current: {current_price:.2f}, P/L: {pnl_pct:.2f}%")
            
            # Check if stop loss or take profit has been hit
            if current_price <= stop_loss:
                if verbose:
                    print(f"Stop loss hit for {asset} at {current_price:.2f} {self.quote_currency}")
                
                # Mark the position for removal
                positions_to_remove.append(position)
                
                # Update trade statistics
                self.losing_trades += 1
            elif current_price >= take_profit:
                if verbose:
                    print(f"Take profit hit for {asset} at {current_price:.2f} {self.quote_currency}")
                
                # Mark the position for removal
                positions_to_remove.append(position)
                
                # Update trade statistics
                self.winning_trades += 1
        
        # Remove closed positions
        for position in positions_to_remove:
            self.positions.remove(position)
    
    def update_portfolio_value(self):
        """
        Update the portfolio value based on current positions.
        """
        # Start with the initial capital
        portfolio_value = self.initial_capital
        
        # Add the value of all open positions
        for position in self.positions:
            asset = position['asset']
            units = position['units']
            
            # Get the current price
            realtime_data = fetch_realtime_data(asset, self.quote_currency)
            if not realtime_data:
                continue
            
            current_price = realtime_data['rate']
            
            # Add the position value to the portfolio value
            position_value = units * current_price
            portfolio_value += position_value
        
        # Update the portfolio value
        self.portfolio_value = portfolio_value
    
    def print_summary(self):
        """
        Print a summary of the trading strategy execution.
        """
        print("\n" + "=" * 50)
        print("TRADING STRATEGY EXECUTION COMPLETED")
        print("=" * 50)
        print(f"Total trades: {self.total_trades}")
        print(f"Winning trades: {self.winning_trades}")
        print(f"Losing trades: {self.losing_trades}")
        print(f"Win rate: {(self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0:.2f}%")
        print(f"Final portfolio value: {self.portfolio_value:.2f} {self.quote_currency}")
        print(f"Total return: {((self.portfolio_value / self.initial_capital) - 1) * 100:.2f}%")
        print("=" * 50)
        
        # Save the equity curve
        self.save_equity_curve()

    def run(self, prices: Dict[str, float]) -> List[Dict]:
        """
        Run a single iteration of the trading strategy with the provided prices.
        
        Args:
            prices: Dictionary mapping asset symbols to current prices
            
        Returns:
            list: List of executed trades
        """
        trades = []
        
        # Process each asset
        for asset in self.assets:
            if asset not in prices or prices[asset] is None:
                continue
                
            price = prices[asset]
            
            # First check existing positions for stop loss/take profit
            if asset in self.positions:
                # Check stop loss and take profit
                trade = self.check_stop_loss_take_profit(asset, price, verbose=False)
                if trade:
                    trades.append(trade)
                    continue
            
            # Only analyze for new positions if we haven't hit our maximum
            if len(self.positions) < self.max_positions and asset not in self.positions:
                # Get technical indicators
                rsi, macd, signal_line, bb_position = self.get_technical_indicators(asset, price)
                
                # Get advanced analytics
                hurst_exponent, up_probability = self.get_advanced_analytics(asset, price)
                
                # Generate signal
                signal = self._generate_signal(rsi, macd, signal_line, bb_position, hurst_exponent, up_probability)
                
                # Execute trade if signal is buy
                if signal == "buy":
                    trade = self.execute_trade(asset, "Buy", price, verbose=False)
                    if trade:
                        trades.append(trade)
        
        return trades
    
    def initialize(self):
        """
        Initialize the trading strategy.
        """
        # Initialize the StocksAPI
        self.stocks_api.initialize()
        
        # Initialize the Advanced Analytics
        self.advanced_analytics.initialize()
        
        # Reset performance metrics
        self.execution_times = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Reset portfolio state
        self.positions = {}
        self.closed_positions = []
        
        # Reset capital to initial amount
        self.capital = float(self.initial_capital)  # Ensure it's a float
        
        # Reset equity curve
        self.equity_curve = [self.initial_capital]
        self.timestamps = [datetime.datetime.now()]
        
        # Reset signals history
        self.signals_history = {asset: [] for asset in self.assets}
        
        # Reset risk metrics
        self.portfolio_var = 0.0
        self.portfolio_es = 0.0
        
        # Reset last analysis time
        self.last_analysis_time = 0.0
        
        print(f"Strategy initialized with {self.capital} capital")

    def get_technical_indicators(self, asset: str, current_price: float) -> Tuple[float, float, float, float]:
        """
        Calculate technical indicators for an asset.
        
        Args:
            asset: Asset symbol
            current_price: Current price of the asset
            
        Returns:
            tuple: (RSI, MACD, Signal Line, Bollinger Band Position)
        """
        try:
            # Use StocksAPI to calculate technical indicators
            rsi = self.stocks_api.calculate_rsi(asset, current_price)
            macd, signal_line = self.stocks_api.calculate_macd(asset, current_price)
            bb_position = self.stocks_api.calculate_bollinger_band_position(asset, current_price)
            
            return rsi, macd, signal_line, bb_position
        except Exception as e:
            # Return default values if calculation fails
            return 50.0, 0.0, 0.0, 0.5
    
    def get_advanced_analytics(self, asset: str, current_price: float) -> Tuple[float, float]:
        """
        Calculate advanced analytics for an asset.
        
        Args:
            asset: Asset symbol
            current_price: Current price of the asset
            
        Returns:
            tuple: (Hurst Exponent, Up Probability)
        """
        try:
            # Use AdvancedAnalytics to calculate advanced indicators
            hurst_exponent = self.advanced_analytics.calculate_hurst_exponent(asset, current_price)
            up_probability = self.advanced_analytics.calculate_price_movement_probability(asset, current_price)
            
            return hurst_exponent, up_probability
        except Exception as e:
            # Return default values if calculation fails
            return 0.5, 0.5 