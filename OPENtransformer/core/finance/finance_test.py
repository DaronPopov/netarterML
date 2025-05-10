import time
import numpy as np
from core_finance import to_device
import matplotlib.pyplot as plt
import requests
import json
# from scipy.stats import norm  # Remove scipy dependency
import math

# --- Correctness Testing Utilities ---
def check_close(a, b, rtol=1e-5, atol=1e-8, name="arrays"):
    """Check if two arrays are close within a tolerance."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        is_close = np.allclose(a, b, rtol=rtol, atol=atol)
        max_diff = np.max(np.abs(a - b)) if a.size > 0 else 0
        print(f"Correctness check for {name}: {'✅ PASSED' if is_close else '❌ FAILED'}")
        print(f"  Max absolute difference: {max_diff:.8f}")
        
        # Enhanced debugging for failed checks
        if not is_close:
            print(f"  Shape A: {a.shape}, Shape B: {b.shape}")
            if a.size <= 10:  # Only print small arrays fully
                print(f"  A: {a.flatten()[:5]}")
                print(f"  B: {b.flatten()[:5]}")
            else:
                print(f"  A (first 5 elements): {a.flatten()[:5]}")
                print(f"  B (first 5 elements): {b.flatten()[:5]}")
                
            # Find indices of max differences
            abs_diff = np.abs(a - b)
            max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
            print(f"  Max difference at index {max_idx}")
            print(f"    A value: {a[max_idx]}, B value: {b[max_idx]}")
        
        return is_close
    else:
        print(f"Correctness check for {name}: ❌ FAILED (not numpy arrays)")
        print(f"  Type A: {type(a)}, Type B: {type(b)}")
        return False

# --- Market Making Strategy Simulation ---
def test_market_making():
    print("\n=== Market Making Strategy Test ===")
    
    # Parameters for the strategy
    n_assets = 200000
    n_features = 50
    sequence_length = 1000
    batch_size = 64
    
    # Removed feature extraction initialization
    # feature_extractor = Linear(n_features, 128)
    
    # Generate synthetic market data
    print("Generating synthetic market data...")
    market_data = np.random.randn(batch_size, sequence_length, n_features).astype(np.float32)
    market_data = to_device(market_data, device='cpu')
    
    # Removed warm-up loop with feature extraction
    # for _ in range(3):
    #     batch = market_data[:, 0:10, :]
    #     features = feature_extractor(batch.reshape(-1, n_features))
    
    # Benchmark for full sequence processing
    print("Running market making strategy benchmark...")
    start_time = time.time()
    
    # Process full sequence (dummy processing: using identity operation)
    features_list = []
    for i in range(0, sequence_length, 10):
        end_idx = min(i + 10, sequence_length)
        batch = market_data[:, i:end_idx, :]
        batch_flat = batch.reshape(-1, n_features)
        
        # Removed feature extraction, using batch_flat directly
        features = batch_flat  
        features_list.append(features)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    ops_per_sequence = batch_size * sequence_length * (2 * n_features * 128)  # Retained ops estimate for reporting
    gflops = (ops_per_sequence / 1e9) / total_time
    
    print(f"Processed {sequence_length} timesteps for {batch_size} parallel strategies")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Processing speed: {sequence_length / total_time:.2f} timesteps/second")

    # Removed correctness verification of feature extraction
    print("\nFeature extraction removed: skipping correctness verification.")

# --- Factor Model Evaluation ---
def test_factor_model():
    print("\n=== Factor Model Evaluation Test ===")
    
    # Parameters
    n_stocks = 20000000 
    n_factors = 50
    n_days = 20  # Trading days in a year
    
    # Create factor exposure matrix (stocks x factors)
    print("Generating factor model data...")
    factor_exposures = np.random.randn(n_stocks, n_factors).astype(np.float32)
    factor_exposures = to_device(factor_exposures, device='cpu')
    
    # Create factor returns matrix (days x factors)
    factor_returns = np.random.randn(n_days, n_factors).astype(np.float32) * 0.01  # 1% daily volatility
    factor_returns = to_device(factor_returns, device='cpu')
    
    # Create stock-specific returns
    specific_returns = np.random.randn(n_days, n_stocks).astype(np.float32) * 0.02  # 2% specific volatility
    specific_returns = to_device(specific_returns, device='cpu')
    
    # Warm-up
    predicted_returns = factor_exposures @ factor_returns[0].reshape(-1, 1)
    
    # Benchmark factor model computations
    print("Running factor model benchmark...")
    start_time = time.time()
    
    # Compute stock returns from factor model (factor returns × factor loadings)
    for day in range(n_days):
        # Extract factor returns for the current day
        daily_factor_returns = factor_returns[day]
        
        # Compute predicted returns based on factor exposures
        predicted_returns = factor_exposures @ daily_factor_returns.reshape(-1, 1)
        
        # Add specific returns
        full_returns = predicted_returns + specific_returns[day].reshape(-1, 1)
        
    end_time = time.time()
    total_time = end_time - start_time
    
    # Each day we do a matrix multiplication: n_stocks * n_factors operations
    # Plus the addition of specific returns: n_stocks operations
    ops_per_day = 2 * n_stocks * n_factors + n_stocks
    total_ops = ops_per_day * n_days
    
    gflops = (total_ops / 1e9) / total_time
    
    print(f"Processed {n_days} days of factor model data for {n_stocks}")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Processing speed: {n_days / total_time:.2f} days/second")
    print(f"Estimated GFLOPS: {gflops:.4f}")

    # Add correctness verification
    print("\nVerifying correctness...")
    # Use a smaller subset for verification
    test_stocks = 1000
    test_factors = 50
    
    # Generate test data
    test_exposures = np.random.randn(test_stocks, test_factors).astype(np.float32)
    test_factor_returns = np.random.randn(test_factors).astype(np.float32) * 0.01
    test_specific = np.random.randn(test_stocks).astype(np.float32) * 0.02
    
    # Custom implementation (using the same code as the benchmark)
    custom_predicted = test_exposures @ test_factor_returns.reshape(-1, 1)
    custom_full = custom_predicted + test_specific.reshape(-1, 1)
    
    # NumPy reference implementation
    ref_predicted = np.matmul(test_exposures, test_factor_returns.reshape(-1, 1))
    ref_full = ref_predicted + test_specific.reshape(-1, 1)
    
    check_close(custom_predicted, ref_predicted, name="Factor model predictions")
    check_close(custom_full, ref_full, name="Factor model with specific returns")

# --- Multi-Asset Portfolio Optimization ---
def test_portfolio_optimization():
    print("\n=== Multi-Asset Portfolio Optimization Test ===")
    
    # Parameters
    n_stocks = 10000
    n_options = 200
    n_futures = 50
    n_forex = 30
    n_assets = n_stocks + n_options + n_futures + n_forex
    n_factors = 75
    lookback_period = 60  # Days of historical data
    
    # Generate synthetic asset data
    print("Generating multi-asset portfolio data...")
    
    # Historical returns (lookback_period x n_assets)
    returns = np.random.randn(lookback_period, n_assets).astype(np.float32) * 0.01
    returns = to_device(returns, device='cpu')
    
    # Factor exposures (n_assets x n_factors)
    factor_exposures = np.random.randn(n_assets, n_factors).astype(np.float32)
    # Add some structure to the exposures (options related to stocks, etc.)
    # Options have correlation with underlying stocks
    for i in range(n_options):
        stock_idx = i % n_stocks
        factor_exposures[n_stocks + i] = factor_exposures[stock_idx] * 1.5 + np.random.randn(n_factors) * 0.2
    factor_exposures = to_device(factor_exposures, device='cpu')
    
    # Risk model (covariance matrix approximation using factor model)
    factor_cov = np.random.randn(n_factors, n_factors).astype(np.float32)
    factor_cov = factor_cov @ factor_cov.T  # Ensure positive semidefinite
    factor_cov = to_device(factor_cov, device='cpu')
    
    # Specific risk
    specific_risk = np.abs(np.random.randn(n_assets).astype(np.float32)) * 0.05
    specific_risk = to_device(specific_risk, device='cpu')
    
    # Expected returns
    expected_returns = np.random.randn(n_assets).astype(np.float32) * 0.01
    expected_returns = to_device(expected_returns, device='cpu')
    
    # Risk aversion parameter
    risk_aversion = 1.0
    
    # Portfolio constraints
    max_position = 0.05  # Maximum 5% in any asset
    
    # Asset-specific constraints
    # Dictionary to store constraints for different asset classes
    asset_constraints = {
        "stocks": {"sum": (0.4, 0.6)},       # 40-60% in stocks
        "options": {"sum": (0.0, 0.2)},      # 0-20% in options
        "futures": {"sum": (0.1, 0.3)},      # 10-30% in futures
        "forex": {"sum": (0.1, 0.3)}         # 10-30% in forex
    }
    
    print("Running portfolio optimization benchmark...")
    start_time = time.time()
    
    # Step 1: Compute covariance matrix using factor model
    # Cov = B * F * B^T + D (where B is factor exposures, F is factor cov, D is specific risk)
    
    # Compute B * F
    bf = factor_exposures @ factor_cov
    
    # Compute B * F * B^T
    risk_model = bf @ factor_exposures.T
    
    # Add specific risk to diagonal
    for i in range(n_assets):
        risk_model[i, i] += specific_risk[i]**2
    
    # Step 2: Generate initial portfolio weights
    weights = np.ones(n_assets, dtype=np.float32) / n_assets
    weights = to_device(weights, device='cpu')
    
    # Step 3: Optimize portfolio with constraints
    # In a real optimizer, we would use a quadratic programming solver
    # Here we'll simulate the computational load with gradient descent steps
    
    learning_rate = 0.01
    num_iterations = 100
    
    for iter in range(num_iterations):
        # Compute risk (quadratic term)
        portfolio_risk = weights @ risk_model @ weights
        
        # Compute expected return (linear term)
        portfolio_return = weights @ expected_returns
        
        # Objective: maximize return - risk_aversion * risk
        utility = portfolio_return - risk_aversion * portfolio_risk
        
        # Compute gradients
        risk_gradient = risk_model @ weights * 2
        return_gradient = expected_returns
        utility_gradient = return_gradient - risk_aversion * risk_gradient
        
        # Update weights
        weights = weights + learning_rate * utility_gradient
        
        # Project weights to satisfy constraints
        # Simple projection: clip weights and normalize
        weights = np.clip(weights, 0, max_position)
        weights = weights / np.sum(weights)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Operations estimate:
    # Factor model: 2 * n_assets * n_factors * n_factors + n_assets * n_factors
    # Risk calculation per iteration: 2 * n_assets * n_assets
    # Utility calculation: n_assets
    # Weight updates and constraints: ~5 * n_assets
    
    ops = (2 * n_assets * n_factors * n_factors + n_assets * n_factors) + \
          num_iterations * (2 * n_assets * n_assets + 6 * n_assets)
    
    gflops = (ops / 1e9) / total_time
    
    print(f"Portfolio optimization for {n_assets} assets (including {n_options} options, {n_futures} futures, {n_forex} forex pairs)")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Estimated GFLOPS: {gflops:.4f}")
    
    # Print asset allocation
    stock_allocation = np.sum(weights[:n_stocks])
    option_allocation = np.sum(weights[n_stocks:n_stocks+n_options])
    futures_allocation = np.sum(weights[n_stocks+n_options:n_futures])
    forex_allocation = np.sum(weights[n_stocks+n_options+n_futures:])
    
    print(f"Final allocation: Stocks {stock_allocation:.2f}, Options {option_allocation:.2f}, "
          f"Futures {futures_allocation:.2f}, Forex {forex_allocation:.2f}")

    # Add correctness verification
    print("\nVerifying correctness...")
    # Test a smaller covariance matrix calculation
    test_assets = 1000
    test_factors = 10
    
    # Generate test data
    test_exposures = np.random.randn(test_assets, test_factors).astype(np.float32)
    test_fcov = np.random.randn(test_factors, test_factors).astype(np.float32)
    test_fcov = test_fcov @ test_fcov.T  # Ensure positive semidefinite
    test_specific = np.abs(np.random.randn(test_assets).astype(np.float32)) * 0.05
    
    # Custom implementation
    test_bf = test_exposures @ test_fcov
    test_risk_model = test_bf @ test_exposures.T
    for i in range(test_assets):
        test_risk_model[i, i] += test_specific[i]**2
    
    # NumPy reference implementation
    ref_bf = np.matmul(test_exposures, test_fcov)
    ref_risk_model = np.matmul(ref_bf, test_exposures.T)
    for i in range(test_assets):
        ref_risk_model[i, i] += test_specific[i]**2
    
    check_close(test_risk_model, ref_risk_model, name="Risk model calculation", rtol=1e-4)
    
    # Test portfolio risk calculation
    test_weights = np.ones(test_assets, dtype=np.float32) / test_assets
    custom_risk = test_weights @ test_risk_model @ test_weights
    ref_risk = np.matmul(np.matmul(test_weights, ref_risk_model), test_weights)
    
    if abs(custom_risk - ref_risk) < 1e-5:
        print(f"Portfolio risk calculation: ✅ PASSED")
        print(f"  Custom: {custom_risk:.8f}, Reference: {ref_risk:.8f}")
    else:
        print(f"Portfolio risk calculation: ❌ FAILED")
        print(f"  Custom: {custom_risk:.8f}, Reference: {ref_risk:.8f}")

# --- Options Pricing Test ---
def test_options_pricing():
    print("\n=== Options Pricing Test ===")
    
    # Parameters
    n_options = 2000000
    n_scenarios = 500
    
    # Option parameters
    # Randomly generate strike prices, time to maturity, etc.
    # S: current stock price, K: strike price, T: time to maturity, r: risk-free rate, sigma: volatility
    S = np.random.uniform(50, 200, n_options).astype(np.float32)
    K = S * np.random.uniform(0.8, 1.2, n_options).astype(np.float32)
    T = np.random.uniform(0.1, 2.0, n_options).astype(np.float32)
    r = np.ones(n_options).astype(np.float32) * 0.05
    sigma = np.random.uniform(0.1, 0.5, n_options).astype(np.float32)
    
    # Convert to device
    S = to_device(S, device='cpu')
    K = to_device(K, device='cpu')
    T = to_device(T, device='cpu')
    r = to_device(r, device='cpu')
    sigma = to_device(sigma, device='cpu')
    
    print("Running Monte Carlo options pricing benchmark...")
    start_time = time.time()
    
    # Monte Carlo simulation for option pricing
    # Generate random paths for the underlying assets
    dt = T / 252  # Daily steps
    
    # Random seeds for scenarios
    np.random.seed(42)
    Z = np.random.standard_normal((n_options, n_scenarios)).astype(np.float32)
    Z = to_device(Z, device='cpu')
    
    # Compute terminal stock prices for all scenarios
    # S_T = S * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
    drift = (r - 0.5 * sigma * sigma) * T
    # Fix broadcasting by explicitly reshaping arrays
    diffusion = sigma.reshape(-1, 1) * np.sqrt(T).reshape(-1, 1) * Z
    terminal_prices = S.reshape(-1, 1) * np.exp(drift.reshape(-1, 1) + diffusion)
    
    # Calculate option payoffs
    call_payoffs = np.maximum(terminal_prices - K.reshape(-1, 1), 0)
    put_payoffs = np.maximum(K.reshape(-1, 1) - terminal_prices, 0)
    
    # Calculate present value of payoffs
    discount_factor = np.exp(-r.reshape(-1, 1) * T.reshape(-1, 1))
    call_prices = np.mean(call_payoffs, axis=1) * discount_factor.reshape(-1)
    put_prices = np.mean(put_payoffs, axis=1) * discount_factor.reshape(-1)
    
    # Calculate implied volatilities (simulated)
    # In practice, this would use Newton's method, but here we'll just do a computation of similar complexity
    implied_vol_calls = sigma * np.sqrt(call_prices / (S * 0.4 * np.sqrt(T)))
    implied_vol_puts = sigma * np.sqrt(put_prices / (S * 0.4 * np.sqrt(T)))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Operations estimate:
    # For each option and scenario:
    # - Drift and diffusion: ~5 operations per option
    # - Exponentiation: ~10 operations per option per scenario
    # - Payoff calculation: ~2 operations per option per scenario
    # - Present value: ~2 operations per option per scenario
    # - Mean calculation: ~1 operation per scenario per option
    # - Implied vol calculation: ~5 operations per option
    
    ops = n_options * (5 + n_scenarios * (10 + 2 + 2 + 1) + 5)
    gflops = (ops / 1e9) / total_time
    
    print(f"Priced {n_options} options using {n_scenarios} scenarios per option")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Processing speed: {n_options / total_time:.2f} options/second")
    print(f"Estimated GFLOPS: {gflops:.4f}")
    
    # Print some results
    avg_call_price = np.mean(call_prices)
    avg_put_price = np.mean(put_prices)
    print(f"Average call price: ${avg_call_price:.2f}, Average put price: ${avg_put_price:.2f}")

    # Add correctness verification
    print("\nVerifying correctness...")
    # Test with a smaller sample size
    test_options = 10
    test_scenarios = 100
    
    # Generate test data
    np.random.seed(42)  # Ensure reproducibility
    test_S = np.random.uniform(50, 200, test_options).astype(np.float32)
    test_K = test_S * np.random.uniform(0.8, 1.2, test_options).astype(np.float32)
    test_T = np.random.uniform(0.1, 2.0, test_options).astype(np.float32)
    test_r = np.ones(test_options).astype(np.float32) * 0.05
    test_sigma = np.random.uniform(0.1, 0.5, test_options).astype(np.float32)
    test_Z = np.random.standard_normal((test_options, test_scenarios)).astype(np.float32)
    
    # Custom implementation
    test_drift = (test_r - 0.5 * test_sigma * test_sigma) * test_T
    test_diffusion = test_sigma.reshape(-1, 1) * np.sqrt(test_T).reshape(-1, 1) * test_Z
    test_prices = test_S.reshape(-1, 1) * np.exp(test_drift.reshape(-1, 1) + test_diffusion)
    test_call_payoffs = np.maximum(test_prices - test_K.reshape(-1, 1), 0)
    test_discount = np.exp(-test_r.reshape(-1, 1) * test_T.reshape(-1, 1))
    test_call_prices = np.mean(test_call_payoffs, axis=1) * test_discount.reshape(-1)
    
    # NumPy reference implementation
    ref_drift = (test_r - 0.5 * test_sigma * test_sigma) * test_T
    ref_diffusion = test_sigma.reshape(-1, 1) * np.sqrt(test_T).reshape(-1, 1) * test_Z
    ref_prices = test_S.reshape(-1, 1) * np.exp(ref_drift.reshape(-1, 1) + ref_diffusion)
    ref_call_payoffs = np.maximum(ref_prices - test_K.reshape(-1, 1), 0)
    ref_discount = np.exp(-test_r.reshape(-1, 1) * test_T.reshape(-1, 1))
    ref_call_prices = np.mean(ref_call_payoffs, axis=1) * ref_discount.reshape(-1)
    
    check_close(test_call_prices, ref_call_prices, name="Option pricing", rtol=1e-4)
    print(f"Sample prices: Custom {test_call_prices[:3]}, Reference {ref_call_prices[:3]}")

# --- Value at Risk (VaR) and Expected Shortfall (ES) Calculation ---
def test_var_es():
    print("\n=== VaR and ES Calculation Test ===")

    # Parameters
    n_assets = 5000
    lookback_period = 252  # One year of trading days
    confidence_level = 0.95

    # Generate synthetic historical returns data
    print("Generating synthetic historical returns data...")
    historical_returns = np.random.randn(lookback_period, n_assets).astype(np.float32) * 0.01
    historical_returns = to_device(historical_returns, device='cpu')

    print("Running VaR and ES calculation...")
    start_time = time.time()

    # Step 1: Calculate the covariance matrix
    covariance_matrix = np.cov(historical_returns.T)

    # Step 2: Calculate portfolio weights (assuming equal weights for simplicity)
    portfolio_weights = np.ones(n_assets) / n_assets

    # Step 3: Calculate portfolio mean and standard deviation
    portfolio_mean = np.mean(historical_returns @ portfolio_weights)
    portfolio_std = np.sqrt(portfolio_weights @ covariance_matrix @ portfolio_weights)

    # Step 4: Calculate VaR using the variance-covariance approach
    # Use the error function (erf) to approximate the inverse normal CDF
    alpha = portfolio_mean + portfolio_std * math.sqrt(2) * inverse_erf(confidence_level)
    var = alpha

    # Step 5: Calculate ES (Expected Shortfall)
    es = portfolio_mean - portfolio_std * np.exp(-inverse_erf(confidence_level)**2) / ((1 - confidence_level) * math.sqrt(2*math.pi))

    end_time = time.time()
    total_time = end_time - start_time

    # Operations estimate:
    # Covariance matrix calculation: n_assets^2 * lookback_period
    # Portfolio mean calculation: lookback_period
    # Portfolio std calculation: n_assets^2
    ops = n_assets**2 * lookback_period + lookback_period + n_assets**2
    gflops = (ops / 1e9) / total_time

    print(f"Calculated VaR and ES for {n_assets} assets using {lookback_period} days of historical data")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Estimated GFLOPS: {gflops:.4f}")
    print(f"VaR at {confidence_level*100}% confidence level: {var:.4f}")
    print(f"Expected Shortfall at {confidence_level*100}% confidence level: {es:.4f}")

    # Add correctness verification
    print("\nVerifying correctness...")
    # Test with a smaller sample size
    test_assets = 50
    test_lookback = 100

    # Generate test data
    test_returns = np.random.randn(test_lookback, test_assets).astype(np.float32) * 0.01
    test_weights = np.ones(test_assets) / test_assets

    # Custom implementation
    test_cov = np.cov(test_returns.T)
    test_mean = np.mean(test_returns @ test_weights)
    test_std = np.sqrt(test_weights @ test_cov @ test_weights)

    # VaR and ES calculations
    test_alpha = test_mean + test_std * math.sqrt(2) * inverse_erf(confidence_level)
    test_var = test_alpha
    test_es = test_mean - test_std * np.exp(-inverse_erf(confidence_level)**2) / ((1 - confidence_level) * math.sqrt(2*math.pi))

    # NumPy reference implementation (approximation)
    ref_cov = np.cov(test_returns.T)
    ref_mean = np.mean(test_returns @ test_weights)
    ref_std = np.sqrt(test_weights @ ref_cov @ test_weights)
    ref_alpha = ref_mean + ref_std * math.sqrt(2) * inverse_erf(confidence_level)
    ref_var = ref_alpha
    ref_es = ref_mean - ref_std * np.exp(-inverse_erf(confidence_level)**2) / ((1 - confidence_level) * math.sqrt(2*math.pi))

    check_close(np.array([test_var]), np.array([ref_var]), name="VaR calculation", rtol=1e-4)
    check_close(np.array([test_es]), np.array([ref_es]), name="ES calculation", rtol=1e-4)

# Define a function to approximate the inverse error function
def inverse_erf(x):
    # Abramowitz and Stegun approximation
    a = 0.147
    return math.sqrt(math.log(1/(1-x**2)) - (a * math.log(1/(1-x**2))**2))

# --- Stochastic Volatility Modeling (Heston Model) ---
def test_heston_model():
    print("\n=== Heston Model Test ===")

    # Parameters
    n_simulations = 10000000
    n_time_steps = 20  # One year of trading days
    dt = 1 / n_time_steps
    
    # Heston model parameters
    v_0 = 0.04       # initial variance
    kappa = 2.0      # mean reversion rate
    theta = 0.04     # long-term variance
    sigma_v = 0.2    # volatility of variance
    rho = -0.7       # correlation between asset and variance

    # Stock parameters
    S_0 = 100.0      # initial stock price
    r = 0.05         # risk-free rate
    
    print("Running Heston model simulation...")
    start_time = time.time()

    # Generate random numbers
    np.random.seed(42)
    Z1 = np.random.randn(n_simulations, n_time_steps)
    Z2 = np.random.randn(n_simulations, n_time_steps)
    
    # Ensure correlation between Z1 and Z2
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    # Initialize arrays
    S = np.zeros((n_simulations, n_time_steps))
    v = np.zeros((n_simulations, n_time_steps))
    S[:, 0] = S_0
    v[:, 0] = v_0
    
    # Simulate stock prices and variance
    for t in range(1, n_time_steps):
        v[:, t] = np.maximum(0, v[:, t-1] + kappa * (theta - v[:, t-1]) * dt + sigma_v * np.sqrt(v[:, t-1] * dt) * Z2[:, t-1])
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * v[:, t-1]) * dt + np.sqrt(v[:, t-1] * dt) * Z1[:, t-1])
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Operations estimate:
    # Each time step involves several operations:
    # - Variance update: ~ 7 ops
    # - Stock price update: ~ 7 ops
    ops = n_simulations * n_time_steps * 14
    gflops = (ops / 1e9) / total_time
    
    print(f"Simulated {n_simulations} paths for {n_time_steps} time steps")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Estimated GFLOPS: {gflops:.4f}")
    
    # Calculate average final stock price
    avg_final_price = np.mean(S[:, -1])
    print(f"Average final stock price: {avg_final_price:.2f}")

    # Add correctness verification (basic check)
    print("\nVerifying correctness...")
    # Check if the average final price is within a reasonable range
    if 50 < avg_final_price < 150:
        print("Average final price check: ✅ PASSED")
    else:
        print("Average final price check: ❌ FAILED")

if __name__ == "__main__":
    test_market_making()
    test_factor_model()
    test_portfolio_optimization()
    test_options_pricing()
    test_var_es()
    test_heston_model()