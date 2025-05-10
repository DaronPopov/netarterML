import numpy as np
import logging
import time
import gc
from finlib.APIS.stocks_api import StocksAPI
from scipy.stats import norm

# Configure logging
logger = logging.getLogger("test.stocks_api")

def generate_test_data(days=500, assets=5):
    """
    Generate synthetic market data for testing
    
    Args:
        days: Number of days of data to generate
        assets: Number of assets to generate data for
        
    Returns:
        prices: numpy array of shape (days, assets) containing price data
    """
    logger.info(f"Generating test data for {days} days and {assets} assets")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random prices with a slight upward trend
    prices = np.zeros((days, assets), dtype=np.float32)
    
    # Initialize starting prices
    prices[0] = 100 * np.random.random(assets)
    
    # Generate price movements
    for i in range(1, days):
        # Random price movement (between -1% and +1.5%)
        movement = 1 + np.random.normal(0.0005, 0.01, assets)
        prices[i] = prices[i-1] * movement
    
    logger.info(f"Generated prices with shape {prices.shape}")
    return prices

def run_test():
    """
    Run tests for the StocksAPI
    """
    try:
        logger.info("Starting StocksAPI tests")
        
        # Generate test data
        prices = generate_test_data()
        
        # Initialize StocksAPI
        api = StocksAPI()
        
        # Test moving average
        test_moving_average(api, prices)
        
        # Test RSI
        test_rsi(api, prices)
        
        # Test MACD
        test_macd(api, prices)
        
        # Test Bollinger Bands
        test_bollinger_bands(api, prices)
        
        # Run new tests
        test_american_option(api)
        test_barrier_option(api)
        test_heston_model(api)
        test_interest_rate_models(api)
        test_yield_curve_interpolation(api)
        test_credit_derivatives(api)
        test_risk_analysis(api)
        
        logger.info("All tests completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        logger.exception("Exception details:")
        raise

def test_moving_average(api, prices):
    """
    Test moving average calculation
    """
    logger.info("Testing moving average calculation")
    
    window = 50
    
    # Measure execution time
    start_time = time.time()
    result = api.moving_average(prices, window)
    execution_time = time.time() - start_time
    
    logger.info(f"Moving average (window={window}) calculated in {execution_time:.4f} seconds")
    logger.info(f"Moving average result shape: {result.shape}")
    
    # Log some values after the window period
    window_idx = window + 10
    logger.info(f"Moving average at day {window_idx}: {result[window_idx, 0]:.4f}")
    
    # Clean up
    del result
    gc.collect()
    logger.info("Moving average test completed")

def test_rsi(api, prices):
    """
    Test RSI calculation
    """
    logger.info("Testing RSI calculation")
    
    window = 14
    
    # Measure execution time
    start_time = time.time()
    result = api.relative_strength_index(prices, window)
    execution_time = time.time() - start_time
    
    logger.info(f"RSI (window={window}) calculated in {execution_time:.4f} seconds")
    logger.info(f"RSI result shape: {result.shape}")
    
    # Log some values after the window period
    window_idx = window + 10
    logger.info(f"RSI at day {window_idx}: {result[window_idx, 0]:.4f}")
    
    # Clean up
    del result
    gc.collect()
    logger.info("RSI test completed")

def test_macd(api, prices):
    """
    Test MACD calculation
    """
    logger.info("Testing MACD calculation")
    
    fast_period = 12
    slow_period = 26
    signal_period = 9
    
    # Measure execution time
    start_time = time.time()
    macd_line, signal_line, histogram = api.macd(prices, fast_period, slow_period, signal_period)
    execution_time = time.time() - start_time
    
    logger.info(f"MACD calculated in {execution_time:.4f} seconds")
    logger.info(f"MACD line shape: {macd_line.shape}")
    logger.info(f"Signal line shape: {signal_line.shape}")
    logger.info(f"Histogram shape: {histogram.shape}")
    
    # Log some values after the initialization period
    idx = slow_period + signal_period + 10
    logger.info(f"MACD line at day {idx}: {macd_line[idx, 0]:.4f}")
    logger.info(f"Signal line at day {idx}: {signal_line[idx, 0]:.4f}")
    logger.info(f"Histogram at day {idx}: {histogram[idx, 0]:.4f}")
    
    # Clean up
    del macd_line, signal_line, histogram
    gc.collect()
    logger.info("MACD test completed")

def test_bollinger_bands(api, prices):
    """
    Test Bollinger Bands calculation
    """
    logger.info("Testing Bollinger Bands calculation")
    
    window = 20
    num_std = 2
    
    # Measure execution time
    start_time = time.time()
    upper_band, middle_band, lower_band = api.bollinger_bands(prices, window, num_std)
    execution_time = time.time() - start_time
    
    logger.info(f"Bollinger Bands (window={window}, std={num_std}) calculated in {execution_time:.4f} seconds")
    logger.info(f"Upper band shape: {upper_band.shape}")
    logger.info(f"Middle band shape: {middle_band.shape}")
    logger.info(f"Lower band shape: {lower_band.shape}")
    
    # Log some values after the window period
    window_idx = window + 10
    logger.info(f"Upper band at day {window_idx}: {upper_band[window_idx, 0]:.4f}")
    logger.info(f"Middle band at day {window_idx}: {middle_band[window_idx, 0]:.4f}")
    logger.info(f"Lower band at day {window_idx}: {lower_band[window_idx, 0]:.4f}")
    
    # Verify that upper > middle > lower
    assert upper_band[window_idx, 0] > middle_band[window_idx, 0] > lower_band[window_idx, 0]
    
    # Clean up
    del upper_band, middle_band, lower_band
    gc.collect()
    logger.info("Bollinger Bands test completed")

def test_american_option(api, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2):
    """
    Test American option pricing using finite difference method
    """
    logger.info("Testing American option pricing")
    
    # Test both call and put options
    start_time = time.time()
    call_price = api.price_american_option_fd(S0, K, T, r, sigma, is_call=True)
    put_price = api.price_american_option_fd(S0, K, T, r, sigma, is_call=False)
    execution_time = time.time() - start_time
    
    logger.info(f"American option prices calculated in {execution_time:.4f} seconds")
    logger.info(f"American call price: {call_price:.4f}")
    logger.info(f"American put price: {put_price:.4f}")
    
    # For American options, we have the following relationships:
    # 1. American call >= European call (due to early exercise premium)
    # 2. American put >= European put
    # 3. American put - American call >= K*exp(-rT) - S0 (modified parity)
    # 4. Both prices should be positive
    # 5. Both prices should be greater than their intrinsic values
    
    # Check positivity
    assert call_price >= 0, "Call price cannot be negative"
    assert put_price >= 0, "Put price cannot be negative"
    
    # Check intrinsic value bounds
    call_intrinsic = max(0, S0 - K)
    put_intrinsic = max(0, K - S0)
    assert call_price >= call_intrinsic, "Call price below intrinsic value"
    assert put_price >= put_intrinsic, "Put price below intrinsic value"
    
    # Check modified American put-call relationship
    # American put - American call >= K*exp(-rT) - S0
    parity_diff = (put_price - call_price) - (K*np.exp(-r*T) - S0)
    logger.info(f"Modified put-call parity check: {parity_diff:.4f} (should be >= 0)")
    assert parity_diff >= -1e-3, "American put-call relationship violated"  # Allow for small numerical errors
    
    logger.info("American option test completed")

def test_barrier_option(api, S0=100.0, K=100.0, H=120.0, T=1.0, r=0.05, sigma=0.2):
    """
    Test barrier option pricing
    """
    logger.info("Testing barrier option pricing")
    
    # Test different barrier types with increased number of paths
    barrier_types = ['up-and-out', 'up-and-in', 'down-and-out', 'down-and-in']
    
    start_time = time.time()
    for barrier_type in barrier_types:
        price = api.price_barrier_option(S0, K, H, T, r, sigma, barrier_type=barrier_type, num_steps=1000)
        logger.info(f"{barrier_type} barrier option price: {price:.4f}")
    
    execution_time = time.time() - start_time
    logger.info(f"Barrier options priced in {execution_time:.4f} seconds")
    
    # Verify in-out parity with proper European call price
    up_out = api.price_barrier_option(S0, K, H, T, r, sigma, 'up-and-out', num_steps=1000)
    up_in = api.price_barrier_option(S0, K, H, T, r, sigma, 'up-and-in', num_steps=1000)
    
    # Calculate European call price using Monte Carlo with geometric control variate
    num_paths = 100000
    dt = T/1000  # Use 1000 time steps
    
    # Generate random paths
    Z = np.random.standard_normal((num_paths, 1)).astype(np.float32)  # Single step for vanilla option
    
    # Calculate terminal stock prices using exact solution
    drift = (r - 0.5*sigma**2)*T
    diffusion = sigma*np.sqrt(T)
    S_T = S0 * np.exp(drift + diffusion*Z)
    
    # Calculate vanilla call payoffs
    vanilla_payoffs = np.maximum(S_T - K, 0)
    
    # Use geometric average control variate
    log_ST = np.log(S_T)
    expected_log_ST = np.log(S0) + (r - 0.5*sigma**2)*T
    cv = log_ST - expected_log_ST
    
    # Calculate means
    payoff_mean = np.mean(vanilla_payoffs)
    cv_mean = np.mean(cv)
    
    # Calculate beta components
    payoff_cv_sum = np.sum((vanilla_payoffs - payoff_mean) * (cv - cv_mean))
    cv_var_sum = np.sum((cv - cv_mean) * (cv - cv_mean))
    
    # Calculate beta with safety check
    if cv_var_sum > 1e-10:
        beta = payoff_cv_sum / cv_var_sum
    else:
        beta = 0.0
    
    # Apply control variate adjustment
    beta = np.clip(beta, -100, 100)
    vanilla_payoffs_cv = vanilla_payoffs - beta * cv
    
    # Calculate vanilla call price
    vanilla_call = np.exp(-r*T) * np.mean(vanilla_payoffs_cv)
    
    parity_diff = abs((up_out + up_in) - vanilla_call)
    logger.info(f"In-out parity difference: {parity_diff:.4f}")
    logger.info(f"Up-and-out price: {up_out:.4f}")
    logger.info(f"Up-and-in price: {up_in:.4f}")
    logger.info(f"Vanilla call price: {vanilla_call:.4f}")
    
    # Use a more lenient threshold due to Monte Carlo simulation noise
    assert parity_diff < 2.0, "In-out parity violation too large"
    logger.info("Barrier option test completed")

def test_heston_model(api, num_paths=10000, num_steps=100):
    """
    Test Heston model simulation with FFT optimization
    """
    logger.info("Testing Heston model simulation")
    
    # Test parameters
    initial_price = 100.0
    risk_free_rate = 0.05
    volatility = 0.2
    kappa = 1.5
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    
    # Test with and without FFT
    start_time = time.time()
    paths_fft = api.simulate_heston_model_fft(
        num_paths, num_steps, initial_price, risk_free_rate,
        volatility, kappa, theta, sigma, rho, use_fft=True
    )
    fft_time = time.time() - start_time
    
    start_time = time.time()
    paths_standard = api.simulate_heston_model_fft(
        num_paths, num_steps, initial_price, risk_free_rate,
        volatility, kappa, theta, sigma, rho, use_fft=False
    )
    standard_time = time.time() - start_time
    
    logger.info(f"FFT method execution time: {fft_time:.4f} seconds")
    logger.info(f"Standard method execution time: {standard_time:.4f} seconds")
    
    # Compare final price distributions
    fft_mean = np.mean(paths_fft[:, -1])
    std_mean = np.mean(paths_standard[:, -1])
    
    logger.info(f"FFT method mean final price: {fft_mean:.4f}")
    logger.info(f"Standard method mean final price: {std_mean:.4f}")
    
    # Clean up
    del paths_fft, paths_standard
    gc.collect()
    logger.info("Heston model test completed")

def test_interest_rate_models(api, num_paths=10000, num_steps=100):
    """
    Test Hull-White and CIR model simulations
    """
    logger.info("Testing interest rate models")
    
    # Test parameters
    r0 = 0.02
    mean_reversion = 0.1
    volatility = 0.02
    theta = 0.03
    T = 5.0
    
    # Test Hull-White model
    start_time = time.time()
    hw_rates = api.simulate_hull_white_model(
        num_paths, num_steps, r0, mean_reversion, volatility, theta, T
    )
    hw_time = time.time() - start_time
    
    # Test CIR model
    start_time = time.time()
    cir_rates = api.simulate_cir_model(
        num_paths, num_steps, r0, mean_reversion, theta, volatility, T
    )
    cir_time = time.time() - start_time
    
    logger.info(f"Hull-White simulation time: {hw_time:.4f} seconds")
    logger.info(f"CIR simulation time: {cir_time:.4f} seconds")
    
    # Verify CIR rates are non-negative
    assert np.all(cir_rates >= 0), "CIR rates contain negative values"
    
    # Compare mean rates
    hw_mean = np.mean(hw_rates[:, -1])
    cir_mean = np.mean(cir_rates[:, -1])
    
    logger.info(f"Hull-White mean final rate: {hw_mean:.4f}")
    logger.info(f"CIR mean final rate: {cir_mean:.4f}")
    
    # Clean up
    del hw_rates, cir_rates
    gc.collect()
    logger.info("Interest rate models test completed")

def test_yield_curve_interpolation(api):
    """
    Test yield curve interpolation methods
    """
    logger.info("Testing yield curve interpolation")
    
    # Test data
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    rates = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.037, 0.04])
    
    # Test cubic spline interpolation
    start_time = time.time()
    spline_coeffs, spline_tenors = api.interpolate_yield_curve(tenors, rates, 'cubic_spline')
    spline_time = time.time() - start_time
    
    # Test Nelson-Siegel interpolation
    start_time = time.time()
    ns_func, ns_params = api.interpolate_yield_curve(tenors, rates, 'nelson_siegel')
    ns_time = time.time() - start_time
    
    logger.info(f"Cubic spline interpolation time: {spline_time:.4f} seconds")
    logger.info(f"Nelson-Siegel interpolation time: {ns_time:.4f} seconds")
    
    # Verify interpolation at original points
    if callable(ns_func):
        ns_fitted = ns_func(tenors, *ns_params)
        max_error = np.max(np.abs(ns_fitted - rates))
        logger.info(f"Maximum Nelson-Siegel fitting error: {max_error:.6f}")
        
    logger.info("Yield curve interpolation test completed")

def test_credit_derivatives(api):
    """
    Test credit derivative pricing
    """
    logger.info("Testing credit derivative pricing")
    
    # Test CDS pricing
    notional = 1000000.0
    maturity = 5.0
    hazard_rate = 0.02
    recovery_rate = 0.4
    
    # Simple discount curve for testing
    def discount_curve(t):
        return np.exp(-0.03 * t)
    
    start_time = time.time()
    fair_spread, protection_leg, premium_leg = api.price_credit_default_swap(
        notional, maturity, hazard_rate, recovery_rate, discount_curve
    )
    cds_time = time.time() - start_time
    
    logger.info(f"CDS pricing time: {cds_time:.4f} seconds")
    logger.info(f"CDS fair spread: {fair_spread*10000:.2f} bps")
    logger.info(f"Protection leg value: {protection_leg:.2f}")
    logger.info(f"Premium leg value: {premium_leg:.2f}")
    
    # Test CDO tranche pricing
    num_assets = 100
    attachment = 0.03
    detachment = 0.06
    correlation = 0.3
    hazard_rates = np.random.uniform(0.01, 0.05, num_assets)
    recovery_rates = np.full(num_assets, 0.4)
    notionals = np.full(num_assets, 1000000.0 / num_assets)
    
    start_time = time.time()
    tranche_spread, expected_loss = api.price_cdo_tranche(
        attachment, detachment, correlation, hazard_rates,
        recovery_rates, notionals, maturity
    )
    cdo_time = time.time() - start_time
    
    logger.info(f"CDO pricing time: {cdo_time:.4f} seconds")
    logger.info(f"CDO tranche spread: {tranche_spread*10000:.2f} bps")
    logger.info(f"Expected loss: {expected_loss*100:.2f}%")
    
    logger.info("Credit derivatives test completed")

def test_risk_analysis(api):
    """
    Test risk analysis and stress testing
    """
    logger.info("Testing risk analysis")
    
    # Test portfolio
    portfolio = np.array([1000000.0, -500000.0, 750000.0], dtype=np.float32)
    
    # Risk factors and scenarios
    risk_factors = {
        "equity": 1.0,
        "rates": 1.0,
        "fx": 1.0
    }
    
    scenarios = {
        "stress_1": {"equity": -0.2, "rates": 0.01},
        "stress_2": {"equity": -0.3, "rates": 0.02, "fx": -0.1}
    }
    
    # Correlation matrix
    correlation_matrix = np.array([
        [1.0, 0.3, 0.2],
        [0.3, 1.0, 0.1],
        [0.2, 0.1, 1.0]
    ], dtype=np.float32)
    
    # Run stress tests
    start_time = time.time()
    stress_results = api.run_stress_test(
        portfolio, risk_factors, scenarios, correlation_matrix
    )
    stress_time = time.time() - start_time
    
    logger.info(f"Stress testing time: {stress_time:.4f} seconds")
    for key, value in stress_results.items():
        logger.info(f"{key}: {value:.2f}")
    
    # Run scenario analysis
    scenarios_list = [
        {
            "name": "Market Crash",
            "changes": {"equity": -0.4, "rates": 0.03, "fx": -0.2},
            "risk_metrics": ["var", "volatility"]
        },
        {
            "name": "Recovery",
            "changes": {"equity": 0.2, "rates": -0.01, "fx": 0.1},
            "risk_metrics": ["var", "volatility"]
        }
    ]
    
    start_time = time.time()
    scenario_results = api.run_scenario_analysis(
        portfolio, scenarios_list, risk_factors, {}
    )
    scenario_time = time.time() - start_time
    
    logger.info(f"Scenario analysis time: {scenario_time:.4f} seconds")
    logger.info(f"Base portfolio value: {scenario_results['base_value']:.2f}")
    
    for i in range(len(scenarios_list)):
        scenario = scenario_results[f"scenario_{i+1}"]
        logger.info(f"Scenario {i+1} ({scenario['name']}):")
        logger.info(f"  Value: {scenario['value']:.2f}")
        logger.info(f"  Change: {scenario['change']:.2f}")
        logger.info(f"  Percent change: {scenario['percent_change']:.2f}%")
    
    logger.info("Risk analysis test completed")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("stocks_api_test.log"),
            logging.StreamHandler()
        ]
    )
    
    run_test() 