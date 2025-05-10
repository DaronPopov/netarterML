import numpy as np
from core_finance import to_device
import torch

class FinanceAPI:
    """
    A user-friendly API for common financial computations.
    """

    def __init__(self, device='cpu'):
        """
        Initializes the FinanceAPI.

        Args:
            device (str): The device to run computations on ('cpu' or 'cuda').
        """
        self.device = device
        if device not in ['cpu', 'cuda']:
            raise ValueError("Invalid device specified. Must be 'cpu' or 'cuda'.")

    def factor_model(self, factor_exposures, factor_returns, specific_returns):
        """
        Computes stock returns using a factor model.

        Args:
            factor_exposures (numpy.ndarray): Factor exposure matrix (n_stocks x n_factors).
            factor_returns (numpy.ndarray): Factor returns matrix (n_days x n_factors).
            specific_returns (numpy.ndarray): Stock-specific returns matrix (n_days x n_stocks).

        Returns:
            numpy.ndarray: A matrix of predicted stock returns (n_days x n_stocks).
        """
        # Input validation
        if not isinstance(factor_exposures, np.ndarray) or factor_exposures.ndim != 2:
            raise ValueError("factor_exposures must be a 2D numpy array.")
        if not isinstance(factor_returns, np.ndarray) or factor_returns.ndim != 2:
            raise ValueError("factor_returns must be a 2D numpy array.")
        if not isinstance(specific_returns, np.ndarray) or specific_returns.ndim != 2:
            raise ValueError("specific_returns must be a 2D numpy array.")

        num_stocks, num_factors = factor_exposures.shape
        num_days, num_factors_returns = factor_returns.shape

        if num_factors != num_factors_returns:
            raise ValueError("Number of factors in factor_exposures and factor_returns must match.")
        if specific_returns.shape[1] != num_stocks:
            raise ValueError("Number of stocks in factor_exposures and specific_returns must match.")
        if specific_returns.shape[0] != num_days:
             raise ValueError("Number of days in factor_returns and specific_returns must match.")

        factor_exposures = to_device(factor_exposures, device=self.device)
        factor_returns = to_device(factor_returns, device=self.device)
        specific_returns = to_device(specific_returns, device=self.device)

        predicted_returns = np.zeros((num_days, num_stocks), dtype=np.float32)

        for day in range(num_days):
            daily_factor_returns = factor_returns[day]
            predicted_returns[day, :] = (factor_exposures @ daily_factor_returns).flatten() + specific_returns[day, :]

        return predicted_returns

    def monte_carlo_option_price(self, spot_prices, strike_prices, time_to_maturity, risk_free_rate, volatility, num_scenarios=5000):
        """
        Prices European call options using Monte Carlo simulation.

        Args:
            spot_prices (numpy.ndarray): Current stock price(s).
            strike_prices (numpy.ndarray): Strike price(s).
            time_to_maturity (numpy.ndarray): Time to maturity (in years).
            risk_free_rate (numpy.ndarray): Risk-free rate(s).
            volatility (numpy.ndarray): Volatility(ies).
            num_scenarios (int): Number of Monte Carlo scenarios.

        Returns:
            numpy.ndarray: Prices of the call options.
        """
        # Input validation
        if not isinstance(spot_prices, np.ndarray) or not isinstance(strike_prices, np.ndarray) or \
           not isinstance(time_to_maturity, np.ndarray) or not isinstance(risk_free_rate, np.ndarray) or \
           not isinstance(volatility, np.ndarray):
            raise ValueError("All price, rate, and volatility inputs must be numpy arrays.")

        num_options = spot_prices.shape[0]
        if not all(arr.shape[0] == num_options for arr in [strike_prices, time_to_maturity, risk_free_rate, volatility]):
            raise ValueError("All input arrays must have the same number of options.")

        spot_prices = to_device(spot_prices, device=self.device)
        strike_prices = to_device(strike_prices, device=self.device)
        time_to_maturity = to_device(time_to_maturity, device=self.device)
        risk_free_rate = to_device(risk_free_rate, device=self.device)
        volatility = to_device(volatility, device=self.device)
        
        # Generate random paths for the underlying assets
        np.random.seed(42)
        z = np.random.standard_normal((num_options, num_scenarios)).astype(np.float32)
        z = to_device(z, device=self.device)
        
        # Compute terminal stock prices for all scenarios
        drift = (risk_free_rate - 0.5 * volatility * volatility) * time_to_maturity
        diffusion = volatility.reshape(-1, 1) * np.sqrt(time_to_maturity).reshape(-1, 1) * z
        terminal_prices = spot_prices.reshape(-1, 1) * np.exp(drift.reshape(-1, 1) + diffusion)
        
        # Calculate option payoffs
        call_payoffs = np.maximum(terminal_prices - strike_prices.reshape(-1, 1), 0)
        
        # Calculate present value of payoffs
        discount_factor = np.exp(-risk_free_rate.reshape(-1, 1) * time_to_maturity.reshape(-1, 1))
        call_payoffs_np = call_payoffs.cpu().numpy()
        discount_factor_np = discount_factor.cpu().numpy()
        call_prices = np.mean(call_payoffs_np, axis=1) * discount_factor_np.reshape(-1)
        
        return call_prices

    def risk_model(self, factor_exposures, factor_covariance_matrix, specific_risk):
        """
        Computes the covariance matrix using a factor model.

        Args:
            factor_exposures (numpy.ndarray): Factor exposure matrix (n_assets x n_factors).
            factor_covariance_matrix (numpy.ndarray): Factor covariance matrix (n_factors x n_factors).
            specific_risk (numpy.ndarray): Vector of specific risk for each asset (n_assets).

        Returns:
            numpy.ndarray: The computed covariance matrix (n_assets x n_assets).
        """
        # Input validation
        if not isinstance(factor_exposures, np.ndarray) or factor_exposures.ndim != 2:
            raise ValueError("factor_exposures must be a 2D numpy array.")
        if not isinstance(factor_covariance_matrix, np.ndarray) or factor_covariance_matrix.ndim != 2:
            raise ValueError("factor_covariance_matrix must be a 2D numpy array.")
        if not isinstance(specific_risk, np.ndarray) or specific_risk.ndim != 1:
            raise ValueError("specific_risk must be a 1D numpy array.")

        num_assets, num_factors = factor_exposures.shape
        if factor_covariance_matrix.shape != (num_factors, num_factors):
            raise ValueError("factor_covariance_matrix must be square with dimension equal to the number of factors.")
        if specific_risk.shape[0] != num_assets:
            raise ValueError("specific_risk must have a length equal to the number of assets.")

        factor_exposures = to_device(factor_exposures, device=self.device)
        factor_covariance_matrix = to_device(factor_covariance_matrix, device=self.device)
        specific_risk = to_device(specific_risk, device=self.device)

        # Cov = B * F * B^T + D (where B is factor exposures, F is factor cov, D is specific risk)
        bf = factor_exposures @ factor_covariance_matrix
        risk_model = bf @ factor_exposures.T

        # Add specific risk to diagonal
        for i in range(num_assets):
            risk_model[i, i] += specific_risk[i]**2

        return risk_model
    
def to_device(data, device='cpu'):
    """
    Converts a numpy array to a PyTorch tensor and moves it to the specified device.

    Args:
        data (numpy.ndarray): The input data.
        device (str): The device to move the data to ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The data on the specified device.
    """
    return torch.tensor(data, device=device)

def execute_kernel(kernel_func, *args):
    """
    Executes a kernel function with the specified arguments.

    Args:
        kernel_func (function): The kernel function to execute.
        args: The arguments to pass to the kernel function.
    """
    kernel_func(*args)

# Example usage (can be placed in a separate test script or at the end of this file)
if __name__ == '__main__':
    # --- Factor Model Example ---
    print("=== Factor Model Example ===")
    n_stocks = 3000
    n_factors = 1000000
    n_days = 252

    # Generate synthetic data
    factor_exposures = np.random.randn(n_stocks, n_factors).astype(np.float32)
    factor_returns = np.random.randn(n_days, n_factors).astype(np.float32) * 0.01
    specific_returns = np.random.randn(n_days, n_stocks).astype(np.float32) * 0.02

    # Initialize the API
    finance_api = FinanceAPI()

    # Compute stock returns
    predicted_returns = finance_api.factor_model(factor_exposures, factor_returns, specific_returns)

    print(f"Shape of predicted returns: {predicted_returns.shape}")
    print(f"Sample predicted returns (first 5 stocks on first day): {predicted_returns[0, :5]}")

    # --- Monte Carlo Option Pricing Example ---
    print("\n=== Monte Carlo Option Pricing Example ===")
    n_options = 100000
    S = np.random.uniform(50, 200, n_options).astype(np.float32)
    K = S * np.random.uniform(0.8, 1.2, n_options).astype(np.float32)
    T = np.random.uniform(0.1, 2.0, n_options).astype(np.float32)
    r = np.ones(n_options).astype(np.float32) * 0.05
    sigma = np.random.uniform(0.1, 0.5, n_options).astype(np.float32)
    
    call_prices = finance_api.monte_carlo_option_price(S, K, T, r, sigma)
    
    print(f"Shape of call prices: {call_prices.shape}")
    print(f"Sample call prices (first 5 options): {call_prices[:5]}")

    # --- Risk Model Example ---
    print("\n=== Risk Model Example ===")
    n_assets = 10000
    n_factors = 50
    
    factor_exposures = np.random.randn(n_assets, n_factors).astype(np.float32)
    factor_cov = np.random.randn(n_factors, n_factors).astype(np.float32)
    factor_cov = factor_cov @ factor_cov.T  # Ensure positive semidefinite
    specific_risk = np.abs(np.random.randn(n_assets).astype(np.float32)) * 0.05
    
    risk_model = finance_api.risk_model(factor_exposures, factor_cov, specific_risk)
    
    print(f"Shape of risk model: {risk_model.shape}")
    print(f"Sample risk model (first 5x5): \n{risk_model[:5, :5]}")