from finlib.APIS.finance_api import StocksAPI
import time  # Import the time module

api = StocksAPI()
num_paths = 20000000
num_time_steps = 10
initial_price = 100.0
risk_free_rate = 0.05
volatility = 0.2
kappa = 1.5
theta = 0.04
sigma = 0.3
rho = -0.7

start_time = time.time()  # Start the timer
simulated_prices = api.simulate_heston_model(num_paths, num_time_steps, initial_price, risk_free_rate, volatility, kappa, theta, sigma, rho)
end_time = time.time()  # End the timer

print(simulated_prices.shape)
print(f"Simulation time: {end_time - start_time} seconds")