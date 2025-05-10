import numpy as np
import time
from kernels.finlib.finance.stocks_api import StocksAPI  # Correctly import StocksAPI

api = StocksAPI()
factor_exposures = np.random.randn(10000, 5).astype(np.float32)  # 10,000assets, 5 factors
factor_covariance_matrix = np.random.randn(5, 5).astype(np.float32)  # 5x5 factor covariance matrix
factor_covariance_matrix = factor_covariance_matrix @ factor_covariance_matrix.T # Make sure it is positive semi-definite
specific_risk = np.random.rand(10000).astype(np.float32)  # Specific risk for 10,000 assets

start_time = time.time()
covariance_matrix = api.risk_model(factor_exposures, factor_covariance_matrix, specific_risk)
end_time = time.time()

print(covariance_matrix.shape)  
print(f"Execution time: {end_time - start_time} seconds")
