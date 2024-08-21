import numpy as np
from scipy import stats

def heaviside_step(x):
    """Heaviside step function."""
    return np.where(x >= 0, 1.0, 0.0)

def crps_ensemble(forecasts, observation):
    """
    Calculate the Continuous Rank Probability Score (CRPS) for an ensemble forecast.
    
    :param forecasts: Array of forecast ensemble members
    :param observation: Actual observed value
    :return: CRPS score
    """
    n = len(forecasts)
    sorted_forecasts = np.sort(forecasts)
    
    # Calculate empirical CDF
    def empirical_cdf(x):
        return np.sum(heaviside_step(x - sorted_forecasts)) / n
    
    # Vectorize the empirical CDF function
    v_empirical_cdf = np.vectorize(empirical_cdf)
    
    # Calculate CRPS using numerical integration
    x = np.linspace(min(np.min(forecasts), observation) - 1, 
                    max(np.max(forecasts), observation) + 1, 
                    1000)
    integrand = (v_empirical_cdf(x) - heaviside_step(x - observation))**2
    crps = np.trapz(integrand, x)
    
    return crps

# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    forecasts = np.random.normal(10, 2, 100)  # 100 ensemble members
    observation = 9.5
    
    # Calculate CRPS
    score = crps_ensemble(forecasts, observation)
    print(f"CRPS Score: {score:.4f}")