import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

# Statsmodels diagnostics
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

# 1. Data Collection

start_date = "2025-02-10"
end_date = "2026-02-10"

walmart = yf.download("WMT", start=start_date, end=end_date, interval="1h")
walmart.to_csv("wmt_data.csv")

walmart_returns = np.log(walmart['Close'] / walmart['Close'].shift(1)).dropna()

print(f"WMT Hourly Log Returns: {len(walmart_returns)} observations")
print(walmart_returns.describe())

# 2. PRE-FIT DIAGNOSTICS

print("\n" + "="*80)
print("PRE-FIT DIAGNOSTICS")
print("="*80)

# 2.1 ADF Test (Stationarity)
adf_result = adfuller(walmart_returns)
print(f'ADF Statistic: {adf_result[0]:.6f}')
print(f'p-value: {adf_result[1]:.6f}')
print('Critical Values:', adf_result[4])

# 2.2 ACF Plot of Squared Returns (to visually detect ARCH effects)
plt.figure(figsize=(12, 6))
plot_acf(walmart_returns**2, lags=40, title='ACF of Squared Log Returns (WMT)')
plt.tight_layout()
plt.savefig('ACF_Squared_Returns_WMT.png', dpi=300)
plt.show()

# 2.3 Engle's ARCH-LM Test
arch_test = het_arch(walmart_returns, nlags=12)
print(f'\nEngle\'s ARCH-LM Test (lag=12):')
print(f'LM Statistic: {arch_test[0]:.4f}')
print(f'p-value: {arch_test[1]:.6f}')

# 3. GARCH(1,1) Model
model_walmart = arch_model(walmart_returns, vol='Garch', p=1, q=1, mean='Zero', dist='normal')
res_walmart = model_walmart.fit(disp='off', show_warning=False)

print("\n" + "="*80)
print("WMT GARCH(1,1) RESULTS")
print("="*80)
print(res_walmart.summary())

# 4. POST-FIT DIAGNOSTICS
print("\n" + "="*80)
print("POST-FIT DIAGNOSTICS")
print("="*80)

std_resid = res_walmart.std_resid

# 4.1 Ljung-Box on Standardized Residuals (should be white noise)
lb_resid = acorr_ljungbox(std_resid, lags=[10, 20], return_df=True)
print("Ljung-Box Test - Standardized Residuals:")
print(lb_resid)

# 4.2 Ljung-Box on Squared Standardized Residuals (most important!)
lb_sq_resid = acorr_ljungbox(std_resid**2, lags=[10, 20], return_df=True)
print("\nLjung-Box Test - Squared Standardized Residuals:")
print(lb_sq_resid)

# 5. Volatility Plot
plt.figure(figsize=(12, 6))
plt.plot(res_walmart.conditional_volatility)
plt.title('Walmart Conditional Volatility - GARCH(1,1)')
plt.ylabel('Volatility')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('WMT_Conditional_Volatility.png', dpi=300)
plt.show()

# Save full summary
with open("garch_full_results.txt", "w") as f:
    f.write(str(res_walmart.summary()))
    f.write("\n\n=== POST-FIT DIAGNOSTICS ===\n")
    f.write(f"Ljung-Box (std_resid):\n{str(lb_resid)}\n\n")
    f.write(f"Ljung-Box (std_resid²):\n{str(lb_sq_resid)}")