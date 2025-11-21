import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Sample historical cash flow data (monthly cash flow in thousands)
data = {
    'Month': pd.date_range(start='2023-01-01', periods=24, freq='M'),
    'CashFlow': [
        120, 130, 125, 140, 135, 150, 155, 160, 170, 165, 175, 180,
        185, 190, 195, 200, 210, 215, 220, 225, 230, 240, 245, 250
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)
df.set_index('Month', inplace=True)

# Plot historical data
df.plot(title="Historical Monthly Cash Flow", ylabel="Cash Flow (in thousands)")
plt.show()

# Fit the Exponential Smoothing model
model = ExponentialSmoothing(df['CashFlow'], trend='add', seasonal=None, initialization_method='estimated')
fit = model.fit()

# Forecast the next 6 months
forecast = fit.forecast(6)
print("Forecasted Cash Flow for next 6 months:")
print(forecast)

# Plot the historical data and forecast
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['CashFlow'], label='Historical Cash Flow')
plt.plot(forecast.index, forecast, label='Forecasted Cash Flow', marker='o')
plt.title('Cash Flow Forecast')
plt.xlabel('Month')
plt.ylabel('Cash Flow (in thousands)')
plt.legend()
plt.show()
