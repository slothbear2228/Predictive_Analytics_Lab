import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Simulated historical daily temperatures (°C)
# You can replace this with real historical weather data (CSV)
data = {
    'day': range(1, 31),  # Days 1 to 30
    'temperature': [
        22, 21, 23, 24, 25, 23, 22, 21, 22, 24,
        26, 25, 27, 28, 29, 27, 26, 25, 24, 23,
        22, 23, 24, 25, 26, 27, 26, 25, 24, 23
    ]
}

df = pd.DataFrame(data)

# Step 2: Create features - previous day temperature to predict next day
df['temp_prev_day'] = df['temperature'].shift(1)
df.dropna(inplace=True)  # Drop first row with NaN

X = df[['temp_prev_day']]  # Feature: temp of previous day
y = df['temperature']      # Target: temp today

# Step 3: Train-test split (last 5 days for testing)
train_size = len(df) - 5
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 4: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on test set
y_pred = model.predict(X_test)

# Step 6: Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Step 7: Plot results
plt.plot(df['day'][train_size:], y_test, label='Actual Temp')
plt.plot(df['day'][train_size:], y_pred, label='Predicted Temp', linestyle='--')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('Weather Forecasting - Temperature Prediction')
plt.legend()
plt.show()
