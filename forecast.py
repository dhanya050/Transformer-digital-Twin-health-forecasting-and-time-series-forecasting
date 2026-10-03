import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# 1. Load data
data = pd.read_csv("temperature_data.csv")

data["timestamp"] = pd.to_datetime(data["timestamp"])

temperature = data["temperature"]


# 2. Split data into training and testing
train_size = int(len(temperature) * 0.8)

train = temperature[:train_size]
test = temperature[train_size:]


print("Total data:", len(temperature))
print("Training data:", len(train))
print("Testing data:", len(test))


# 3. Create ARIMA model
model = ARIMA(train, order=(1, 1, 1))


# 4. Train model
model_fit = model.fit()


# 5. Predict the test period
predictions = model_fit.forecast(steps=len(test))


# 6. Calculate errors
mae = mean_absolute_error(test, predictions)

rmse = np.sqrt(
    mean_squared_error(test, predictions)
)


print("\nModel Performance")
print("----------------------")
print("MAE :", mae)
print("RMSE:", rmse)


# 7. Plot actual vs predicted
plt.figure(figsize=(12, 5))

plt.plot(
    test.index,
    test,
    label="Actual"
)

plt.plot(
    test.index,
    predictions,
    label="Predicted"
)

plt.xlabel("Time")
plt.ylabel("Temperature (°C)")

plt.title("ARIMA Temperature Forecast")

plt.legend()

plt.tight_layout()

plt.show()