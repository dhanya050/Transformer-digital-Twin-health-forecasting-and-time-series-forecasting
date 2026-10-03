import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# 1. Load the dataset
data = pd.read_csv("temperature_data.csv")

temperature = data["temperature"].values.reshape(-1, 1)


# 2. Normalize the temperature
scaler = MinMaxScaler()

temperature_scaled = scaler.fit_transform(temperature)


# 3. Create sequences
sequence_length = 10

X = []
y = []

for i in range(sequence_length, len(temperature_scaled)):

    X.append(
        temperature_scaled[i-sequence_length:i]
    )

    y.append(
        temperature_scaled[i]
    )


X = np.array(X)
y = np.array(y)


print("X shape:", X.shape)
print("y shape:", y.shape)

# 4. Split data into training and testing

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Create LSTM model

model = Sequential()

model.add(
    LSTM(
        50,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model.summary()

# 6. Train the model

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 7. Make predictions

predicted_scaled = model.predict(X_test)

# Convert predictions back to actual temperature
predicted = scaler.inverse_transform(predicted_scaled)

actual = scaler.inverse_transform(y_test)

# 8. Calculate errors

mae = mean_absolute_error(actual, predicted)

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

print("\nLSTM Model Performance")
print("----------------------")
print("MAE :", mae)
print("RMSE:", rmse)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.plot(
    actual,
    label="Actual Temperature"
)

plt.plot(
    predicted,
    label="LSTM Predicted Temperature"
)

plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("LSTM Transformer Temperature Forecast")

plt.legend()

plt.tight_layout()

plt.show()