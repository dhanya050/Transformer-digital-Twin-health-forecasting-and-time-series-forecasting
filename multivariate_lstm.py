
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# ==========================================
# 1. LOAD TRANSFORMER DATA
# ==========================================

data = pd.read_csv("transformer_data.csv")

data["timestamp"] = pd.to_datetime(data["timestamp"])


# ==========================================
# 2. SELECT INPUT FEATURES
# ==========================================

features = [
    "temperature",
    "current",
    "voltage",
    "humidity",
    "motion"
]

values = data[features].values


# ==========================================
# 3. NORMALIZE DATA
# ==========================================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(values)


# ==========================================
# 4. CREATE TIME SEQUENCES
# ==========================================

sequence_length = 10

X = []
y = []

for i in range(sequence_length, len(scaled_data)):

    # Previous 10 sensor readings
    X.append(
        scaled_data[i-sequence_length:i]
    )

    # Predict temperature
    y.append(
        scaled_data[i, 0]
    )


X = np.array(X)
y = np.array(y)


print("\nDataset prepared")
print("----------------")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 6. BUILD LSTM MODEL
# ==========================================

model = Sequential()

model.add(
    LSTM(
        64,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(1)
)


model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


print("\nModel structure:")
model.summary()


# ==========================================
# 7. TRAIN MODEL
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# ==========================================
# 8. MAKE TEST PREDICTIONS
# ==========================================

predicted_scaled = model.predict(X_test)


# ==========================================
# 9. CONVERT TEMPERATURE BACK TO °C
# ==========================================

temperature_min = scaler.data_min_[0]
temperature_max = scaler.data_max_[0]

predicted = (
    predicted_scaled *
    (temperature_max - temperature_min)
    + temperature_min
)

actual = (
    y_test *
    (temperature_max - temperature_min)
    + temperature_min
)


# ==========================================
# 10. CALCULATE MODEL PERFORMANCE
# ==========================================

mae = mean_absolute_error(
    actual,
    predicted
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted
    )
)


print("\nMultivariate LSTM Performance")
print("--------------------------------")
print("MAE :", mae)
print("RMSE:", rmse)


# ==========================================
# 11. SAVE TRAINED MODEL
# ==========================================

model.save(
    "multivariate_lstm.keras"
)

print(
    "\nLSTM model saved successfully!"
)

print(
    "File: multivariate_lstm.keras"
)


# ==========================================
# 12. SAVE SCALER
# ==========================================

joblib.dump(
    scaler,
    "transformer_scaler.pkl"
)

print(
    "Scaler saved successfully!"
)

print(
    "File: transformer_scaler.pkl"
)


# ==========================================
# 13. PLOT ACTUAL VS PREDICTED
# ==========================================

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

plt.title(
    "Multivariate LSTM Transformer Temperature Forecast"
)

plt.legend()

plt.tight_layout()

plt.show()
