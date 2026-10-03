import pandas as pd
import matplotlib.pyplot as plt

# Load transformer data
data = pd.read_csv("transformer_data.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"])

# ------------------------------------------------
# These are the predictions from your LSTM output
# ------------------------------------------------

future_temperatures = [
    48.77,
    48.65,
    48.60,
    48.49,
    48.49,
    48.56,
    48.54,
    48.63,
    48.71,
    48.78
]

# Create future timestamps
last_time = data["timestamp"].iloc[-1]

future_times = pd.date_range(
    start=last_time + pd.Timedelta(minutes=10),
    periods=10,
    freq="10min"
)

# ------------------------------------------------
# Plot actual + predicted temperature
# ------------------------------------------------

plt.figure(figsize=(14, 6))

# Last 100 actual readings
recent_data = data.tail(100)

plt.plot(
    recent_data["timestamp"],
    recent_data["temperature"],
    label="Actual Temperature"
)

# Future prediction
plt.plot(
    future_times,
    future_temperatures,
    marker="o",
    linestyle="--",
    label="LSTM Forecast"
)

plt.xlabel("Time")
plt.ylabel("Temperature (°C)")

plt.title(
    "Transformer Temperature - Actual vs Forecast"
)

plt.xticks(rotation=45)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()