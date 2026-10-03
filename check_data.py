import pandas as pd
import matplotlib.pyplot as plt

# Load transformer data
data = pd.read_csv("transformer_data.csv")

data["timestamp"] = pd.to_datetime(data["timestamp"])


# -------------------------------
# Temperature
# -------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    data["timestamp"],
    data["temperature"]
)

plt.title("Transformer Temperature")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# -------------------------------
# Current
# -------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    data["timestamp"],
    data["current"]
)

plt.title("Transformer Current")
plt.xlabel("Time")
plt.ylabel("Current (A)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# -------------------------------
# Voltage
# -------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    data["timestamp"],
    data["voltage"]
)

plt.title("Transformer Voltage")
plt.xlabel("Time")
plt.ylabel("Voltage (V)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# -------------------------------
# Humidity
# -------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    data["timestamp"],
    data["humidity"]
)

plt.title("Transformer Humidity")
plt.xlabel("Time")
plt.ylabel("Humidity (%)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()