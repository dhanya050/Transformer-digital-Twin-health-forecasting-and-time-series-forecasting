
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model


# ============================================================
# TRANSFORMER FUTURE HEALTH FORECAST
# ============================================================


# ============================================================
# 1. LOAD TRANSFORMER DATA
# ============================================================

data = pd.read_csv("transformer_data.csv")

data["timestamp"] = pd.to_datetime(
    data["timestamp"]
)


# ============================================================
# 2. FEATURES
# ============================================================

features = [
    "temperature",
    "current",
    "voltage",
    "humidity",
    "motion"
]


# ============================================================
# 3. LOAD TRAINED LSTM MODEL
# ============================================================

print("\nLoading trained LSTM model...")

model = load_model(
    "multivariate_lstm.keras"
)

print("LSTM model loaded successfully!")


# ============================================================
# 4. LOAD SCALER
# ============================================================

scaler = joblib.load(
    "transformer_scaler.pkl"
)

print("Scaler loaded successfully!")


# ============================================================
# 5. PREPARE DATA
# ============================================================

values = data[features].values

scaled_data = scaler.transform(
    values
)


# ============================================================
# 6. GET LATEST 10 SENSOR READINGS
# ============================================================

sequence_length = 10

latest_sequence = scaled_data[
    -sequence_length:
]

latest_sequence = latest_sequence.reshape(
    1,
    sequence_length,
    len(features)
)


# ============================================================
# 7. FORECAST FUTURE TEMPERATURE
# ============================================================

future_predictions = []

current_sequence = latest_sequence.copy()


print("\nGenerating future predictions...")


for step in range(10):

    prediction = model.predict(
        current_sequence,
        verbose=0
    )

    predicted_temperature_scaled = (
        prediction[0][0]
    )

    future_predictions.append(
        predicted_temperature_scaled
    )


    # ----------------------------------------
    # Create next input row
    # ----------------------------------------

    new_row = (
        current_sequence[0, -1, :]
        .copy()
    )

    # Replace temperature with
    # predicted temperature

    new_row[0] = (
        predicted_temperature_scaled
    )


    # ----------------------------------------
    # Shift sequence
    # ----------------------------------------

    current_sequence = np.roll(
        current_sequence,
        -1,
        axis=1
    )


    # Add predicted row

    current_sequence[
        0,
        -1,
        :
    ] = new_row


# ============================================================
# 8. CONVERT TEMPERATURE BACK TO °C
# ============================================================

temperature_min = (
    scaler.data_min_[0]
)

temperature_max = (
    scaler.data_max_[0]
)


future_temperatures = []


for value in future_predictions:

    temperature = (

        value *
        (
            temperature_max -
            temperature_min
        )

        + temperature_min

    )

    future_temperatures.append(
        temperature
    )


# ============================================================
# 9. HEALTH SCORE FUNCTION
# ============================================================

def calculate_health(
    temperature,
    current,
    voltage,
    humidity
):

    # ----------------------------------------
    # Temperature
    # ----------------------------------------

    if temperature <= 50:

        temperature_score = 100

    elif temperature <= 60:

        temperature_score = (
            100 -
            (temperature - 50) * 5
        )

    else:

        temperature_score = max(
            0,
            50 -
            (temperature - 60) * 5
        )


    # ----------------------------------------
    # Current
    # ----------------------------------------

    if current <= 2.5:

        current_score = 100

    elif current <= 3.0:

        current_score = (
            100 -
            (current - 2.5) * 100
        )

    else:

        current_score = max(
            0,
            50 -
            (current - 3.0) * 50
        )


    # ----------------------------------------
    # Voltage
    # ----------------------------------------

    if 220 <= voltage <= 240:

        voltage_score = 100

    elif 215 <= voltage < 220:

        voltage_score = 80

    elif 240 < voltage <= 245:

        voltage_score = 80

    else:

        voltage_score = 50


    # ----------------------------------------
    # Humidity
    # ----------------------------------------

    if humidity <= 65:

        humidity_score = 100

    elif humidity <= 75:

        humidity_score = (
            100 -
            (humidity - 65) * 5
        )

    else:

        humidity_score = max(
            0,
            50 -
            (humidity - 75) * 5
        )


    # ----------------------------------------
    # Weighted health score
    # ----------------------------------------

    health_score = (

        temperature_score * 0.40

        + current_score * 0.30

        + voltage_score * 0.15

        + humidity_score * 0.15

    )


    return round(
        health_score,
        2
    )


# ============================================================
# 10. CURRENT CONDITIONS
# ============================================================

latest = data.iloc[-1]


current_temperature = (
    latest["temperature"]
)

current_current = (
    latest["current"]
)

current_voltage = (
    latest["voltage"]
)

current_humidity = (
    latest["humidity"]
)


# ============================================================
# 11. CURRENT HEALTH
# ============================================================

current_health = calculate_health(

    current_temperature,

    current_current,

    current_voltage,

    current_humidity

)


# ============================================================
# 12. STATUS FUNCTION
# ============================================================

def get_status(score):

    if score >= 80:

        return "HEALTHY"

    elif score >= 60:

        return "WARNING"

    else:

        return "CRITICAL"


current_status = get_status(
    current_health
)


# ============================================================
# 13. FUTURE HEALTH
# ============================================================

future_health = []

future_status = []


for temperature in future_temperatures:

    score = calculate_health(

        temperature,

        current_current,

        current_voltage,

        current_humidity

    )

    future_health.append(
        score
    )

    future_status.append(
        get_status(score)
    )


# ============================================================
# 14. PRINT CURRENT CONDITIONS
# ============================================================

print("\n")
print("=" * 60)
print(
    "       TRANSFORMER FUTURE HEALTH FORECAST"
)
print("=" * 60)


print("\nCurrent Conditions")
print("------------------")


print(
    f"Temperature : "
    f"{current_temperature:.2f} °C"
)

print(
    f"Current     : "
    f"{current_current:.2f} A"
)

print(
    f"Voltage     : "
    f"{current_voltage:.2f} V"
)

print(
    f"Humidity    : "
    f"{current_humidity:.2f} %"
)

print(
    f"Health      : "
    f"{current_health:.2f}"
)

print(
    f"Status      : "
    f"{current_status}"
)


# ============================================================
# 15. PRINT FUTURE PREDICTIONS
# ============================================================

print("\nFuture Predictions")
print("------------------")


for i in range(10):

    print(

        f"Step {i+1:02d}  | "

        f"Temperature: "
        f"{future_temperatures[i]:.2f} °C | "

        f"Health: "
        f"{future_health[i]:.2f} | "

        f"Status: "
        f"{future_status[i]}"

    )


print("\n")
print("=" * 60)


# ============================================================
# 16. FUTURE TEMPERATURE GRAPH
# ============================================================

steps = np.arange(
    1,
    11
)


plt.figure(
    figsize=(12, 5)
)


plt.plot(
    steps,
    future_temperatures,
    marker="o",
    label="Predicted Temperature"
)


plt.axhline(
    50,
    linestyle="--",
    label="Temperature Warning Level"
)


plt.xlabel(
    "Future Time Step"
)

plt.ylabel(
    "Temperature (°C)"
)

plt.title(
    "Transformer Future Temperature Forecast"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 17. FUTURE HEALTH GRAPH
# ============================================================

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    steps,
    future_health,
    marker="o",
    label="Predicted Health"
)


plt.axhline(
    80,
    linestyle="--",
    label="Healthy Threshold"
)


plt.axhline(
    60,
    linestyle="--",
    label="Warning Threshold"
)


plt.xlabel(
    "Future Time Step"
)

plt.ylabel(
    "Health Score"
)

plt.title(
    "Transformer Future Health Forecast"
)

plt.ylim(
    0,
    105
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()
