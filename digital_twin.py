
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from tensorflow.keras.models import load_model
import joblib


# ============================================================
# TRANSFORMER DIGITAL TWIN
# ML CONNECTED VERSION
# ============================================================


# ============================================================
# 1. LOAD TRANSFORMER DATA
# ============================================================

data = pd.read_csv("transformer_data.csv")

data["timestamp"] = pd.to_datetime(
    data["timestamp"]
)


# ============================================================
# 2. LOAD TRAINED ML MODEL
# ============================================================

print("\nLoading ML model...")

model = load_model(
    "multivariate_lstm.keras"
)

scaler = joblib.load(
    "transformer_scaler.pkl"
)

print("ML model loaded successfully!")


# ============================================================
# 3. GET LATEST SENSOR READING
# ============================================================

latest = data.iloc[-1]

temperature = latest["temperature"]
current = latest["current"]
voltage = latest["voltage"]
humidity = latest["humidity"]
motion = latest["motion"]

timestamp = latest["timestamp"]


# ============================================================
# 4. CURRENT HEALTH SCORE
# ============================================================

def calculate_health(
    temperature,
    current,
    voltage,
    humidity
):

    # Temperature score

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


    # Current score

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


    # Voltage score

    if 220 <= voltage <= 240:

        voltage_score = 100

    elif 215 <= voltage < 220:

        voltage_score = 80

    elif 240 < voltage <= 245:

        voltage_score = 80

    else:

        voltage_score = 50


    # Humidity score

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


    # Weighted health

    health = (

        temperature_score * 0.40

        + current_score * 0.30

        + voltage_score * 0.15

        + humidity_score * 0.15

    )

    return round(
        health,
        2
    )


# ============================================================
# 5. STATUS
# ============================================================

def get_status(score):

    if score >= 80:

        return "HEALTHY"

    elif score >= 60:

        return "WARNING"

    else:

        return "CRITICAL"


health_score = calculate_health(

    temperature,
    current,
    voltage,
    humidity

)

status = get_status(
    health_score
)


# ============================================================
# 6. FUTURE TEMPERATURE FORECAST
# ============================================================

features = [
    "temperature",
    "current",
    "voltage",
    "humidity",
    "motion"
]


values = data[features].values

scaled_data = scaler.transform(
    values
)


sequence_length = 10


current_sequence = scaled_data[
    -sequence_length:
].reshape(
    1,
    sequence_length,
    len(features)
)


future_predictions = []


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


    # Create next row

    new_row = (
        current_sequence[0, -1, :]
        .copy()
    )

    new_row[0] = (
        predicted_temperature_scaled
    )


    # Shift sequence

    current_sequence = np.roll(
        current_sequence,
        -1,
        axis=1
    )


    current_sequence[
        0,
        -1,
        :
    ] = new_row


# ============================================================
# 7. CONVERT PREDICTIONS TO °C
# ============================================================

temperature_min = (
    scaler.data_min_[0]
)

temperature_max = (
    scaler.data_max_[0]
)


future_temperatures = []


for value in future_predictions:

    temperature_future = (

        value *
        (
            temperature_max -
            temperature_min
        )

        + temperature_min

    )

    future_temperatures.append(
        temperature_future
    )


# ============================================================
# 8. FUTURE HEALTH
# ============================================================

future_health = []

future_status = []


for future_temperature in future_temperatures:

    score = calculate_health(

        future_temperature,

        current,

        voltage,

        humidity

    )

    future_health.append(
        score
    )

    future_status.append(
        get_status(score)
    )


# ============================================================
# 9. PRINT DIGITAL TWIN INFORMATION
# ============================================================

print("\n")
print("=" * 65)
print("             TRANSFORMER DIGITAL TWIN")
print("=" * 65)


print("\nCURRENT TRANSFORMER STATE")
print("-------------------------")

print(
    f"Timestamp    : {timestamp}"
)

print(
    f"Temperature  : {temperature:.2f} °C"
)

print(
    f"Current      : {current:.2f} A"
)

print(
    f"Voltage      : {voltage:.2f} V"
)

print(
    f"Humidity     : {humidity:.2f} %"
)

print(
    f"Motion       : {motion}"
)

print(
    f"Health Score : {health_score:.2f}%"
)

print(
    f"Status       : {status}"
)


print("\nML FUTURE FORECAST")
print("------------------")


for i in range(10):

    print(

        f"Step {i+1:02d} | "

        f"Temperature: "
        f"{future_temperatures[i]:.2f} °C | "

        f"Health: "
        f"{future_health[i]:.2f}% | "

        f"{future_status[i]}"

    )


print("=" * 65)


# ============================================================
# 10. CREATE DIGITAL TWIN
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 8)
)


ax.set_xlim(
    0,
    14
)

ax.set_ylim(
    0,
    9
)

ax.axis(
    "off"
)


# ============================================================
# 11. TITLE
# ============================================================

ax.text(
    7,
    8.4,
    "TRANSFORMER DIGITAL TWIN",
    ha="center",
    fontsize=22,
    fontweight="bold"
)


ax.text(
    7,
    7.95,
    "Real-Time State + ML Future Forecast",
    ha="center",
    fontsize=12
)


# ============================================================
# 12. TRANSFORMER BODY
# ============================================================

transformer = Rectangle(
    (5, 2.5),
    4,
    4,
    linewidth=3,
    edgecolor="black",
    facecolor="lightgray"
)

ax.add_patch(
    transformer
)


# ============================================================
# 13. TRANSFORMER COILS
# ============================================================

for y in [
    3.2,
    3.8,
    4.4,
    5.0,
    5.6
]:

    ax.plot(
        [5.5, 8.5],
        [y, y],
        linewidth=3
    )


# ============================================================
# 14. TRANSFORMER LABEL
# ============================================================

ax.text(
    7,
    2.05,
    "PHYSICAL TRANSFORMER",
    ha="center",
    fontsize=15,
    fontweight="bold"
)


# ============================================================
# 15. TEMPERATURE
# ============================================================

ax.text(
    0.5,
    6.0,

    f"Temperature\n"
    f"{temperature:.2f} °C",

    fontsize=12,

    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black"
    )
)


# ============================================================
# 16. CURRENT
# ============================================================

ax.text(
    0.5,
    4.2,

    f"Current\n"
    f"{current:.2f} A",

    fontsize=12,

    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black"
    )
)


# ============================================================
# 17. VOLTAGE
# ============================================================

ax.text(
    10.2,
    6.0,

    f"Voltage\n"
    f"{voltage:.2f} V",

    fontsize=12,

    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black"
    )
)


# ============================================================
# 18. HUMIDITY
# ============================================================

ax.text(
    10.2,
    4.2,

    f"Humidity\n"
    f"{humidity:.2f} %",

    fontsize=12,

    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black"
    )
)


# ============================================================
# 19. HEALTH
# ============================================================

ax.text(
    7,
    0.8,

    f"CURRENT HEALTH: "
    f"{health_score:.1f}%   |   "
    f"{status}",

    ha="center",

    fontsize=15,

    fontweight="bold",

    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black"
    )
)


# ============================================================
# 20. FUTURE FORECAST BOX
# ============================================================

future_text = (

    "ML FORECAST\n"

    f"Next Temperature: "
    f"{future_temperatures[0]:.2f} °C\n"

    f"Next Health: "
    f"{future_health[0]:.2f}%\n"

    f"Predicted Status: "
    f"{future_status[0]}"

)


ax.text(
    10.0,
    2.0,

    future_text,

    fontsize=11,

    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black"
    )
)


# ============================================================
# 21. DISPLAY
# ============================================================

plt.tight_layout()

plt.show()
