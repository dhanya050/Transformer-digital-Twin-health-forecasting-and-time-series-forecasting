import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Transformer Digital Twin",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("⚡ TRANSFORMER DIGITAL TWIN")
st.subheader("Transformer Health Forecasting with Digital Twin & Time Series Forecasting")

st.write(
    "This dashboard represents the current operating condition "
    "of the transformer using sensor data and machine learning."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv("transformer_data.csv")

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    data = data.sort_values("timestamp")

    return data


data = load_data()


# ============================================================
# LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_ml_model():

    model = load_model("multivariate_lstm.keras")

    scaler = joblib.load("transformer_scaler.pkl")

    return model, scaler


model, scaler = load_ml_model()


# ============================================================
# GET LATEST SENSOR DATA
# ============================================================

latest = data.iloc[-1]

temperature = latest["temperature"]
current = latest["current"]
voltage = latest["voltage"]
humidity = latest["humidity"]
motion = latest["motion"]
timestamp = latest["timestamp"]


# ============================================================
# HEALTH SCORE CALCULATION
# ============================================================

def calculate_health(temperature, current, voltage, humidity):

    # Temperature score
    if temperature <= 50:
        temp_score = 100

    elif temperature <= 60:
        temp_score = 100 - (temperature - 50) * 5

    else:
        temp_score = max(
            0,
            50 - (temperature - 60) * 5
        )


    # Current score
    if current <= 2.5:
        current_score = 100

    elif current <= 3.0:
        current_score = 100 - (current - 2.5) * 100

    else:
        current_score = max(
            0,
            50 - (current - 3.0) * 50
        )


    # Voltage score
    if 220 <= voltage <= 240:
        voltage_score = 100

    elif 215 <= voltage < 220 or 240 < voltage <= 245:
        voltage_score = 80

    else:
        voltage_score = 50


    # Humidity score
    if humidity <= 65:
        humidity_score = 100

    elif humidity <= 75:
        humidity_score = 100 - (humidity - 65) * 5

    else:
        humidity_score = max(
            0,
            50 - (humidity - 75) * 5
        )


    # Weighted health score
    health_score = (
        temp_score * 0.40 +
        current_score * 0.30 +
        voltage_score * 0.15 +
        humidity_score * 0.15
    )

    return round(health_score, 2)


# ============================================================
# HEALTH STATUS
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

status = get_status(health_score)


# ============================================================
# SENSOR CARDS
# ============================================================

st.header("📡 Real-Time Transformer Parameters")

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "🌡 Temperature",
        f"{temperature:.2f} °C"
    )


with col2:

    st.metric(
        "⚡ Current",
        f"{current:.2f} A"
    )


with col3:

    st.metric(
        "🔌 Voltage",
        f"{voltage:.2f} V"
    )


with col4:

    st.metric(
        "💧 Humidity",
        f"{humidity:.2f} %"
    )


with col5:

    motion_text = "Detected" if motion == 1 else "Normal"

    st.metric(
        "🚨 Motion",
        motion_text
    )


# ============================================================
# HEALTH SECTION
# ============================================================

st.divider()

st.header("❤️ Transformer Health")


col1, col2 = st.columns([1, 2])


with col1:

    st.metric(
        "Overall Health Score",
        f"{health_score}%"
    )

    if status == "HEALTHY":

        st.success("STATUS: HEALTHY")

    elif status == "WARNING":

        st.warning("STATUS: WARNING")

    else:

        st.error("STATUS: CRITICAL")


with col2:

    st.write("### Health Index")

    st.progress(
        int(health_score)
    )


# ============================================================
# DIGITAL TWIN
# ============================================================

st.divider()

st.header("🖥️ Digital Twin")


fig, ax = plt.subplots(figsize=(10, 5))

ax.set_xlim(0, 10)
ax.set_ylim(0, 6)

ax.axis("off")


# Transformer body

transformer_x = 3
transformer_y = 1.5
transformer_width = 4
transformer_height = 3


rectangle = plt.Rectangle(
    (transformer_x, transformer_y),
    transformer_width,
    transformer_height,
    fill=False,
    linewidth=3
)

ax.add_patch(rectangle)


# Transformer coils

for y in np.linspace(2, 4, 6):

    ax.plot(
        [4, 5],
        [y, y],
        linewidth=2
    )

    ax.plot(
        [5.2, 6.2],
        [y, y],
        linewidth=2
    )


# Input / Output lines

ax.plot([1.5, 3], [3, 3], linewidth=3)

ax.plot([7, 8.5], [3, 3], linewidth=3)


ax.text(
    1.0,
    3,
    "INPUT",
    fontsize=12,
    ha="center"
)

ax.text(
    9,
    3,
    "OUTPUT",
    fontsize=12,
    ha="center"
)


# Transformer label

ax.text(
    5,
    5,
    "TRANSFORMER",
    fontsize=16,
    ha="center",
    fontweight="bold"
)


# Current state

ax.text(
    5,
    0.7,
    f"Health: {health_score}%   |   Status: {status}",
    fontsize=12,
    ha="center"
)


st.pyplot(fig)


# ============================================================
# HISTORICAL SENSOR DATA
# ============================================================

st.divider()

st.header("📈 Sensor History")


sensor_option = st.selectbox(
    "Select parameter",
    [
        "temperature",
        "current",
        "voltage",
        "humidity"
    ]
)


chart_data = data.set_index("timestamp")[
    [sensor_option]
]


st.line_chart(chart_data)


# ============================================================
# CURRENT STATE INFORMATION
# ============================================================

st.divider()

st.header("📋 Current Transformer State")


state_data = pd.DataFrame({

    "Parameter": [
        "Temperature",
        "Current",
        "Voltage",
        "Humidity",
        "Motion",
        "Health Score",
        "Status"
    ],

    "Value": [
        f"{temperature:.2f} °C",
        f"{current:.2f} A",
        f"{voltage:.2f} V",
        f"{humidity:.2f} %",
        "Detected" if motion == 1 else "Normal",
        f"{health_score} %",
        status
    ]

})


st.table(state_data)


# ============================================================
# TIMESTAMP
# ============================================================

st.caption(
    f"Latest sensor reading: {timestamp}"
)

st.caption(
    "Transformer Digital Twin | Major Project"
)