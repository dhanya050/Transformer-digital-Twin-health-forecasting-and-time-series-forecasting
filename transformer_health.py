import pandas as pd
import numpy as np


# ==========================================
# 1. LOAD TRANSFORMER DATA
# ==========================================

data = pd.read_csv("transformer_data.csv")


# ==========================================
# 2. HEALTH SCORE FUNCTION
# ==========================================

def calculate_health(row):

    temperature = row["temperature"]
    current = row["current"]
    voltage = row["voltage"]
    humidity = row["humidity"]

    score = 100


    # --------------------------------------
    # TEMPERATURE
    # --------------------------------------

    if temperature <= 50:
        temperature_score = 100

    elif temperature <= 60:
        temperature_score = 100 - (
            (temperature - 50) * 5
        )

    else:
        temperature_score = max(
            0,
            50 - (temperature - 60) * 5
        )


    # --------------------------------------
    # CURRENT
    # --------------------------------------

    if current <= 2.5:
        current_score = 100

    elif current <= 3.0:
        current_score = 100 - (
            (current - 2.5) * 100
        )

    else:
        current_score = max(
            0,
            50 - (current - 3.0) * 50
        )


    # --------------------------------------
    # VOLTAGE
    # --------------------------------------

    if 220 <= voltage <= 240:

        voltage_score = 100

    elif 215 <= voltage < 220:

        voltage_score = 80

    elif 240 < voltage <= 245:

        voltage_score = 80

    else:

        voltage_score = 50


    # --------------------------------------
    # HUMIDITY
    # --------------------------------------

    if humidity <= 65:

        humidity_score = 100

    elif humidity <= 75:

        humidity_score = 100 - (
            (humidity - 65) * 5
        )

    else:

        humidity_score = max(
            0,
            50 - (humidity - 75) * 5
        )


    # ======================================
    # WEIGHTED HEALTH SCORE
    # ======================================

    score = (
        temperature_score * 0.40
        + current_score * 0.30
        + voltage_score * 0.15
        + humidity_score * 0.15
    )


    return round(score, 2)


# ==========================================
# 3. CALCULATE HEALTH
# ==========================================

data["health_score"] = data.apply(
    calculate_health,
    axis=1
)


# ==========================================
# 4. DETERMINE STATUS
# ==========================================

def get_status(score):

    if score >= 80:
        return "HEALTHY"

    elif score >= 60:
        return "WARNING"

    else:
        return "CRITICAL"


data["status"] = data["health_score"].apply(
    get_status
)


# ==========================================
# 5. DISPLAY RESULTS
# ==========================================

print("\nTransformer Health Results")
print("==========================\n")

print(
    data[
        [
            "timestamp",
            "temperature",
            "current",
            "voltage",
            "humidity",
            "health_score",
            "status"
        ]
    ].head(20)
)


# ==========================================
# 6. SAVE HEALTH DATA
# ==========================================

data.to_csv(
    "transformer_health_data.csv",
    index=False
)


print("\nHealth dataset saved successfully!")

print("\nHealth status counts:")
print(data["status"].value_counts())