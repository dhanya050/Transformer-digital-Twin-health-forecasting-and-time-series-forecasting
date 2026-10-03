
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SIMULATED TRANSFORMER DETERIORATION TEST
# ============================================================

temperature = np.array([
    48, 48.5, 49, 49.5, 50,
    51, 52, 53, 54, 55,
    56, 57, 58, 59, 60,
    61, 62, 63, 64, 65
])

current = np.array([
    2.20, 2.20, 2.25, 2.30, 2.30,
    2.35, 2.40, 2.45, 2.50, 2.55,
    2.60, 2.65, 2.70, 2.75, 2.80,
    2.85, 2.90, 2.95, 3.00, 3.05
])

voltage = np.array([
    230, 230, 230, 229, 229,
    229, 228, 228, 228, 227,
    227, 227, 226, 226, 225,
    225, 224, 224, 223, 223
])

humidity = np.array([
    58, 58, 59, 59, 60,
    60, 61, 61, 62, 62,
    63, 64, 64, 65, 66,
    67, 68, 69, 70, 71
])


# ============================================================
# CHECK DATA LENGTHS
# ============================================================

if not (
    len(temperature)
    == len(current)
    == len(voltage)
    == len(humidity)
):
    raise ValueError(
        "Sensor arrays must have the same number of values."
    )


# ============================================================
# HEALTH SCORE FUNCTION
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
            100 - (temperature - 50) * 5
        )

    else:
        temperature_score = max(
            0,
            50 - (temperature - 60) * 5
        )

    # Current score
    if current <= 2.5:
        current_score = 100

    elif current <= 3.0:
        current_score = (
            100 - (current - 2.5) * 100
        )

    else:
        current_score = max(
            0,
            50 - (current - 3.0) * 50
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
            100 - (humidity - 65) * 5
        )

    else:
        humidity_score = max(
            0,
            50 - (humidity - 75) * 5
        )

    # Weighted health score
    health = (
        temperature_score * 0.40
        + current_score * 0.30
        + voltage_score * 0.15
        + humidity_score * 0.15
    )

    return round(health, 2)


# ============================================================
# CALCULATE HEALTH
# ============================================================

health = []

for i in range(len(temperature)):

    score = calculate_health(
        temperature[i],
        current[i],
        voltage[i],
        humidity[i]
    )

    health.append(score)


# ============================================================
# DETERMINE STATUS
# ============================================================

def get_status(score):

    if score >= 80:
        return "HEALTHY"

    elif score >= 60:
        return "WARNING"

    else:
        return "CRITICAL"


status = [
    get_status(score)
    for score in health
]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("           TRANSFORMER DETERIORATION TEST")
print("=" * 70)

print()

for i in range(len(health)):

    print(
        f"Step {i + 1:02d} | "
        f"Temperature: {temperature[i]:5.1f} °C | "
        f"Current: {current[i]:4.2f} A | "
        f"Voltage: {voltage[i]:3.0f} V | "
        f"Humidity: {humidity[i]:4.1f}% | "
        f"Health: {health[i]:6.2f} | "
        f"Status: {status[i]}"
    )


# ============================================================
# STATUS SUMMARY
# ============================================================

healthy_count = status.count("HEALTHY")
warning_count = status.count("WARNING")
critical_count = status.count("CRITICAL")

print()
print("=" * 70)
print("                    STATUS SUMMARY")
print("=" * 70)

print(f"HEALTHY  : {healthy_count}")
print(f"WARNING  : {warning_count}")
print(f"CRITICAL : {critical_count}")


# ============================================================
# TEMPERATURE GRAPH
# ============================================================

steps = np.arange(
    1,
    len(temperature) + 1
)

plt.figure(figsize=(12, 5))

plt.plot(
    steps,
    temperature,
    marker="o",
    linewidth=2
)

plt.xlabel("Time Step")
plt.ylabel("Temperature (°C)")
plt.title("Transformer Temperature Deterioration")

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# HEALTH SCORE GRAPH
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    steps,
    health,
    marker="o",
    linewidth=2,
    label="Health Score"
)

plt.axhline(
    80,
    linestyle="--",
    label="Healthy Threshold (80)"
)

plt.axhline(
    60,
    linestyle="--",
    label="Warning Threshold (60)"
)

plt.xlabel("Time Step")
plt.ylabel("Health Score")
plt.title("Transformer Health During Deterioration")

plt.ylim(0, 105)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)

if critical_count > 0:
    print("RESULT: CRITICAL CONDITION DETECTED")

elif warning_count > 0:
    print("RESULT: WARNING CONDITION DETECTED")

else:
    print("RESULT: TRANSFORMER IS HEALTHY")

print("=" * 70)
