import pandas as pd
import numpy as np

np.random.seed(42)

# Number of readings
n = 2000

# Time
time = pd.date_range(
    start="2026-01-01",
    periods=n,
    freq="10min"
)

# ==========================================
# BASE PARAMETERS
# ==========================================

current = (
    2.0
    + 0.4 * np.sin(np.arange(n) / 50)
    + np.random.normal(0, 0.08, n)
)

current = np.maximum(current, 0.5)


voltage = (
    230
    + 3 * np.sin(np.arange(n) / 80)
    + np.random.normal(0, 1.2, n)
)


humidity = (
    55
    + 5 * np.sin(np.arange(n) / 100)
    + np.random.normal(0, 1.5, n)
)

humidity = np.clip(humidity, 30, 80)


# ==========================================
# TEMPERATURE
# ==========================================

temperature = (
    40
    + (current - 2.0) * 12
    + 3 * np.sin(np.arange(n) / 70)
    + np.random.normal(0, 0.7, n)
)

temperature += np.linspace(0, 5, n)


# ==========================================
# ADD ABNORMAL CONDITIONS
# ==========================================

# ------------------------------------------
# WARNING CONDITION
# Readings 1000 - 1300
# ------------------------------------------

current[1000:1300] += 0.8

temperature[1000:1300] += 8

humidity[1000:1300] += 8


# ------------------------------------------
# CRITICAL CONDITION
# Readings 1600 - 1800
# ------------------------------------------

current[1600:1800] += 1.5

temperature[1600:1800] += 20

voltage[1600:1800] -= 15

humidity[1600:1800] += 15


# Keep values within realistic ranges
humidity = np.clip(humidity, 30, 90)


# ==========================================
# MOTION
# ==========================================

motion = np.random.choice(
    [0, 1],
    size=n,
    p=[0.95, 0.05]
)

# More motion events during abnormal period
motion[1000:1300] = np.random.choice(
    [0, 1],
    size=300,
    p=[0.85, 0.15]
)

motion[1600:1800] = np.random.choice(
    [0, 1],
    size=200,
    p=[0.75, 0.25]
)


# ==========================================
# CREATE DATAFRAME
# ==========================================

data = pd.DataFrame({
    "timestamp": time,
    "temperature": temperature,
    "current": current,
    "voltage": voltage,
    "humidity": humidity,
    "motion": motion
})


# ==========================================
# SAVE
# ==========================================

data.to_csv(
    "transformer_data.csv",
    index=False
)


print("\nTransformer dataset created successfully!")

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nBasic statistics:")
print(data.describe())