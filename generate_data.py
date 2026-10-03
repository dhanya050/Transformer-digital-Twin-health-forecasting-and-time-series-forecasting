import pandas as pd
import numpy as np

np.random.seed(42)

# Generate 1000 time points
time = pd.date_range(
    start="2026-01-01",
    periods=1000,
    freq="10min"
)

# Simulated transformer temperature
temperature = (
    45
    + 5 * np.sin(np.arange(1000) / 50)
    + np.linspace(0, 10, 1000)
    + np.random.normal(0, 0.8, 1000)
)

data = pd.DataFrame({
    "timestamp": time,
    "temperature": temperature
})

data.to_csv("temperature_data.csv", index=False)

print(data.head())
print("\nDataset created successfully!")