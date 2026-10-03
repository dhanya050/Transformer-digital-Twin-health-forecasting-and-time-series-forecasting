import pandas as pd
import matplotlib.pyplot as plt

# Load health data
data = pd.read_csv("transformer_health_data.csv")

data["timestamp"] = pd.to_datetime(data["timestamp"])


# ==========================================
# HEALTH SCORE GRAPH
# ==========================================

plt.figure(figsize=(14, 6))

plt.plot(
    data["timestamp"],
    data["health_score"],
    label="Transformer Health Score"
)

# Health thresholds
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


plt.xlabel("Time")

plt.ylabel("Health Score")

plt.title(
    "Transformer Health Score Over Time"
)

plt.ylim(0, 105)

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()

plt.show()