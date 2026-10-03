import matplotlib.pyplot as plt

# Future health values from our forecast
future_health = [
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00
]

steps = list(range(1, 11))

plt.figure(figsize=(12, 5))

plt.plot(
    steps,
    future_health,
    marker="o",
    linewidth=2,
    label="Predicted Health"
)

# Health thresholds
plt.axhline(
    80,
    linestyle="--",
    label="Healthy threshold"
)

plt.axhline(
    60,
    linestyle="--",
    label="Warning threshold"
)

plt.xlabel("Future Time Step")
plt.ylabel("Health Score")

plt.title(
    "Transformer Future Health Forecast"
)

plt.ylim(0, 105)

plt.xticks(steps)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()