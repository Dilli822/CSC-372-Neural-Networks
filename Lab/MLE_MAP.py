import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# Generate synthetic dataset
np.random.seed(42)
X = np.linspace(0, 10, 100)
true_slope = 2
y = true_slope * X + np.random.normal(0, 3, 100)  # y = 2x + noise

# Regularized Least-Squares (Ridge Regression)
alpha = 1.0  # Regularization strength (lambda)
ridge = Ridge(alpha=alpha)
ridge.fit(X.reshape(-1, 1), y)
y_pred_rls = ridge.predict(X.reshape(-1, 1))

# True relationship (without noise)
y_true = true_slope * X

# Plotting
plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(X, y, label="Data Points", color="blue", alpha=0.6)

# Plot true relationship
plt.plot(X, y_true, label="True Relationship (No Noise)", color="green", linestyle="--", linewidth=2)

# Plot RLS/MAP fitted line
plt.plot(X, y_pred_rls, label=f"RLS/MAP Fit (alpha={alpha})", color="red", linewidth=2)

# Add annotations for intuitive understanding
plt.text(2, 20, "RLS/MAP balances data fit and regularization (prior)", fontsize=12, color="red")
plt.text(2, 15, "True relationship is unknown in real-world scenarios", fontsize=12, color="green")
plt.text(2, 10, "Data points are noisy observations", fontsize=12, color="blue")

# Labels and title
plt.xlabel("X")
plt.ylabel("y")
plt.title("Relationship Between RLS and MAP Estimation\n(RLS = MAP with Gaussian Prior)")
plt.legend()
plt.grid(True)
plt.show()