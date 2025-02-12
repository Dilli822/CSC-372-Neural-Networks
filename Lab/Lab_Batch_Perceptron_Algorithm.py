import numpy as np
import pandas as pd

# Define OR gate dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
y = np.array([-1, 1, 1, 1])  # Target labels (-1 for 0, +1 for 1)

# Initialize weights and bias
w = np.zeros(X.shape[1])  # Weight vector (2D)
b = 0                     # Bias
eta = 1                   # Learning rate
max_epochs = 10           # Max number of iterations

# Training Loop
for epoch in range(max_epochs):
    print(f"\nEpoch {epoch + 1}")
    print("-" * 40)
    print(f"{'x1':<5}{'x2':<5}{'y':<5}{'w1*x1 + w2*x2 + b':<20}{'Prediction':<12}{'Misclassified?':<10}")
    print("-" * 40)

    total_weight_update = np.zeros_like(w)  # Accumulate weight updates
    total_bias_update = 0                   # Accumulate bias updates
    misclassified = 0

    for i in range(len(X)):  # Iterate over all samples
        net_input = np.dot(w, X[i]) + b  # Compute weighted sum
        y_pred = 1 if net_input > 0 else -1  # Apply step function
        misclassified_flag = "Yes" if y_pred != y[i] else "No"

        # Print detailed computation
        print(f"{X[i][0]:<5}{X[i][1]:<5}{y[i]:<5}{net_input:<20}{y_pred:<12}{misclassified_flag:<10}")

        if y_pred != y[i]:  # Misclassification check
            total_weight_update += y[i] * X[i]  # Accumulate weight update
            total_bias_update += y[i]  # Accumulate bias update
            misclassified += 1

    # Apply batch weight update
    w += eta * total_weight_update
    b += eta * total_bias_update

    print(f"\nUpdated Weights: w1 = {w[0]}, w2 = {w[1]}, Bias = {b}")
    print(f"Total Misclassified: {misclassified}")
    
    # Stop if all samples are classified correctly
    if misclassified == 0:
        break
