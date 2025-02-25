import numpy as np
import matplotlib.pyplot as plt

# Static dataset
# .reshape(-1,1) converts a 1D array into a column vector for correct matrix operations.
X = np.array([1, 2, 3, 4, 5], dtype=float).reshape(-1, 1)  # Input (features)
y = np.array([3, 5, 7, 9, 11], dtype=float).reshape(-1, 1)  # Target (labels)

print("Input Features X", X)
print("Output", y)

# Initialize parameters (random values)
w = np.random.randn(1) # Uses Standard Normal Distribution 1-D Vector
print("random weight is ", w)
b = np.random.randn(1)
print("random bias is ", b)
learning_rate = 0.001
epochs = 10000

# Training using Backpropagation
losses = []
for epoch in range(epochs):
    # Forward pass (prediction)
    y_pred = w * X + b
    loss = np.mean((y - y_pred) ** 2)  # Mean Squared Error (MSE)

    # Compute gradients
    dw = -2 * np.mean(X * (y - y_pred))  # Derivative w.r.t w
    db = -2 * np.mean(y - y_pred)        # Derivative w.r.t b

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    # Store loss for visualization
    losses.append(loss)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss:.4f}, w = {w[0]:.4f}, b = {b[0]:.4f}')
        
# Plot loss reduction over time
plt.plot(range(epochs), losses, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Reduction Over Time")
plt.legend()
plt.show()
# Display final model parameters
print(f"Final Model: y = {w[0]:.4f}x + {b[0]:.4f}")

# User testing
while True:
    try:
        x_test = float(input("\nEnter an x value to predict y (or type 'exit' to quit): "))
        y_test = w[0] * x_test + b[0]
        print(f"Predicted y: {y_test:.4f}")
    except ValueError:
        print("Exiting...")
        break



