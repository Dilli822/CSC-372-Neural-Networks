import numpy as np
import matplotlib.pyplot as plt

# Multiple features dataset (X1 and X2)
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
], dtype=float)  # Shape (5,2)

y = np.array([5, 8, 11, 14, 17], dtype=float).reshape(-1, 1)  # Shape (5,1)

print("Input Features X:", X)
print("Output y:", y)

# Initialize parameters (random values)
w = np.random.randn(2, 1)  # Two weights (one per feature)
b = np.random.randn(1)
learning_rate = 0.001
epochs = 10000

# Training using Gradient Descent
losses = []
for epoch in range(epochs):
    # Forward pass (prediction)
    y_pred = np.dot(X, w) + b  # Linear model
    loss = np.mean((y - y_pred) ** 2)  # Mean Squared Error (MSE)
    
    # Compute gradients
    dw = -2 * np.dot(X.T, (y - y_pred)) / len(X)  # Derivative w.r.t w
    db = -2 * np.mean(y - y_pred)  # Derivative w.r.t b
    
    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db  # b = b - learning_rate * db

    # Store loss for visualization
    losses.append(loss)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}: Loss = {loss:.4f}')
        
# Plot loss reduction over time
plt.plot(range(epochs), losses, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Reduction Over Time")
plt.legend()
plt.show()

# Display final model parameters
print(f"Final Model: y = {w[0,0]:.4f}x1 + {w[1,0]:.4f}x2 + {b[0]:.4f}")

# User testing
while True:
    try:
        x1_test = float(input("\nEnter x1: "))
        x2_test = float(input("Enter x2: "))
        y_test = w[0, 0] * x1_test + w[1, 0] * x2_test + b[0]
        print(f"Predicted y: {y_test:.4f}")
    except ValueError:
        print("Exiting...")
        break
