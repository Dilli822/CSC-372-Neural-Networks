import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Activation function (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Generate training data (approximating y = sin(x))
np.random.seed(42)
X = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)  # Input values
y_string = "np.sin(X)"
y = eval(y_string)  # This will print the result of np.sin(X)

# Initialize weights and biases globally
input_size = 1
hidden_size = 10  # Number of neurons in the hidden layer
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Store all weights at each epoch
weights_history = []

# Training parameters
epochs = 50000  # Reduced number of epochs for quicker feedback
learning_rate = 0.01

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
line_true, = ax.plot([], [], label=f"True Function {y_string}", color="blue")
line_pred, = ax.plot([], [], label="Neural Network Approximation", color="red")
ax.legend()
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-1.5, 1.5)

# Initialize text element for displaying details
text = ax.text(0.00, 0.00, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Training loop
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    y_pred = output_layer_input  # Linear activation for output

    # Compute Mean Squared Error Loss
    loss = np.mean((y_pred - y) ** 2)

    # Backpropagation
    d_loss = 2 * (y_pred - y) / y.shape[0]
    d_W2 = np.dot(hidden_layer_output.T, d_loss)
    d_b2 = np.sum(d_loss, axis=0, keepdims=True)

    d_hidden_layer = np.dot(d_loss, W2.T) * sigmoid_derivative(hidden_layer_output)
    d_W1 = np.dot(X.T, d_hidden_layer)
    d_b1 = np.sum(d_hidden_layer, axis=0, keepdims=True)

    # Gradient descent update
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1

    # Store weights at each epoch (append all weight matrices)
    weights_history.append((W1.copy(), b1.copy(), W2.copy(), b2.copy()))

    # Print loss every 500 epochs
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Update function for animation
def update(frame):
    # Get the weights for the current frame (epoch)
    W1_epoch, b1_epoch, W2_epoch, b2_epoch = weights_history[frame]

    # Forward propagation with current weights
    hidden_layer_input = np.dot(X, W1_epoch) + b1_epoch
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, W2_epoch) + b2_epoch
    y_pred = output_layer_input  # Linear activation for output

    # Update plot with new predictions
    line_pred.set_data(X, y_pred)
    
    # Update text with iteration number and sample weights (optional for display)
    text.set_text(f"Epoch: {frame}\nW1: {W1_epoch.flatten()[:5]}...\nW2: {W2_epoch.flatten()[:5]}...\nLoss: {np.mean((y_pred - y) ** 2):.6f}")
    
    return line_true, line_pred, text

# Create animation with a 5-second delay between frames
ani = FuncAnimation(fig, update, frames=len(weights_history), interval=10, repeat=False)

# Show the plot with animation
plt.show()
