import numpy as np
import matplotlib.pyplot as plt

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
# Use eval to evaluate the string as code
y = eval(y_string)


print(y)  # This will print the result of np.sin(X)
# Initialize weights and biases
input_size = 1
hidden_size = 10  # Number of neurons in the hidden layer
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training parameters
epochs = 50000
learning_rate = 0.01

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

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Plot results
plt.scatter(X, y, label=f"True Function {y_string}", color="blue")
print(f"True Function {y}")
plt.plot(X, y_pred, label="Neural Network Approximation", color="red")
plt.legend()
plt.show()


"""
np.abs()
np.arccos()
np.arcsin()
np.arctan()
np.arcsinh()
np.arccosh()
np.arctanh()
np.cos()
np.cosh()
np.sin()
np.sinh()
np.tan()
np.tanh()
np.log()
np.log10()
np.log2()
np.exp()
np.sqrt()
np.square()
np.radians()
np.degrees()
np.max()
np.min()
np.mean()
np.median()
np.std()
np.var()
np.sum()
np.prod()
np.cumsum()
np.diff()
np.unique()
np.concatenate()
np.transpose()
np.reshape()
np.split()
np.tile()
np.vstack()
np.hstack()
np.column_stack()
"""