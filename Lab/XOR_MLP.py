# import numpy as np

# # Sigmoid activation function and its derivative
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1 - x)

# # XOR inputs and expected outputs
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input pairs
# y = np.array([[0], [1], [1], [0]])  # XOR outputs

# # Set random seed for reproducibility
# np.random.seed(1)

# # Initialize weights and biases with small random values instead of zeros
# weights_input_hidden = np.random.uniform(0.0, 0.5, (2, 3))  # 2 inputs, 3 neurons in hidden layer
# weights_hidden_output = np.random.uniform(0.0, 0.5, (3, 1))  # 3 neurons in hidden layer, 1 output

# # Initialize biases with small random values
# bias_hidden = np.random.uniform(0.0, 0.5, (1, 3))
# bias_output = np.random.uniform(0.0, 0.5, (1, 1))

# # Learning rate
# learning_rate = 0.1  # Increased from 0.001 for faster convergence

# # Training the model
# epochs = 10000

# for epoch in range(epochs):
#     # Feedforward
#     hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
#     hidden_layer_output = sigmoid(hidden_layer_input)
    
#     output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
#     output_layer_output = sigmoid(output_layer_input)
    
#     # Compute the error
#     error = y - output_layer_output
    
#     # Backpropagation
#     d_output = error * sigmoid_derivative(output_layer_output)
    
#     error_hidden_layer = d_output.dot(weights_hidden_output.T)
#     d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
#     # Update weights and biases
#     weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
#     bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    
#     weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
#     bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
#     # Print progress every 1000 epochs
#     if epoch % 1000 == 0:
#         print(f"\nEpoch {epoch}")
#         print("Input 1  Input 2  Output  Target  Error")
#         print("-" * 45)
#         for i in range(len(X)):
#             # Forward propagation for single input
#             hidden = sigmoid(np.dot(X[i], weights_input_hidden) + bias_hidden)
#             output = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)
#             print(f"{X[i][0]:<8} {X[i][1]:<8} {output[0][0]:.4f}  {y[i][0]:<6} {(y[i][0] - output[0][0]):.4f}")

# # Final predictions
# print("\nFinal Results:")
# print("Input 1  Input 2  Predicted  Target")
# print("-" * 40)
# for i in range(len(X)):
#     hidden = sigmoid(np.dot(X[i], weights_input_hidden) + bias_hidden)
#     output = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)
#     predicted = 1 if output[0][0] > 0.5 else 0  # Fixed: access output[0][0] before converting to int
#     print(f"{X[i][0]:<8} {X[i][1]:<8} {predicted:<9} {y[i][0]}")



# import numpy as np

# # Sigmoid activation function and its derivative
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1 - x)

# # XOR inputs and expected outputs
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input pairs
# y = np.array([[0], [1], [1], [0]])  # XOR outputs

# # Set random seed for reproducibility
# np.random.seed(1)

# # Initialize weights and biases with small random values
# weights_input_hidden = np.random.uniform(0.0, 0.5, (2, 3))
# weights_hidden_output = np.random.uniform(0.0, 0.5, (3, 1))
# bias_hidden = np.random.uniform(0.0, 0.5, (1, 3))
# bias_output = np.random.uniform(0.0, 0.5, (1, 1))

# # Learning rate
# learning_rate = 0.1

# # Training the model
# epochs = 10000

# for epoch in range(epochs):
#     # Feedforward
#     hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
#     hidden_layer_output = sigmoid(hidden_layer_input)
    
#     output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
#     output_layer_output = sigmoid(output_layer_input)
    
#     # Compute the error
#     error = y - output_layer_output
    
#     # Backpropagation
#     d_output = error * sigmoid_derivative(output_layer_output)
    
#     error_hidden_layer = d_output.dot(weights_hidden_output.T)
#     d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
#     # Update weights and biases
#     weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
#     bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    
#     weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
#     bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
#     # Print progress for every epoch
#     print(f"\nEpoch {epoch}")
#     print("Input 1  Input 2  Output  Target  Error")
#     print("-" * 45)
#     for i in range(len(X)):
#         # Forward propagation for single input
#         hidden = sigmoid(np.dot(X[i], weights_input_hidden) + bias_hidden)
#         output = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)
#         print(f"{X[i][0]:<8} {X[i][1]:<8} {output[0][0]:.4f}  {y[i][0]:<6} {(y[i][0] - output[0][0]):.4f}")

# # Final predictions
# print("\nFinal Results:")
# print("Input 1  Input 2  Predicted  Target")
# print("-" * 40)
# for i in range(len(X)):
#     hidden = sigmoid(np.dot(X[i], weights_input_hidden) + bias_hidden)
#     output = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)
#     predicted = 1 if output[0][0] > 0.5 else 0
#     print(f"{X[i][0]:<8} {X[i][1]:<8} {predicted:<9} {y[i][0]}")



# import numpy as np

# # Sigmoid activation function and its derivative
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1 - x)

# # XOR inputs and expected outputs
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input pairs
# y = np.array([[0], [1], [1], [0]])  # XOR outputs

# # Set random seed for reproducibility
# np.random.seed(1)

# # Initialize weights and biases with small random values
# weights_input_hidden = np.random.uniform(0.0, 0.5, (2, 3))
# weights_hidden_output = np.random.uniform(0.0, 0.5, (3, 1))
# bias_hidden = np.random.uniform(0.0, 0.5, (1, 3))
# bias_output = np.random.uniform(0.0, 0.5, (1, 1))

# # Learning rate
# learning_rate = 0.1

# # Training the model
# epochs = 10000

# for epoch in range(epochs):
#     # Feedforward
#     hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
#     hidden_layer_output = sigmoid(hidden_layer_input)
    
#     output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
#     output_layer_output = sigmoid(output_layer_input)
    
#     # Compute the error
#     error = y - output_layer_output
    
#     # Backpropagation
#     d_output = error * sigmoid_derivative(output_layer_output)
    
#     error_hidden_layer = d_output.dot(weights_hidden_output.T)
#     d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
#     # Calculate weight and bias updates
#     weight_hidden_output_update = hidden_layer_output.T.dot(d_output) * learning_rate
#     bias_output_update = np.sum(d_output, axis=0, keepdims=True) * learning_rate
    
#     weight_input_hidden_update = X.T.dot(d_hidden_layer) * learning_rate
#     bias_hidden_update = np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
#     # Update weights and biases
#     weights_hidden_output += weight_hidden_output_update
#     bias_output += bias_output_update
    
#     weights_input_hidden += weight_input_hidden_update
#     bias_hidden += bias_hidden_update
    
#     # Print progress for every epoch
#     print(f"\nEpoch {epoch}")
#     print("Input 1  Input 2  Output  Target  Error")
#     print("-" * 45)
#     for i in range(len(X)):
#         # Forward propagation for single input
#         hidden = sigmoid(np.dot(X[i], weights_input_hidden) + bias_hidden)
#         output = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)
#         print(f"{X[i][0]:<8} {X[i][1]:<8} {output[0][0]:.4f}  {y[i][0]:<6} {(y[i][0] - output[0][0]):.4f}")
    
#     # Print weight and bias updates
#     print("\nWeight Updates:")
#     print("\nInput-Hidden Layer Weights:")
#     print(weight_input_hidden_update)
#     print("\nHidden-Output Layer Weights:")
#     print(weight_hidden_output_update)
    
#     print("\nBias Updates:")
#     print("\nHidden Layer Biases:")
#     print(bias_hidden_update)
#     print("\nOutput Layer Bias:")
#     print(bias_output_update)
    
#     print("\nCurrent Weights:")
#     print("\nInput-Hidden Layer Weights:")
#     print(weights_input_hidden)
#     print("\nHidden-Output Layer Weights:")
#     print(weights_hidden_output)
    
#     print("\nCurrent Biases:")
#     print("\nHidden Layer Biases:")
#     print(bias_hidden)
#     print("\nOutput Layer Bias:")
#     print(bias_output)
#     print("-" * 45)

# # Final predictions
# print("\nFinal Results:")
# print("Input 1  Input 2  Predicted  Target")
# print("-" * 40)
# for i in range(len(X)):
#     hidden = sigmoid(np.dot(X[i], weights_input_hidden) + bias_hidden)
#     output = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)
#     predicted = 1 if output[0][0] > 0.5 else 0
#     print(f"{X[i][0]:<8} {X[i][1]:<8} {predicted:<9} {y[i][0]}")




import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR inputs and expected outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Set random seed for reproducibility
np.random.seed(1)

# Initialize weights and biases
weights_input_hidden = np.random.uniform(0.0, 0.5, (2, 3))
weights_hidden_output = np.random.uniform(0.0, 0.5, (3, 1))
bias_hidden = np.random.uniform(0.0, 0.5, (1, 3))
bias_output = np.random.uniform(0.0, 0.5, (1, 1))

# Learning rate
learning_rate = 0.1

# Training the model
epochs = 10000

# Initialize previous error
previous_error = np.inf
total_error = 0

for epoch in range(epochs):
    # Store old weights and biases
    old_weights_input_hidden = weights_input_hidden.copy()
    old_weights_hidden_output = weights_hidden_output.copy()
    old_bias_hidden = bias_hidden.copy()
    old_bias_output = bias_output.copy()
    
    # Feedforward
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)
    
    # Compute the error
    error = y - output_layer_output
    current_error = np.mean(np.abs(error))
    
    # Backpropagation
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Calculate updates
    weight_hidden_output_update = hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output_update = np.sum(d_output, axis=0, keepdims=True) * learning_rate
    
    weight_input_hidden_update = X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden_update = np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
    # Apply updates
    weights_hidden_output += weight_hidden_output_update
    bias_output += bias_output_update
    weights_input_hidden += weight_input_hidden_update
    bias_hidden += bias_hidden_update
    
    # Print detailed information every 100 epochs
    if epoch % 100 == 0:
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}")
        print(f"{'='*80}")
        
        print("\nERROR COMPARISON:")
        print(f"Previous Error: {previous_error:.6f}")
        print(f"Current Error:  {current_error:.6f}")
        print(f"Error Change:   {current_error - previous_error:.6f}")
        
        print("\nINPUT-HIDDEN LAYER WEIGHTS:")
        print("Old weights:")
        print(old_weights_input_hidden)
        print("\nWeight updates:")
        print(weight_input_hidden_update)
        print("\nNew weights:")
        print(weights_input_hidden)
        print("\nTotal change:")
        print(weights_input_hidden - old_weights_input_hidden)
        
        print("\nHIDDEN-OUTPUT LAYER WEIGHTS:")
        print("Old weights:")
        print(old_weights_hidden_output)
        print("\nWeight updates:")
        print(weight_hidden_output_update)
        print("\nNew weights:")
        print(weights_hidden_output)
        print("\nTotal change:")
        print(weights_hidden_output - old_weights_hidden_output)
        
        print("\nHIDDEN LAYER BIAS:")
        print("Old bias:")
        print(old_bias_hidden)
        print("\nBias updates:")
        print(bias_hidden_update)
        print("\nNew bias:")
        print(bias_hidden)
        print("\nTotal change:")
        print(bias_hidden - old_bias_hidden)
        
        print("\nOUTPUT LAYER BIAS:")
        print("Old bias:")
        print(old_bias_output)
        print("\nBias updates:")
        print(bias_output_update)
        print("\nNew bias:")
        print(bias_output)
        print("\nTotal change:")
        print(bias_output - old_bias_output)
        
        print("\nCURRENT PREDICTIONS:")
        print("Input 1  Input 2  Output  Target  Error")
        print("-" * 45)
        for i in range(len(X)):
            hidden = sigmoid(np.dot(X[i], weights_input_hidden) + bias_hidden)
            output = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)
            print(f"{X[i][0]:<8} {X[i][1]:<8} {output[0][0]:.4f}  {y[i][0]:<6} {(y[i][0] - output[0][0]):.4f}")
    
    previous_error = current_error

# Final predictions
print("\nFINAL RESULTS:")
print("Input 1  Input 2  Predicted  Target")
print("-" * 40)
for i in range(len(X)):
    hidden = sigmoid(np.dot(X[i], weights_input_hidden) + bias_hidden)
    output = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)
    predicted = 1 if output[0][0] > 0.5 else 0
    print(f"{X[i][0]:<8} {X[i][1]:<8} {predicted:<9} {y[i][0]}")