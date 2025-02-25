import numpy as np
import time

# Activation Function and Its Derivative
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): 
    return x * (1 - x)  # Assuming x is already sigmoid-activated

# XOR Input and Expected Output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # Expected Output

# Training Parameters
learning_rate = 0.001  # Increased Learning Rate
epochs = 100000

# Manually Defined Weights and Biases
def initialize_weights():
    np.random.seed(42)  # For reproducibility
    
    weights_input_hidden = np.array([
        [0.7, 0.6, 0.8],  # First input neuron weights
        [0.4, 0.8, 0.2]  # Second input neuron weights
    ], dtype=np.float64)
    
    weights_hidden_output = np.array([
        [0.4],
        [0.7],
        [0.8]
    ], dtype=np.float64)
    
    bias_hidden = np.array([[1.00, 1.00, 1.00]], dtype=np.float64)  
    bias_output = np.array([[1.00]], dtype=np.float64)  

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Train the MLP
def train_xor_mlp():
    print("\nTraining XOR MLP with Manual Weight Initialization")

    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = initialize_weights()
    start_time = time.time()

    for epoch in range(epochs):
        # Forward Propagation
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)

        output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        output_output = sigmoid(output_input)

        # Compute Error
        error = y - output_output
        loss = np.mean(np.abs(error))

        # Backpropagation
        d_output = error * sigmoid_derivative(output_output)
        d_hidden = d_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_output)

        # Update Weights and Biases
        weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
        bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

        weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

        # Print loss at intervals
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    end_time = time.time()
    print(f"\nTraining completed in {round(end_time - start_time, 4)} seconds")
    print(f"Final Loss: {loss:.6f}\n")

    # Print Final Weights and Biases
    print("Final Weights and Biases After Training:")
    print("Weights Input-Hidden:\n", weights_input_hidden)
    print("------------------------------------")
    print("Bias Hidden:\n", bias_hidden)
    print("------------------------------------")
    print("Weights Hidden-Output:\n", weights_hidden_output)
    print("------------------------------------")
    print("Bias Output:\n", bias_output)

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Function to Predict XOR Output for User Input
def predict(weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    while True:
        try:
            user_input = input("\nEnter two binary values (e.g., 0 1) to test the XOR Function or type 'exit' to quit: ").strip().lower()
            if user_input == "exit":
                break
            
            values = list(map(int, user_input.split()))
            if len(values) != 2 or not all(v in [0, 1] for v in values):
                print("Invalid input. Please enter two binary values (0 or 1).")
                continue

            # Convert input to numpy array and perform forward propagation
            X_test = np.array([values])
            hidden_input = np.dot(X_test, weights_input_hidden) + bias_hidden
            hidden_output = sigmoid(hidden_input)

            output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
            output_output = sigmoid(output_input)

            # Print Prediction
            print(f"Predicted Output by XOR MLP: {output_output[0][0]:.6f} (Rounded: {round(output_output[0][0])})")

        except Exception as e:
            print(f"Error: {e}. Please enter valid binary values.")

# Train XOR MLP and Allow User to Test
weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train_xor_mlp()
predict(weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
