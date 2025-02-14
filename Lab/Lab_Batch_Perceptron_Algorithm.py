import numpy as np
from tabulate import tabulate

# Define OR gate dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
y = np.array([-1, 1, 1, 1])  # Target labels (-1 for 0, +1 for 1)

# Initialize weights and bias
w = np.zeros(X.shape[1])  # Weight vector (2D)
b = 0                     # Bias
eta = 1                   # Learning rate
max_epochs = 10           # Max number of iterations

# Batch Perceptron Algorithm formula
print("""
Batch Perceptron Algorithm:

    w = w + eta * sum(y_i * x_i)
    b = b + eta * sum(y_i)

Where:
    w  = Weight vector
    eta = Learning rate
    y_i = Actual output for sample i
    x_i = Input vector for sample i
    b  = Bias term
    sum = Sum over all misclassified samples

Steps:
1. Initialize weights and bias to zero or small random values.
2. For each training sample, compute the output of the perceptron (y_pred).
3. Update the weights and bias if the sample is misclassified.
4. Repeat until convergence or for a fixed number of iterations.
""")


# Training Loop
for epoch in range(max_epochs):
    print(f"\nEpoch {epoch + 1}")
    print("-" * 80)

    # Prepare table headers
    headers = ['x1', 'x2', 'Actual Output', 'w1*x1 + w2*x2 + b', 'Prediction',  'Misclassified?']

    # Initialize a list to store rows for the table
    rows = []

    total_weight_update = np.zeros_like(w)  # Accumulate weight updates
    total_bias_update = 0                   # Accumulate bias updates
    misclassified = 0

    for i in range(len(X)):  # Iterate over all samples
        net_input = np.dot(w, X[i]) + b  # Compute weighted sum
        y_pred = 1 if net_input > 0 else -1  # Apply step function
        misclassified_flag = "Yes" if y_pred != y[i] else "No"

        # Append the row for the table
        rows.append([X[i][0], X[i][1], y[i], net_input, y_pred, misclassified_flag])

        if y_pred != y[i]:  # Misclassification check
            old_w = w.copy()
            old_b = b
            
            total_weight_update += y[i] * X[i]  # Accumulate weight update
            total_bias_update += y[i]  # Accumulate bias update
            misclassified += 1

            # Print misclassified sample details
            print(f"Misclassified Sample -> Old Weights: w1 = {old_w[0]}, w2 = {old_w[1]}, Bias = {old_b}")
            print(f"                     -> New Weights: w1 = {old_w[0] + eta * (y[i] * X[i])[0]}, w2 = {old_w[1] + eta * (y[i] * X[i])[1]}, Bias = {old_b + eta * y[i]}")

    # Apply batch weight update
    w += eta * total_weight_update
    b += eta * total_bias_update

    # Print table using tabulate
    print(tabulate(rows, headers, tablefmt="grid"))

    print(f"\nUpdated Weights after Epoch {epoch + 1}: w1 = {w[0]}, w2 = {w[1]}, Bias = {b}")
    print(f"Total Misclassified: {misclassified}")
    
    # Stop if all samples are classified correctly
    if misclassified == 0:
        break

print("----------------------------------------------------------------------------------------------------\n")
def print_batch_perceptron_algorithm():
    print("Algorithm: The Batch Perceptron Algorithm")
    print("--------------------------------------------------\n")
    
    # Step 1: Perceptron Cost Function
    print("1. Perceptron Cost Function:")
    print("In the perceptron algorithm, the goal is to minimize the number of misclassified samples.")
    print("To do so, we introduce a cost function that allows the use of a gradient search. The generalized cost function is defined as:\n")
    print("    J(w) = ∑[ n ∈ x ] - w^T x(n) d(n)\n")
    print("Where:")
    print("    w     is weight vector")
    print("    x(n)  is input sample")
    print("    d(n)  is desired output for the sample x(n)")
    print("    x     is set of misclassified samples\n")
    print("If all samples are classified correctly, the set x is empty, and J(w) = 0. This cost function is differentiable with respect to the weight vector w.\n")
    
    # Step 2: Gradient of the Cost Function
    print("2. Gradient of the Cost Function:")
    print("To update the weight vector w, we calculate the gradient of the cost function with respect to w. The gradient is:\n")
    print("    ∇J(w) = - ∑[ n ∈ x ] x(n) d(n)\n")
    print("This gradient tells us the direction in which we should adjust the weight vector to minimize the cost function.\n")
    
    # Step 3: Weight Update Rule
    print("3. Weight Update Rule:")
    print("Using gradient descent, the weight update rule is as follows. The adjustment to the weight vector w is applied in the opposite direction of the gradient:\n")
    print("    w(n + 1) = w(n) - η(n) ∇J(w)\n")
    print("This simplifies to:\n")
    print("    w(n + 1) = w(n) + η(n) ∑[ n ∈ x ] x(n) d(n)\n")
    print("Where:")
    print("    η(n)  is learning rate at time-step n")
    print("    ∑[ n ∈ x ] x(n) d(n) is sum of all misclassified samples.\n")
    print("This equation is the Batch Perceptron Algorithm.\n")
    
    # Step 4: Batch Nature of the Algorithm
    print("4. Batch Nature of the Algorithm:")
    print("The key feature of this algorithm is that it corrects the weight vector by summing up the misclassifications over the entire batch,")
    print("rather than just a single sample at a time. Hence, the algorithm is called a batch method because the weight update is computed")
    print("using all misclassified samples within the current batch.\n")

# Print the Batch Perceptron Algorithm
print_batch_perceptron_algorithm()
