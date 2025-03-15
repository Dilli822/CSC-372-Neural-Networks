import numpy as np

# Example data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # Input features (N x 2)
Y = np.array([1, 2, 3, 4])  # Target values (N x 1)

# Define RBF centers as the data points themselves
centers = X

# Gaussian RBF function
def gaussian_rbf(x, c, sigma=1.0):
    return np.exp(-np.linalg.norm(x - c)**2 / (2 * sigma**2))

# Compute RBF activations (Phi matrix)
# Without the inner loop, we would only compute RBF activations for one center.
# With the double loop, we compute the full Φ matrix, which is required for RBF regression
sigma = 1.0  # Spread parameter
Phi = np.array([[gaussian_rbf(x, c, sigma) for c in centers] for x in X])

# Solve for weights using least squares
"""
w = (Φ^T Φ)^-1 Φ^T Y
.
"""
w = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ Y
print("Final weights: ", w)

# Function to make predictions
def predict(x_new):
    return sum(w_i * gaussian_rbf(x_new, c_i, sigma) for w_i, c_i in zip(w, centers))

# Example prediction
x_new = np.array([4.4, 5.4])  # More than 2D Array 
y_pred = predict(x_new)
print(f"Predicted Y for {x_new}: {y_pred}")


"""
φ(x, c) = exp(- ||x - c||^2 / (2 * σ^2))

where:
- φ(x, c) is the Gaussian Radial Basis Function (RBF).
- x is the input feature vector.
- c is the center of the RBF.
- σ is the spread (standard deviation) of the Gaussian function.
- ||x - c||^2 represents the squared Euclidean distance between x and c.
"""

"""
The matrix Φ (Phi matrix) is constructed as follows:

Φ_ij = exp(- ||x_i - c_j||^2 / (2 * σ^2))

where:
- Φ is the RBF activation matrix.
- x_i is the i-th input data point from X.
- c_j is the j-th RBF center from centers.
- σ is the spread (standard deviation) of the Gaussian function.
- ||x_i - c_j||^2 is the squared Euclidean distance between x_i and c_j.
- Each entry Φ_ij represents the response of the j-th RBF function for the i-th input sample.

Structure of Φ Matrix:
If X has N samples and M RBF centers, then Φ is an N x M matrix:

Φ =
[ φ(x_1, c_1)  φ(x_1, c_2)  ...  φ(x_1, c_M) ]
[ φ(x_2, c_1)  φ(x_2, c_2)  ...  φ(x_2, c_M) ]
[    ...        ...         ...      ...     ]
[ φ(x_N, c_1)  φ(x_N, c_2)  ...  φ(x_N, c_M) ]
"""

"""
1.Outer loop (for x in X): Iterates over each data point in 𝑋 (rows of the dataset).
2. Inner loop (for c in centers): Iterates over each center in centers to compute its RBF value with 𝑥.
3. gaussian_rbf(x, c, sigma): Computes the Gaussian RBF function for input 𝑥 with center 𝑐.
"""


"""
predict(x_new) can be mathematically expressed as:



where:

 is the predicted output for a new input .

 are the learned weights.

 is the Gaussian Radial Basis Function (RBF) centered at :



 is the number of RBF centers.

 is the spread parameter.

 is the new input data.

 are the RBF centers.

This equation computes the weighted sum of Gaussian RBF activations to predict the output value.


"""

# Define the mathematical expressions as a formatted string
formula = """
The formula represented by the Python line:

    return sum(w_i * gaussian_rbf(x_new, c_i, sigma) for w_i, c_i in zip(w, centers))

is mathematically expressed as:

    Ŷ(x_new) = Σ (w_i * ϕ(x_new, c_i)), for i = 1 to N

where:

- Ŷ(x_new) is the predicted output for a new input x_new.
- w_i are the learned weights.
- ϕ(x_new, c_i) is the Gaussian Radial Basis Function (RBF) centered at c_i:

      ϕ(x, c) = exp(-(||x - c||²) / (2 * σ²))

- N is the number of RBF centers.
- σ (sigma) is the spread parameter.
- x_new is the new input data.
- c_i are the RBF centers.

This equation computes the weighted sum of Gaussian RBF activations to predict the output value.
"""

# Print the formula
# print(formula)
