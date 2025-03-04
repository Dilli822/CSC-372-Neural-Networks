import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR output

# Step 2: Define RBF Function
def rbf_function(x, c, sigma):
    return np.exp(-np.linalg.norm(x - c, axis=1) ** 2 / (2 * sigma ** 2))

# Step 3: Choose RBF Centers (Same as Input Points)
centers = X.copy()

# Step 4: Compute Sigma (Spread of RBFs)
distances = np.linalg.norm(centers[:, np.newaxis] - centers, axis=2)
sigma = np.mean(distances)

# Step 5: Compute RBF Activations for Training Data
H = np.array([rbf_function(X, c, sigma) for c in centers]).T

# Step 6: Compute Output Weights using Pseudo-Inverse
weights = np.linalg.pinv(H) @ y  # Moore-Penrose Pseudo-Inverse Solution
bias = np.mean(y - H @ weights)  # Compute Bias

# Print final weights and bias
print("Final Weights:", weights)
print("Final Bias:", bias)

# Step 7: Define Prediction Function
def predict(X_new):
    H_new = np.array([rbf_function(X_new, c, sigma) for c in centers]).T
    return np.round(H_new @ weights + bias)

# Step 8: Allow user to input XOR function for testing
def user_test():
    while True:
        try:
            x1 = float(input("Enter first input (0 or 1): "))
            x2 = float(input("Enter second input (0 or 1): "))
            if x1 not in [0, 1] or x2 not in [0, 1]:
                print("Invalid input. Please enter 0 or 1.")
                continue
            user_input = np.array([[x1, x2]])
            prediction = predict(user_input)
            print(f"Predicted Output: {int(prediction[0])}")
        except ValueError:
            print("Invalid input. Please enter numeric values.")

        cont = input("Do you want to test another input? (y/n): ")
        if cont.lower() != 'y':
            break

# Step 9: Test Predictions
y_pred = predict(X)
print("Predictions:", y_pred)
print("Actual:", y)

# Print the RBF equation
print("Equation: y = sum(w_i * phi_i(x)) + bias")

# Step 10: Plot Decision Boundary
def plot_decision_boundary():
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(grid).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm', s=100)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200, color='black', label='Centers')
    plt.legend()
    plt.title("XOR Classification using RBF Network")
    plt.show()

plot_decision_boundary()
user_test()


# https://chatgpt.com/share/67c54cee-d810-8010-b71d-5bce2b67710e