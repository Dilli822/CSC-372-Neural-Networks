import numpy as np

# Define a simple quadratic function and its derivative
def func(x):
    return x**2 + 3*x + 2

def grad_func(x):
    return 2*x + 3

# Gradient descent algorithm
def gradient_descent(learning_rate, iterations, initial_guess):
    x = initial_guess
    for i in range(iterations):
        grad = grad_func(x)
        x = x - learning_rate * grad  # Update the value of x using the gradient
        print(f"Iteration {i+1}: x = {x}, f(x) = {func(x)}")
    return x

# Parameters
learning_rate = 0.1
iterations = 10
initial_guess = -5

# Running gradient descent
optimal_x = gradient_descent(learning_rate, iterations, initial_guess)
print(f"Optimal value of x: {optimal_x}")
