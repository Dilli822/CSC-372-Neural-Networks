import numpy as np
import matplotlib.pyplot as plt

# Function to implement K-means clustering
def kmeans(X, k, max_iters=100):
    # Randomly initialize centroids by picking k random data points
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for i in range(max_iters):
        # Step 1: Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Step 2: Recalculate centroids as the mean of points in each cluster
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # Step 3: Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
        
    return centroids, labels

# Sample data (You can replace this with your own data)
np.random.seed(42)
X = np.random.randn(300, 2)

# Perform K-means clustering
k = 3  # Number of clusters
centroids, labels = kmeans(X, k)

# Create subplots
plt.figure(figsize=(14, 6))

# Plot original data (before K-means) in the first subplot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.scatter(X[:, 0], X[:, 1], c='gray', label='Original Data')
plt.title('Original Data')
plt.legend()

# Plot data after applying K-means in the second subplot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Clustered Data')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering (Using Python & NumPy)')
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()
