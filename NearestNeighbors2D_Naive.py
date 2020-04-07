"""
The purpose of this file is to demonstrate how one might write
naive code to do k-nearest neighbors by manually computing the
distances from a point to a collection of points and then using
argsort to find the indices of the closest points in the collection
"""
import matplotlib.pyplot as plt
import numpy as np

 
# Make 2 clusters.  The first cluster is in the first
# 100 rows, the second cluster is in the next 100 rows
# centered at an offset of (10, 10)
N = 100
X = np.random.randn(N*2, 2)
X[100::, :] += np.array([10, 10])

 

q = np.array([3, 3]) # Query point

# How far is the query point from every other point
distances = np.zeros(N*2)
for i in range(N*2):
    x = X[i, :] #Point under consideration is in the ith row of X
    distances[i] = np.sqrt(np.sum((x-q)**2))

# Find the nearest neighbor indices by using argsort
n_neighbors = 10
neighbors = np.argsort(distances)[0:n_neighbors]


plt.figure(figsize=(8,8))
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(q[0], q[0], 40, marker='x')
# Plot ten nearest neighbors

print(neighbors)
plt.scatter(X[neighbors, 0], X[neighbors, 1], 100, marker='*')
plt.show()