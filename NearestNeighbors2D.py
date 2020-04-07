"""
The purpose of this code is to show how to use the
nearest neighbor class as part of the scikit-learn
library, which is a faster way of finding k-nearest neighbors
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

 
# We first setup the points exactly as we did in
# the naive example
N = 100
X = np.random.randn(N*2, 2)
X[100::, :] += np.array([10, 10])

 
# Query point, which now has to be put along
# the row of a 2D array
q = np.array([[3, 3]]) 

# The code to perform nearest neighbors
n_neighbors = 10
# First we create the nearest neighbors object
nbrs = NearestNeighbors(n_neighbors=n_neighbors)
# Then we "train" it on the points in X
nbrs.fit(X)
# Now we're ready to query the nearest neighbors
# object with a particular point, which returns
# both the distances and indices of the k-nearest
# neighbors in parallel arrays, so we don't
# need to use argsort anymore
distances, neighbors = nbrs.kneighbors(q)
distances = distances.flatten()
neighbors = neighbors.flatten()

 
plt.figure(figsize=(8,8))
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(q[0, 0], q[0, 1], 40, marker='x')

# Plot ten nearest neighbors
print(neighbors)
plt.scatter(X[neighbors, 0], X[neighbors, 1], 100, marker='*')
plt.show()