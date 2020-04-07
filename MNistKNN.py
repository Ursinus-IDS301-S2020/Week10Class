import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import scipy.stats
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import skimage
import skimage.io

# Setup an array that will hold 200 images in 784 dimensions (28x28 pixels)
num_per_class = 100
X = np.zeros((10*num_per_class, 784))

# Load in all digits
# Put the  digits into blocks of rows in the matrix, so the first
# num_per_class rows are 0s, the next num_per_class are 1s, etc
for digit in range(10):
    print("Loading", digit)
    for i in range(num_per_class):    
        x = skimage.io.imread("Digits/{}/{}.png".format(digit, i))
        x = x.flatten()
        X[digit*num_per_class+i, :] = x # Put image i into row i

n_neighbors = 10
## Step 1: Load in digit
q = skimage.io.imread("mydigit.png")
qflat = q.flatten()
qflat = qflat[None, :] # 2D array with a single row (NN library requires this)

## Step 2: Compute nearest neighbors
nbrs = NearestNeighbors(n_neighbors=n_neighbors)
nbrs.fit(X)
distances, neighbors = nbrs.kneighbors(qflat)
distances = distances.flatten()
neighbors = neighbors.flatten()


## Step 3: Plot results
k = int(np.ceil(np.sqrt(n_neighbors+1)))
plt.subplot(k, k, 1)
plt.imshow(q, cmap='gray')
plt.title("Query")
for i, idx in enumerate(neighbors):
    plt.subplot(k, k, i+2)
    plt.imshow(np.reshape(X[idx, :], (28, 28)), cmap='gray')
    plt.title("%i (%.3g)"%(np.floor(idx/num_per_class), distances[i]))
    plt.axis('off')
plt.tight_layout()
plt.show()

## Step 4: Compute the mode of the class
neighbors = np.floor(neighbors/num_per_class)
print(neighbors)
print("I think it is ", scipy.stats.mode(neighbors))