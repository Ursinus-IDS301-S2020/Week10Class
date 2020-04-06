import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import scipy.stats
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import skimage

# Setup an array that will hold 200 images in 784 dimensions (28x28 pixels)
num_per_class = 100
X = np.zeros((10*num_per_class, 784))

# Load in all digits
for digit in range(10):
    print("Loading", digit)
    for i in range(num_per_class):    
        x = skimage.io.imread("Digits/{}/{}.png".format(digit, i))
        x = x.flatten()
        X[digit*num_per_class+i, :] = x # Put image i into row i

n_neighbors = 20

## TODO: Compute nearest neighbors for an example digit

## Step 1: Load in digit
y = skimage.io.imread("mydigit.png")
yflat = y.flatten()

## Step 2: Compute nearest neighbors

## Step 3: Plot results

## Step 4: Compute the mode of the class