import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

COLUMN_NAMES = ["GRE", "TOEFL", "University Rating", "SOP", "Recommendation", "CGPA", "Research"]


A = np.loadtxt("Admissions.csv", delimiter=",")
X = A[:, 0:-1] # Independent variables
y = A[:, -1] # Dependent variable (chance of acceptance)

# Plot distribution plots to show different columns of X

# Plot PCA

# Perform cross-validated ridge regression

# Do a scatterplot of predicted versus actual

# Show each coefficient on a bar plot

# Normalize the coefficients properly