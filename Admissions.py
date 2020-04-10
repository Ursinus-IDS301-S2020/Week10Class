import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

COLUMN_NAMES = ["GRE", "TOEFL", "University Rating", "SOP", "Recommendation", "CGPA", "Research"]


A = np.loadtxt("Admissions.csv", delimiter=",")
X = A[:, 0:-1] # Independent variables
y = A[:, -1] # Dependent variable (chance of acceptance)

print(X.shape)

# Plot PCA
plt.figure(figsize=(10, 10))
pca = PCA(n_components=2)
Y = pca.fit_transform(X)
plt.scatter(Y[:, 0], Y[:, 1], c=y)
plt.colorbar()

# Perform cross-validated ridge regression
clf = RidgeCV(alphas=[1e-2, 1e-1, 1, 10]).fit(X, y)
print(clf.score(X, y))

# Do a scatterplot of predicted versus actual
coeff = clf.coef_
ypred = X.dot(coeff)
plt.figure(figsize=(8, 8))
plt.scatter(y, ypred)
plt.xlabel("Actual Chance")
plt.ylabel("Predicted Chance")

# Show each coefficient on a bar plot
# Normalize the coefficients properly
scales = np.max(X, axis=0) - np.min(X, axis=0)
plt.figure(figsize=(8, 8))
N = X.shape[1]
plt.bar(range(N), scales*coeff)
plt.xticks(range(N), COLUMN_NAMES)
plt.tight_layout()
plt.show()


