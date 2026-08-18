# Experiment: Implementation of Expectation Maximization Algorithm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()
X = iris.data

# Select two features for visualization
X = X[:, [0, 2]]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create Gaussian Mixture Model
gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',
    random_state=42
)

# Train the model using EM algorithm
gmm.fit(X_scaled)

# Predict cluster labels
labels = gmm.predict(X_scaled)

# Display cluster labels
print("Cluster Labels:")
print(labels)

# Display cluster means
print("\nCluster Means:")
print(gmm.means_)

# Display covariance matrices
print("\nCovariance Matrices:")
print(gmm.covariances_)

# Calculate log-likelihood
log_likelihood = gmm.score(X_scaled)

print("\nLog-Likelihood:", log_likelihood)

# Display number of iterations
print("Number of EM Iterations:", gmm.n_iter_)

# Display cluster probabilities
print("\nCluster Probabilities for First 5 Samples:")
print(gmm.predict_proba(X_scaled)[:5])

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap="viridis",
    s=50
)

plt.scatter(
    gmm.means_[:, 0],
    gmm.means_[:, 1],
    marker="X",
    s=200,
    color="red",
    label="Cluster Centers"
)

plt.xlabel("Sepal Length (Standardized)")
plt.ylabel("Petal Length (Standardized)")
plt.title("Expectation Maximization - Gaussian Mixture Model")
plt.legend()
plt.show()
