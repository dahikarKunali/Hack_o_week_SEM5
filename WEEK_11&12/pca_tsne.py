import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Original data shape:", X.shape)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled data shape:", X_scaled.shape)

# Apply PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("PCA data shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

X_tsne = tsne.fit_transform(X_scaled)

print("t-SNE data shape:", X_tsne.shape)

# Create plots
plt.figure(figsize=(12, 5))

# PCA plot
plt.subplot(1, 2, 1)

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA")

# t-SNE plot
plt.subplot(1, 2, 2)

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y
)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE")

plt.tight_layout()
plt.show()
