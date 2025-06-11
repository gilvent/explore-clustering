import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k=3, max_iteration=100, diff_threshold=0.0001):
        self.k = k
        self.max_iteration = max_iteration
        self.diff_threshold = diff_threshold
        self.cluster_centers = None
        self.labels = None

    def fit_predict(self, X):
        # Initialize k prototypes (cluster centers)
        n_samples, n_features = X.shape
        self.cluster_centers = X[np.random.choice(n_samples, self.k, replace=False)]

        for i in range(self.max_iteration):
            # Assign each point to closest centers
            distances = np.sqrt(
                ((X - self.cluster_centers[:, np.newaxis]) ** 2).sum(axis=2)
            )
            
            self.labels = np.argmin(distances, axis=0)

            # Recalculate prototype using means
            new_cluster_centers = np.array(
                [X[self.labels == i].mean(axis=0) for i in range(self.k)]
            )

            # Check for any change in prototype
            if np.all(
                np.abs(self.cluster_centers - new_cluster_centers) < self.diff_threshold
            ):
                break

            self.cluster_centers = new_cluster_centers

        return self.labels
