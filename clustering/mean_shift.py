import numpy as np


class MeanShift:
    def __init__(self, bandwidth=1.0, max_iteration=300, threshold=0.001):
        self.bandwidth = bandwidth
        self.max_iteration = max_iteration
        self.threshold = threshold
        self.cluster_centers = None
        self.labels = None

    def fit_predict(self, X):
        n_samples, n_features = X.shape

        # Initialize centroids at each data point
        centroids = X.copy()

        for i in range(self.max_iteration):
            new_centroids = np.zeros_like(centroids)

            # Shift each centroid
            for i in range(n_samples):
                new_centroids[i] = self._shift_point(centroids[i], X)

            # Check for convergence
            shifts = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1))
            if np.all(shifts < self.threshold):
                break

            centroids = new_centroids

        # Merge nearby centroids to find final cluster centers
        self.cluster_centers = self._merge_centroids(centroids)

        # Assign labels based on closest cluster center
        self.labels = self._assign_labels(X, self.cluster_centers)

        return self.labels

    def _gaussian_kernel(self, distances):
        """Gaussian kernel function"""
        return np.exp(-0.5 * (distances / self.bandwidth) ** 2)

    def _shift_point(self, point, X):
        """Perform one mean shift iteration for a single point"""
        # Calculate distances from point to all data points
        distances = np.sqrt(np.sum((X - point) ** 2, axis=1))

        # Calculate kernel weights
        weights = self._gaussian_kernel(distances)

        # Avoid division by zero
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            return point

        # Calculate weighted mean (new position)
        new_point = np.sum(weights[:, np.newaxis] * X, axis=0) / weights_sum

        return new_point

    def _merge_centroids(self, centroids):
        """Merge centroids that are close to each other"""
        # Use a smaller threshold for merging (fraction of bandwidth)
        merge_threshold = self.bandwidth

        merged_centers = []
        used = np.zeros(len(centroids), dtype=bool)

        for i in range(len(centroids)):
            if used[i]:
                continue

            # Find all centroids within merge threshold
            distances = np.sqrt(np.sum((centroids - centroids[i]) ** 2, axis=1))

            close_indices = np.where(distances <= merge_threshold)[0]

            # Average the close centroids
            center = np.mean(centroids[close_indices], axis=0)
            merged_centers.append(center)

            # Mark as used
            used[close_indices] = True

        return np.array(merged_centers)

    def _assign_labels(self, X, cluster_centers):
        """Assign each point to the nearest cluster center"""
        labels = np.zeros(len(X), dtype=int)

        for i, point in enumerate(X):
            distances = np.sqrt(np.sum((cluster_centers - point) ** 2, axis=1))
            labels[i] = np.argmin(distances)

        return labels
