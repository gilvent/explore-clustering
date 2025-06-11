import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from helpers.mean_shift import (
    plot_meanshift_results,
    print_cluster_stats,
)
from clustering.mean_shift import MeanShift


def estimate_bandwidth(X):
    # Use Scott's rule
    n_samples, n_features = X.shape
    return np.power(n_samples, -1.0 / (n_features + 4)) * np.std(X)


def estimate_convergence_threshold(X, factor=0.001):
    # By data scale
    data_range = np.max(X, axis=0) - np.min(X, axis=0)
    return np.mean(data_range) * factor


def main():
    # Load dataset
    dataset = np.loadtxt(fname="data/wifi_localization.txt")
    X = dataset[:, :-1]  # Remove target column
    y_true = dataset[:, -1]  # Target column for external validity measure

    # Estimate bandwidth using Scott's rule
    scott_rule_bandwidth = estimate_bandwidth(X=X)
    print(f"Optimal bandwidth: {scott_rule_bandwidth:.3f}")

    # Estimate convergence threshold by data scale
    convergence_threshold = estimate_convergence_threshold(X=X, factor=0.1)
    
    # Apply Mean-Shift with optimal bandwidth
    ms = MeanShift(
        bandwidth=scott_rule_bandwidth,
        threshold=convergence_threshold,
    )
    labels = ms.fit_predict(X=X)

    # Calculate metrics
    n_clusters = len(set(labels))
    print(f"Number of clusters: {n_clusters}")

    # Internal validity measure
    if n_clusters > 1:
        silhouette_avg = silhouette_score(X, labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")

    # External validity measure
    if len(np.unique(y_true)) > 1:
        ari_score = adjusted_rand_score(labels_true=y_true, labels_pred=labels)
        print(f"Adjusted Rand Index: {ari_score:.3f}")

    # Generate visualizations
    plot_meanshift_results(
        X_scaled=X, labels=labels, cluster_centers=ms.cluster_centers
    )

    print_cluster_stats(labels=labels)


if __name__ == "__main__":
    main()
