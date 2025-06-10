import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from helpers.mean_shift import (
    test_bandwidth_range,
    plot_bandwidth_analysis,
    plot_meanshift_results,
    print_cluster_stats,
)


def main():
    # Load dataset
    dataset = np.loadtxt(fname="data/wifi_localization.txt")
    X = dataset[:, :-1]  # Remove target column
    y_true = dataset[:, -1]  # Target column for external validity measure

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test different bandwidth parameters
    quantile_range, bandwidths, n_clusters_list, silhouette_scores = (
        test_bandwidth_range(X_scaled)
    )

    # Plot bandwidth analysis
    plot_bandwidth_analysis(
        quantile_range, bandwidths, n_clusters_list, silhouette_scores
    )

    # Select optimal bandwidth based on silhouette score
    valid_scores = [s for s in silhouette_scores if s != -1]
    if valid_scores:
        best_idx = np.argmax(valid_scores)
        optimal_quantile = quantile_range[best_idx]
        optimal_bandwidth = bandwidths[best_idx]
    else:
        optimal_quantile = 0.3
        optimal_bandwidth = estimate_bandwidth(
            X_scaled, quantile=optimal_quantile, n_samples=len(X_scaled)
        )

    print(f"Optimal quantile: {optimal_quantile}")
    print(f"Optimal bandwidth: {optimal_bandwidth:.3f}")

    # Apply Mean-Shift with optimal bandwidth
    ms = MeanShift(bandwidth=optimal_bandwidth, bin_seeding=True)
    labels = ms.fit_predict(X_scaled)
    cluster_centers = ms.cluster_centers_

    # Calculate metrics
    n_clusters = len(set(labels))
    print(f"Number of clusters: {n_clusters}")

    # Internal validity measure
    if n_clusters > 1:
        silhouette_avg = silhouette_score(X_scaled, labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")

    # External validity measure
    if len(np.unique(y_true)) > 1:
        ari_score = adjusted_rand_score(labels_true=y_true, labels_pred=labels)
        print(f"Adjusted Rand Index: {ari_score:.3f}")

    # Generate visualizations
    plot_meanshift_results(
        X_scaled=X_scaled, labels=labels, cluster_centers=cluster_centers
    )

    print_cluster_stats(labels=labels)


if __name__ == "__main__":
    main()
