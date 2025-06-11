import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from helpers.dbscan import (
    plot_dbscan_results,
    plot_eps_selection,
    find_optimal_eps,
    print_cluster_stats,
)


def main():
    dataset = np.loadtxt(fname="data/wifi_localization.txt")
    X = dataset[:, :-1]  # Remove target column
    y_true = dataset[:, -1]  # Target column for external validity measure

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal eps using k-distance plot
    min_pts = X.shape[1]
    distances = find_optimal_eps(X_scaled, min_pts)

    # Suggested eps: 90th percentile of distances
    suggested_eps = np.percentile(distances, 90)

    # Plot k-distance for parameter selection
    plot_eps_selection(
        distances=distances, suggested_eps=suggested_eps, min_pts=min_pts
    )

    # Apply DBSCAN
    eps = suggested_eps
    dbscan = DBSCAN(eps=eps, min_samples=min_pts)
    labels = dbscan.fit_predict(X_scaled)

    # Calculate metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise points: {n_noise}")
    print(f"eps: {eps:.3f}, min_samples: {min_pts}")

    # Internal validity measure: Silhouette score
    if n_clusters > 1:
        # Remove noise points for silhouette calculation
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 1:
            silhouette_avg = silhouette_score(
                X_scaled[non_noise_mask], labels[non_noise_mask]
            )
            print(f"Silhouette Score: {silhouette_avg:.3f}")

    # External validity measure: Adjusted Rand-Score
    if len(np.unique(y_true)) > 1:
        ari_score = adjusted_rand_score(labels_true=y_true, labels_pred=labels)
        print(f"Adjusted Rand Index: {ari_score:.3f}")

    # Generate visualizations
    plot_dbscan_results(X=X, X_scaled=X_scaled, labels=labels)

    print_cluster_stats(labels=labels)


if __name__ == "__main__":
    main()
