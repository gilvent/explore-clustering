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
    # Preprocessing
    dataset = np.genfromtxt(
        fname="data/wholesale_customers.csv", delimiter=",", dtype=float
    )
    mask = np.ones(dataset.shape[1], dtype=bool)
    mask[1] = False
    X = dataset[1:, mask]
    y_true = dataset[1:, 1]  # Target column to calculate external validity measure

    # Generate 2 one-hot encoding columns for 1st column (categorical feature)
    channel_feature = X[:, 0]
    horeca_encoding = (channel_feature == 1).astype(int)
    retail_encoding = (channel_feature == 2).astype(int)

    # Remove the column, add 2 one-hot encoding columns
    X = X[:, 1:]
    X = np.c_[horeca_encoding, retail_encoding, X]

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
