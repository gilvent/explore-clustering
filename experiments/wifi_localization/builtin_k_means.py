import numpy as np
from sklearn.preprocessing import StandardScaler
from helpers.k_means import (
    plot_silhouette_scores,
    plot_feature_distribution,
    plot_cluster_centers,
    plot_clusters_2d,
    print_cluster_infos,
    print_cluster_sizes,
)
from sklearn.metrics import silhouette_score, adjusted_rand_score
from clustering.k_means import KMeans


def find_optimal_k(X, max_k=10):
    """Find optimal number of clusters using silhouette score"""
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        k_means = KMeans(k=k)
        cluster_labels = k_means.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, cluster_labels))

    return k_range, silhouette_scores


def main():

    dataset = np.loadtxt(fname="data/wifi_localization.txt")
    X = dataset[:, :-1]  # Remove target column
    y_true = dataset[:, -1]  # Target column for external validity measure

    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    # Feature names
    feature_names = [
        "Phone 1 Signal",
        "Phone 2 Signal",
        "Phone 3 Signal",
        "Phone 4 Signal",
        "Phone 5 Signal",
        "Phone 6 Signal",
        "Phone 1 Signal",
    ]

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal number of clusters using silhouette scores
    print("\nFinding optimal number of clusters...")
    k_range, silhouette_scores = find_optimal_k(X_scaled, max_k=8)

    # Plot the silhouette scores
    plot_silhouette_scores(k_range, silhouette_scores)

    # Select optimal k (can be manually adjusted based on the plots)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal k based on silhouette score: {optimal_k}")

    # Perform K-means clustering
    k_means = KMeans(k=optimal_k)
    cluster_labels = k_means.fit_predict(X=X_scaled)

    # Get cluster centers
    cluster_centers = k_means.cluster_centers

    # Internal validity measure
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"\nClustering Results:")
    print(f"Silhouette Score: {silhouette_avg:.3f}")

    # External validity measure
    if len(np.unique(y_true)) > 1:
        ari_score = adjusted_rand_score(labels_true=y_true, labels_pred=cluster_labels)
        print(f"Adjusted Rand Index: {ari_score:.3f}")

    # Visualizations
    print("\nGenerating visualizations...")

    plot_clusters_2d(
        X_scaled=X_scaled,
        labels=cluster_labels,
        cluster_centers=cluster_centers,
        title=f"K-Means Clustering (k={optimal_k}) - PCA Projection",
    )

    # Additional infos
    plot_feature_distribution(X_scaled, cluster_labels, feature_names)

    plot_cluster_centers(cluster_centers, feature_names)

    print_cluster_infos(X_scaled, cluster_labels, feature_names)

    print_cluster_sizes(cluster_labels=cluster_labels)


if __name__ == "__main__":
    main()
