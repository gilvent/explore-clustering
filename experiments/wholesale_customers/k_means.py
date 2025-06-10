import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from helpers.k_means import (
    find_optimal_k,
    plot_silhouette_scores,
    plot_feature_distribution,
    plot_cluster_centers,
    plot_clusters_2d,
    print_cluster_infos,
    print_cluster_sizes,
    plot_ground_truth_2d,
)
from sklearn.metrics import silhouette_score, adjusted_rand_score


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

    # Normalize target value to start from 0
    y_true = y_true.astype('int')
    y_min = np.min(y_true, axis=0)
    y_true = y_true - y_min

    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    # Feature names
    feature_names = [
        "Fresh",
        "Milk",
        "Grocery",
        "Frozen",
        "Detergents Paper",
        "Delicassen",
        "Channel (HORECA)",
        "Channel (Retail)"
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
    optimal_k = 4  # k_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal k based on silhouette score: {optimal_k}")

    # Perform K-means clustering
    print(f"\nPerforming K-means clustering with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_

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

    plot_ground_truth_2d(
        X_scaled=X_scaled,
        y_true=y_true,
    )

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
