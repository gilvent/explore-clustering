import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def test_bandwidth_range(X, quantile_range=[0.1, 0.3, 0.5, 0.7, 0.9]):
    bandwidths = []
    n_clusters_list = []
    silhouette_scores = []

    for quantile in quantile_range:
        bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=len(X))
        if bandwidth > 0:
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            labels = ms.fit_predict(X)
            n_clusters = len(set(labels))

            bandwidths.append(bandwidth)
            n_clusters_list.append(n_clusters)

            if n_clusters > 1:
                sil_score = silhouette_score(X, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(-1)

    return quantile_range, bandwidths, n_clusters_list, silhouette_scores


def plot_bandwidth_analysis(
    quantile_range, bandwidths, n_clusters_list, silhouette_scores
):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Bandwidth vs Quantile
    ax1.plot(quantile_range, bandwidths, "bo-")
    ax1.set_xlabel("Quantile")
    ax1.set_ylabel("Bandwidth")
    ax1.set_title("Bandwidth vs Quantile")
    ax1.grid(True, alpha=0.3)

    # Number of clusters vs Quantile
    ax2.plot(quantile_range, n_clusters_list, "ro-")
    ax2.set_xlabel("Quantile")
    ax2.set_ylabel("Number of Clusters")
    ax2.set_title("Number of Clusters vs Quantile")
    ax2.grid(True, alpha=0.3)

    # Silhouette score vs Quantile
    valid_scores = [s for s in silhouette_scores if s != -1]
    valid_quantiles = [q for q, s in zip(quantile_range, silhouette_scores) if s != -1]

    if valid_scores:
        ax3.plot(valid_quantiles, valid_scores, "go-")
        ax3.set_xlabel("Quantile")
        ax3.set_ylabel("Silhouette Score")
        ax3.set_title("Silhouette Score vs Quantile")
        ax3.grid(True, alpha=0.3)

        # Highlight best quantile
        best_idx = np.argmax(valid_scores)
        ax3.axvline(
            x=valid_quantiles[best_idx],
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Best: {valid_quantiles[best_idx]}",
        )
        ax3.legend()

    plt.tight_layout()
    plt.show()


def plot_meanshift_results(
    X_scaled, labels, cluster_centers, title="Mean-Shift Clustering Results"
):
    # PCA for 2d visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    cluster_centers_pca = pca.transform(cluster_centers)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Clustering results
    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "cyan",
        "magenta",
    ]
    unique_labels = set(labels)

    for k in unique_labels:
        class_member_mask = labels == k
        xy = X_pca[class_member_mask]
        ax1.scatter(
            xy[:, 0],
            xy[:, 1],
            c=colors[k % len(colors)],
            s=50,
            alpha=0.8,
            label=f"Cluster {k}",
        )

    # Plot cluster centers
    ax1.scatter(
        cluster_centers_pca[:, 0],
        cluster_centers_pca[:, 1],
        c="black",
        marker="x",
        s=200,
        linewidths=3,
        label="Centers",
    )

    ax1.set_xlabel("First Principal Component")
    ax1.set_ylabel("Second Principal Component")
    ax1.set_title("Mean-Shift Clustering Results")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Density visualization with cluster centers
    # Create a scatter plot with varying point sizes based on local density
    from scipy.spatial.distance import cdist

    # Calculate local density (inverse of mean distance to k nearest neighbors)
    k_neighbors = min(5, len(X_pca) - 1)
    distances = cdist(X_pca, X_pca)
    distances[distances == 0] = np.inf  # Remove self-distances
    nearest_distances = np.sort(distances, axis=1)[:, :k_neighbors]
    density = 1 / (np.mean(nearest_distances, axis=1) + 1e-10)

    # Normalize density for visualization
    density_normalized = (density - density.min()) / (density.max() - density.min())
    sizes = 20 + density_normalized * 80  # Scale between 20 and 100

    for k in unique_labels:
        class_member_mask = labels == k
        xy = X_pca[class_member_mask]
        cluster_sizes = sizes[class_member_mask]
        ax2.scatter(
            xy[:, 0],
            xy[:, 1],
            c=colors[k % len(colors)],
            s=cluster_sizes,
            alpha=0.6,
            label=f"Cluster {k}",
        )

    # Plot cluster centers with special marker
    ax2.scatter(
        cluster_centers_pca[:, 0],
        cluster_centers_pca[:, 1],
        c="black",
        marker="*",
        s=300,
        linewidths=2,
        label="Mode Centers",
        edgecolors="white",
    )

    ax2.set_xlabel("First Principal Component")
    ax2.set_ylabel("Second Principal Component")
    ax2.set_title("Mean-Shift Results (Point Size ∝ Local Density)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_cluster_stats(labels):
    unique_labels = set(labels)
    for k in unique_labels:
        print(f"Cluster {k}: {list(labels).count(k)} points")
