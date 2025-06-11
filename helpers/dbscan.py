import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def find_optimal_eps(X, min_pts=5):
    neighbors = NearestNeighbors(n_neighbors=min_pts)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)

    # Get the average distances to other points.
    avg_distances = distances.mean(axis=1)

    return np.sort(avg_distances, axis=0)


def plot_eps_selection(distances, suggested_eps, min_pts):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(distances)), distances, "b-")
    plt.axhline(
        y=suggested_eps,
        color="red",
        linestyle="--",
        label=f"Suggested eps: {suggested_eps:.3f}",
    )
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{min_pts}-NN Distance")
    plt.title("K-Distance Plot for DBSCAN Parameter Selection")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Mark the elbow point
    knee_point = len(distances) // 4  # Rough estimate for elbow
    plt.axvline(
        x=knee_point, color="green", linestyle=":", alpha=0.7, label=f"Elbow region"
    )
    plt.legend()
    plt.show()


def plot_dbscan_results(X, X_scaled, labels, title="DBSCAN Clustering Results"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot 1: Clustering results with noise
    unique_labels = set(labels)
    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "cyan"]

    for k in unique_labels:
        if k == -1:
            # Noise points in black
            class_member_mask = labels == k
            xy = X_pca[class_member_mask]
            ax1.scatter(
                xy[:, 0],
                xy[:, 1],
                c="black",
                marker="x",
                s=50,
                alpha=0.6,
                label="Noise",
            )
        else:
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

    ax1.set_xlabel("First Principal Component")
    ax1.set_ylabel("Second Principal Component")
    ax1.set_title("DBSCAN Results")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Core vs Border vs Noise points
    # This requires running DBSCAN to get core samples
    db = DBSCAN(eps=0.5, min_samples=5).fit(StandardScaler().fit_transform(X))
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Core points
    core_points = X_pca[core_samples_mask & (labels != -1)]
    if len(core_points) > 0:
        ax2.scatter(
            core_points[:, 0],
            core_points[:, 1],
            c="red",
            s=80,
            alpha=0.8,
            label="Core Points",
            marker="o",
        )

    # Border points (non-core, non-noise)
    border_mask = (~core_samples_mask) & (labels != -1)
    border_points = X_pca[border_mask]
    if len(border_points) > 0:
        ax2.scatter(
            border_points[:, 0],
            border_points[:, 1],
            c="blue",
            s=50,
            alpha=0.6,
            label="Border Points",
            marker="s",
        )

    # Noise points
    noise_points = X_pca[labels == -1]
    if len(noise_points) > 0:
        ax2.scatter(
            noise_points[:, 0],
            noise_points[:, 1],
            c="black",
            s=30,
            alpha=0.4,
            label="Noise Points",
            marker="x",
        )

    ax2.set_xlabel("First Principal Component")
    ax2.set_ylabel("Second Principal Component")
    ax2.set_title("DBSCAN Point Types")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_cluster_stats(labels):
    unique_labels = set(labels)
    for k in unique_labels:
        if k == -1:
            print(f"Noise points: {list(labels).count(k)}")
        else:
            print(f"Cluster {k}: {list(labels).count(k)} points")
