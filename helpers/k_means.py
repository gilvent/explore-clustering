import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def find_optimal_k(X, max_k=10):
    """Find optimal number of clusters using silhouette score"""
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    return k_range, silhouette_scores


def plot_silhouette_scores(k_range, silhouette_scores):
    """Plot silhouette scores"""
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 5))

    # Silhouette scores
    ax1.plot(k_range, silhouette_scores, "ro-")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Silhouette Score")
    ax1.set_title("Silhouette Score for Different k")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_clusters_2d(X_scaled, labels, cluster_centers, title="K-Means Clustering Results"):
    """Plot clusters in 2D using PCA"""
    
    # PCA for visualization
    print("\nPerforming PCA for visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(cluster_centers)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")


    plt.figure(figsize=(10, 8))

    # Create a color map
    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray"]

    # Plot each cluster
    for i in range(len(np.unique(labels))):
        cluster_points = X_pca[labels == i]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=colors[i % len(colors)],
            label=f"Cluster {i}",
            alpha=0.7,
            s=50,
        )

    # Plot cluster centers
    plt.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        c="black",
        marker="x",
        s=200,
        linewidths=3,
        label="Centroids",
    )

    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_feature_distribution(X, labels, feature_names):
    """Plot distribution of features across clusters"""
    n_features = min(6, X.shape[1])  # Show first 6 features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(n_features):
        for cluster in np.unique(labels):
            cluster_data = X[labels == cluster, i]
            axes[i].hist(cluster_data, alpha=0.7, label=f"Cluster {cluster}", bins=15)

        axes[i].set_title(
            f'Feature {i+1}: {feature_names[i] if i < len(feature_names) else f"Feature_{i+1}"}'
        )
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_cluster_centers(centers, feature_names):
    """Plot cluster centers as a heatmap"""
    plt.figure(figsize=(12, 6))

    # Create heatmap of cluster centers
    sns.heatmap(
        centers,
        xticklabels=feature_names[: centers.shape[1]],
        yticklabels=[f"Cluster {i}" for i in range(centers.shape[0])],
        annot=True,
        cmap="RdYlBu_r",
        center=0,
        fmt=".2f",
    )

    plt.title("Cluster Centers Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Clusters")
    plt.tight_layout()
    plt.show()


def print_cluster_infos(X, labels, feature_names):
    """Analyze cluster characteristics"""
    print("Cluster Analysis:")
    print("=" * 50)

    for cluster in np.unique(labels):
        cluster_data = X[labels == cluster]
        print(f"\nCluster {cluster}:")
        print(
            f"  Size: {len(cluster_data)} samples ({len(cluster_data)/len(X)*100:.1f}%)"
        )
        print(f"  Mean values:")

        for i, feature in enumerate(feature_names[: X.shape[1]]):
            mean_val = np.mean(cluster_data[:, i])
            std_val = np.std(cluster_data[:, i])
            print(f"    {feature}: {mean_val:.2f} ± {std_val:.2f}")

def print_cluster_sizes(cluster_labels):
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster sizes:")
    for cluster, count in zip(unique, counts):
        print(
            f"  Cluster {cluster}: {count} samples ({count/len(cluster_labels)*100:.1f}%)"
        )