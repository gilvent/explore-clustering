import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import spectral_embedding


def test_spectral_parameters(
    X,
    n_clusters_range=[2, 3, 4, 5],
    affinity="nearest_neighbors",
    gamma_range=[0.1, 1.0, 10.0],
):
    results = []

    def test_per_gamma(n_clusters):
        for gamma in gamma_range:
            try:
                sc = SpectralClustering(
                    n_clusters=n_clusters,
                    gamma=gamma,
                    affinity="rbf",
                    random_state=42,
                    n_init=10,
                    assign_labels="discretize",
                )
                labels = sc.fit_predict(X)

                if len(set(labels)) > 1:
                    sil_score = silhouette_score(X, labels)
                else:
                    sil_score = -1

                results.append(
                    {
                        "n_clusters": n_clusters,
                        "gamma": gamma,
                        "silhouette": sil_score,
                        "labels": labels,
                    }
                )
            except Exception as e:
                continue

    for n_clusters in n_clusters_range:
        if affinity == "rbf":
            test_per_gamma(n_clusters=n_clusters)
        else:
            sc = SpectralClustering(
                n_clusters=n_clusters,
                affinity="nearest_neighbors",
                random_state=42,
                n_init=10,
                assign_labels="kmeans",
            )
            labels = sc.fit_predict(X)

            if len(set(labels)) > 1:
                sil_score = silhouette_score(X, labels)
            else:
                sil_score = -1

            results.append(
                {
                    "n_clusters": n_clusters,
                    "silhouette": sil_score,
                    "labels": labels,
                }
            )

    return results


def plot_silhouette_scores(results, affinity="nearest_neighbors"):
    """Plot the silhouette scores across different number of clusters

    Parameters
    ----------

    results : list<dict>
        Obtained from test_spectral_parameters()
    """

    # Convert results to arrays for plotting
    n_clusters_list = [r["n_clusters"] for r in results]
    silhouette_list = [r["silhouette"] for r in results]
    gamma_list = [r["gamma"] for r in results] if affinity == "rbf" else None

    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 6))

    # Plot for each gamma for affinity="rbf"
    if affinity == "rbf":
        for gamma in sorted(set(gamma_list)):
            scores = []
            n_clust = []
            for r in results:
                if r["gamma"] == gamma and r["silhouette"] != -1:
                    scores.append(r["silhouette"])
                    n_clust.append(r["n_clusters"])

            if scores:
                ax1.plot(n_clust, scores, "o-", label=f"γ={gamma}")
    else:
        unique_n_clusters = sorted(set(n_clusters_list))
        ax1.plot(unique_n_clusters, silhouette_list, "ro-")

    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Silhouette Score")
    ax1.set_title("Spectral Clustering: Silhouette Score vs Parameters")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_to_2d(X_2d, labels, title):
    fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 6))

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
        xy = X_2d[class_member_mask]
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
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


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
    

    print(f"Dataset shape: {X.shape}")
    print(f"Number of true classes: {len(np.unique(y_true))}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    affinity = "nearest_neighbors"
    # Test different parameters
    results = test_spectral_parameters(
        X_scaled, n_clusters_range=[2, 3, 4], affinity=affinity
    )

    plot_silhouette_scores(results)

    # Find best parameters
    best_result = max(
        results, key=lambda x: x["silhouette"] if x["silhouette"] != -1 else -2
    )

    best_n_clusters = best_result["n_clusters"]
    print(f"\nBest n_clusters={best_n_clusters}")

    # When using affinity = "rbf"
    # best_gamma = best_result["gamma"]
    # print(f"\nBest gamma={best_gamma}")

    print(f"Best silhouette score: {best_result['silhouette']:.3f}")

    # Apply Spectral Clustering with best parameters
    print("Applying spectral clustering with best parameters...")
    n_neighbors = 10
    sc = SpectralClustering(
        n_clusters=best_n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        random_state=42,
        assign_labels="kmeans",
    )
    labels = sc.fit_predict(X_scaled)

    # Visualize spectral embedding
    embedding_2d = spectral_embedding(
        sc.affinity_matrix_, n_components=2, random_state=42
    )

    plot_to_2d(X_2d=embedding_2d, labels=labels, title="Spectral Embedding")

    # Visualize clusters on 2D space
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plot_to_2d(X_2d=X_pca, labels=labels, title="Spectral Clustering")

    # Print cluster statistics
    unique_labels = set(labels)
    n_clusters = len(unique_labels)
    print(f"\nFinal Results:")
    print(f"Number of clusters found: {n_clusters}")

    for k in unique_labels:
        print(f"Cluster {k}: {list(labels).count(k)} points")

    # Internal validity measure
    if n_clusters > 1:
        silhouette_avg = silhouette_score(X_scaled, labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")

    # External validity measure
    if len(np.unique(y_true)) > 1:
        ari_score = adjusted_rand_score(labels_true=y_true, labels_pred=labels)
        print(f"Adjusted Rand Index: {ari_score:.3f}")


if __name__ == "__main__":
    main()
