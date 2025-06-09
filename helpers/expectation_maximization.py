import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


def find_optimal_components(X, max_components=8):
    bic_scores = []
    aic_scores = []
    n_components_range = range(1, max_components + 1)

    for n in n_components_range:
        gm = GaussianMixture(n_components=n, random_state=42)
        gm.fit(X)
        bic_scores.append(gm.bic(X))
        aic_scores.append(gm.aic(X))

    return n_components_range, bic_scores, aic_scores


def plot_model_selection(n_components_range, bic_scores, aic_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, bic_scores, "bo-", label="BIC")
    plt.plot(n_components_range, aic_scores, "ro-", label="AIC")
    plt.xlabel("Number of Components")
    plt.ylabel("Information Criterion")
    plt.title("EM Model Selection: BIC and AIC Scores")
    plt.legend()
    plt.grid(True, alpha=0.3)

    optimal_bic = n_components_range[np.argmin(bic_scores)]
    optimal_aic = n_components_range[np.argmin(aic_scores)]

    plt.axvline(
        x=optimal_bic,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"Optimal BIC: {optimal_bic}",
    )
    plt.axvline(
        x=optimal_aic,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Optimal AIC: {optimal_aic}",
    )
    plt.legend()
    plt.show()

    return optimal_bic, optimal_aic


def plot_em_results(
    X_scaled,
    labels,
    probabilities,
    gaussian_mixture_means,
    title="EM Clustering Results",
):
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    means_pca = pca.transform(gaussian_mixture_means)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Hard clustering results
    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray"]
    for i in range(len(np.unique(labels))):
        cluster_points = X_pca[labels == i]
        ax1.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=colors[i % len(colors)],
            label=f"Component {i}",
            alpha=0.7,
            s=50,
        )

    # Plot cluster centers (means)
    ax1.scatter(
        means_pca[:, 0],
        means_pca[:, 1],
        c="black",
        marker="x",
        s=200,
        linewidths=3,
        label="Means",
    )

    ax1.set_xlabel("First Principal Component")
    ax1.set_ylabel("Second Principal Component")
    ax1.set_title("Hard Assignment")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Soft clustering (membership probabilities)
    # Use the maximum probability for coloring, but adjust alpha by confidence
    max_probs = np.max(probabilities, axis=1)
    for i in range(len(np.unique(labels))):
        cluster_points = X_pca[labels == i]
        cluster_probs = max_probs[labels == i]
        ax2.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=colors[i % len(colors)],
            alpha=cluster_probs,
            s=50,
            label=f"Component {i}",
        )

    ax2.scatter(
        means_pca[:, 0],
        means_pca[:, 1],
        c="black",
        marker="x",
        s=200,
        linewidths=3,
        label="Means",
    )

    ax2.set_xlabel("First Principal Component")
    ax2.set_ylabel("Second Principal Component")
    ax2.set_title("Soft Assignment (Alpha = Confidence)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
