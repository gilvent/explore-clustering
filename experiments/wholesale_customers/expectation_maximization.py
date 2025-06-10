import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from helpers.expectation_maximization import (
    find_optimal_components,
    plot_model_selection,
    plot_em_results,
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

    # Find optimal number of components
    n_components_range, bic_scores, aic_scores = find_optimal_components(X_scaled)
    optimal_bic, optimal_aic = plot_model_selection(
        n_components_range, bic_scores, aic_scores
    )

    # Use BIC optimal for final model
    optimal_n = optimal_bic
    print(f"Optimal number of components (BIC): {optimal_n}")

    # Fit Gaussian Mixture Model
    gm = GaussianMixture(n_components=optimal_n, random_state=42)
    labels = gm.fit_predict(X_scaled)
    probabilities = gm.predict_proba(X_scaled)

    # Internal validity measure
    silhouette_avg = silhouette_score(X_scaled, labels)

    # External validity measure
    if len(np.unique(y_true)) > 1:
        ari_score = adjusted_rand_score(labels_true=y_true, labels_pred=labels)
        print(f"Adjusted Rand Index: {ari_score:.3f}")

    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"BIC Score: {gm.bic(X_scaled):.3f}")
    print(f"AIC Score: {gm.aic(X_scaled):.3f}")

    # Generate visualizations
    plot_em_results(
        X_scaled=X_scaled,
        labels=labels,
        probabilities=probabilities,
        gaussian_mixture_means=gm.means_,
    )


if __name__ == "__main__":
    main()
