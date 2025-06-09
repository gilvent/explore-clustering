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
    # Load dataset
    dataset = np.loadtxt(fname="data/wifi_localization.txt")
    X = dataset[:, :-1]  # Remove target column
    y_true = dataset[:, -1]  # Target column for external validity measure

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
