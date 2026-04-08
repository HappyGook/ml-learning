import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


def plot_gmm_2d(z, model, digit):
    plt.figure(figsize=(6, 5))
    # data points
    plt.scatter(z[:, 0], z[:, 1], s=5, alpha=0.3)

    for k in range(model.k):
        mean = model.means[k]
        covar = model.covariances[k]

        # eigendecomposition for the ellipse axes
        vals, vecs = np.linalg.eigh(covar)

        # sort the eigenvalues
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # angle for the ellipses
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        # ellipse sizes
        width, height = 3 * np.sqrt(vals)  # scaling

        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor='black',
            facecolor='none',
            linewidth=1.5,
            alpha=0.7  # opaqueness
        )

        plt.gca().add_patch(ellipse)

        # plot mean
        plt.scatter(mean[0], mean[1], c='red', s=50)

    plt.title(f"GMM clustering for digit {digit}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.tight_layout()
    plt.show()