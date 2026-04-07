from matplotlib.patches import Ellipse
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pcas
import model
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

digit = 0
mask = (y_train == digit)
X_digit = X_train[mask]
y_digit = y_train[mask]

# reshape for pca
X_digit = X_digit.reshape(X_digit.shape[0], -1)

z,b,m,eigvals = pcas.pca(X_digit,2)

mixture_components = 5
model = model.Model(mixture_components, z)

# run EM for multiple iterations
n_iters = 20
for _ in range(n_iters):
    model.e_step()
    model.m_step()

# visualize in 2D space
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
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))

    # ellipse sizes
    width, height = 3 * np.sqrt(vals) # scaling

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor='black',
        facecolor='none',
        linewidth=1.5,
        alpha=0.7 # opaqueness
    )

    plt.gca().add_patch(ellipse)

    # plot mean
    plt.scatter(mean[0], mean[1], c='red', s=50)


plt.title(f"GMM clustering for digit {digit}")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.show()