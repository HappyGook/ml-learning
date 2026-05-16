import numpy as np
from scipy.linalg import eig, eigvals


def pca(x, n_components):
    # Subtract mean
    m = np.mean(x, axis=0)
    xc = x - m

    # covariance matrix
    s = (xc.T @ xc) / xc.shape[0]

    # eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(s)

    # sort indices and revers the array (largest values first)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]

    eigvecs = eigvecs[:,idx] # columns reordered to have the vectors with biggest values first

    # new coordinate matrix (matrix of transformation)
    b = eigvecs[:,:n_components] # save only n first

    # projection's coordinates' set
    z = xc @ b

    return z, b, m, eigvals


def pca_svd(x, n_components):
    mu = np.mean(x, axis=0)
    xc = x - mu

    # svd decomposition
    u, s, vt = np.linalg.svd(xc, full_matrices=False)

    # projection coordinates
    b = vt.T[:, :n_components]
    z = xc @ b

    return z, b, mu

def generate_from_pca(b, m, eigval_set, n_components):
    # compute the data variance
    sigma2 = np.mean(eigval_set[n_components:])

    # infer the data dimensions (to avoid input pass)
    dims = b.shape[0]

    # sample z from standard normal N(0,I)
    z = np.random.randn(n_components)

    # get W = B * sqrt(Lambda)
    # Take the list of eigenvalues, cap at n and diagonalize
    w = b @ np.diag(np.sqrt(eigval_set[:n_components]))

    # sample noise in data space (not latent space of n components)
    noise = np.random.randn(dims) * sigma2

    x = w @ z + m + noise

    return x
