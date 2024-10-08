"""Utility functions."""
import os
import time
import gzip
import pickle

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.utils import check_random_state
from sklearn.model_selection._split import _validate_shuffle_split
import matplotlib.pyplot as plt

from . import IMCProblem


def make_imc_data(n_1, d_1, n_2, d_2, k, scale=0.05, noise=0, random_state=None,
                  binarize=False, return_noisy_only=True):
    """Create a very simple IMC problem.

    TODO: should probably utilize sklearn's make_regresssion.
    """
    random_state = check_random_state(random_state)

    assert d_1 >= k and d_2 >= k
    if not isinstance(scale, (tuple, list)):
        assert isinstance(scale, float) and scale > 0
        scale = scale, scale

    X_scale, Y_scale = scale
    X = random_state.normal(scale=X_scale, size=(n_1, d_1))
    Y = random_state.normal(scale=Y_scale, size=(n_2, d_2))

    # fixed weights -- first k features are informative
    W, H = np.eye(d_1, k), np.eye(d_2, k)

    R = np.dot(np.dot(X, W), np.dot(Y, H).T)
    R_noise = R.copy()
    if noise > 0:
        R_noise += random_state.normal(scale=noise, size=(n_1, n_2))

    if binarize:
        # We use $\pm 1$ labels in the classification problem.
        R = np.where(R >= 0, 1., -1.)
        R_noise = np.where(R_noise >= 0, 1., -1.)

    if return_noisy_only:
        return X, W, Y, H, R_noise
    else:
        return X, W, Y, H, R_noise, R


def sparsify(R, sparsity=0.10, random_state=None):
    """Sparsify the given matrix."""
    random_state = check_random_state(random_state)

    mask = random_state.uniform(size=R.shape) < sparsity
    R_coo = coo_matrix((R[mask], np.nonzero(mask)),
                       shape=R.shape, dtype=np.float64)

    return R_coo.tocsr(), mask


def sparsify_with_mask(R, mask):
    """Sparcify the given matrix with the mask."""
    R_coo = coo_matrix((R[mask], np.nonzero(mask)),
                       shape=R.shape, dtype=np.float64)

    return R_coo.tocsr()


def save(obj, path, filename=None, gz=None):
    """Pickle a pythonic `obj` into a file given by `path`.

    Parameters
    ----------
    obj: any python object
        An object to pickle.

    path: string
        A file in which to pickle the object.

    filename: string, optinal
        Specify filename for re-building experiments results. If None - will
        be saved as time.

    gz: integer, or None, optinal
        If None, then does not apply compression while pickling. Otherwise
        must be an integer 0-9 which determines the level of GZip compression:
        the lower the level the less thorough but the more faster the
        compression is. Value `0` produces a GZip archive with no compression
        whatsoever, whereas the value of `9` produces the most compressed
        archive.

    Returns
    -------
    filename: string
        The name of the resulting archive.
    """
    if not(gz is None or (isinstance(gz, int) and 0 <= gz <= 9)):
        raise TypeError("""`gz` parameter must be either `None` """
                        """or an integer 0-9.""")

    if not os.path.isdir(path):
        os.makedirs(path)

    open_ = open if gz is None else lambda f, m: gzip.open(f, m, gz)
    if filename is None:
        filename_ = "%s-%s.%s" % (path, time.strftime("%Y%m%d_%H%M%S"),
                                  "pic" if gz is None else "gz")
    else:
        filename_ = "%s%s%s" % (path, filename,
                                '.pic' if gz is None else '.gz')

    with open_(filename_, "wb+") as f:
        pickle.dump(obj, f)
    if filename is None:
        return filename_


def load(filename):
    """Recover an object from the file identified by `filename`.

    Parameters
    ----------
    filename: string
        A `file` in which an object is pickled.

    Returns
    -------
    object: a python object
        The recovered pythonic object.
    """
    open_ = open if not filename.endswith(".gz") else gzip.open

    with open_(filename, "rb") as f:
        obj = pickle.load(f)

    return obj


def mc_split(R, n_splits=1, test_size='default', train_size=None, random_state=None):
    """Train-test splitting."""
    n_samples = np.prod(R.shape)
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size)

    rng = check_random_state(random_state)
    for i in range(n_splits):
        permutation = rng.permutation(n_samples)
        ind_test = permutation[:n_test]
        ind_train = permutation[n_test:(n_test + n_train)]

        yield ind_train, ind_test


def get_submatrix(mat, indices):
    """Extracts a sparse submatrix from a dense one accorind to the provided indices."""
    nz_ij = np.unravel_index(indices, mat.shape)
    subm = coo_matrix((mat.flat[indices], nz_ij), shape=mat.shape)
    return subm.tocsr()


# =================== some questionable functions =========================================


def performance(problem, W, H, C, R_full):
    """Compute the performance of the IMC estimates."""

    assert isinstance(problem, IMCProblem), \
        """`problem` must be an IMC problem."""

    assert W.ndim == H.ndim, """Mismatching dimensionality."""
    if W.ndim < 3:
        W, H = np.atleast_3d(W, H)

    n_iterations = W.shape[-1]
    assert W.shape[-1] == H.shape[-1], """Mismatching number of iterations."""

    # sparsitry coefficients
    sparsity_W = np.isclose(W, 0).mean(axis=(0, 1))
    sparsity_H = np.isclose(H, 0).mean(axis=(0, 1))

    # Regularization -- components
    reg_ridge = (0.5 * np.linalg.norm(W, "fro", axis=(0, 1))**2 +
                 0.5 * np.linalg.norm(H, "fro", axis=(0, 1))**2)

    reg_group = (np.linalg.norm(W, 2, axis=1).sum(axis=0) +
                 np.linalg.norm(H, 2, axis=1).sum(axis=0))

    reg_lasso = (np.linalg.norm(W.reshape(-1, W.shape[-1]), 1, axis=0) +
                 np.linalg.norm(H.reshape(-1, H.shape[-1]), 1, axis=0))

    # Regularization -- full
    C_lasso, C_group, C_ridge = C
    regularizer_value = (C_group * reg_group +
                         C_lasso * reg_lasso +
                         C_ridge * reg_ridge)

    # sequential forbenius norm of the matrices
    div_W = np.r_[np.linalg.norm(np.diff(W, axis=-1),
                                 "fro", axis=(0, 1)), 0]
    div_H = np.r_[np.linalg.norm(np.diff(H, axis=-1),
                                 "fro", axis=(0, 1)), 0]

    # Objective value on the train data
    v_val_train = np.array([problem.value(W[..., i], H[..., i])
                            for i in range(n_iterations)])

    # Objective on the full matrix (expensive!)
    v_val_full = np.array([
        problem.loss(problem.prediction(W[..., i], H[..., i]).ravel(),
                     R_full.ravel()).sum() for i in range(n_iterations)])

    # Score on the full matrix (expensive!)
    score_full = np.array([
        problem.score(problem.prediction(W[..., i], H[..., i]).ravel(),
                      R_full.ravel()) for i in range(n_iterations)])

    metrics = np.stack([v_val_train, regularizer_value,
                        score_full, v_val_full,
                        sparsity_W, sparsity_H,
                        div_W, div_H], axis=0)

    titles = ['Observed Elements', 'Regularization', 'Score',
              'Full Matrix', 'Zero Values of W', 'Zero Values of H',
              'L2-Norm Variation W', 'L2-Norm Variation H']

    units = ['L2-Loss', 'L2-Loss', 'Score', 'L2-Loss',
             '%', '%', 'L2-Norm', 'L2-Norm']

    return metrics, titles, units


def plot_WH(W, H):
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(121, title="$W$", xticks=[])
    ax.imshow(np.concatenate(
        [W, np.full_like(W[:, :1], W.min()),
         np.where(np.linalg.norm(W, axis=1, keepdims=True) > 0,
                  W.max(), W.min())],
        axis=-1), cmap=plt.cm.hot)

    ax = fig.add_subplot(122, title="$H$", xticks=[])
    ax.imshow(np.concatenate(
        [H, np.full_like(H[:, :1], H.min()),
         np.where(np.linalg.norm(H, axis=1, keepdims=True) > 0,
                  H.max(), H.min())],
        axis=-1), cmap=plt.cm.hot)

    fig.tight_layout()
    plt.show()
    plt.close()


def plot_loss(metrics, titles, units, fig_size=4, max_cols=None, **kwargs):
    n_plots = metrics.shape[0]
    assert n_plots == len(titles) == len(units)

    if not isinstance(max_cols, int):
        n_rows, n_cols = 1, n_plots
    else:
        n_cols = min(max_cols, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(fig_size * n_cols, fig_size * n_rows))

    # remove potentially troublesom arguments
    kwargs.pop("ylabel", None)
    kwargs.pop("title", None)

    # plot
    iterations = np.arange(metrics.shape[1])
    for i, (metric, title, units) in enumerate(zip(metrics, titles, units)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, ylabel=units,
                             title=title, **kwargs)

        ax.plot(iterations, metric, color="red",
                linewidth=2, linestyle="solid")

    fig.tight_layout()
    plt.show()
    plt.close()
