import numpy as np
from picard import picard
from .reduce_data import reduce_data
from sklearn.utils.extmath import randomized_svd
# taken from https://github.com/hugorichard/multiviewica
def groupica(
    X,
    n_components=None,
    dimension_reduction="pca",
    max_iter=10000,
    random_state=None,
    tol=1e-7,
    ortho=False,
    extended=False,
    S_true=None,
        c=None
):
    """
    Performs PCA on concatenated data across groups (ex: subjects)
    and apply ICA on reduced data.

    Parameters
    ----------
    X : np array of shape (n_groups, n_features, n_samples)
        Training vector, where n_groups is the number of groups,
        n_samples is the number of samples and
        n_components is the number of components.
    n_components : int, optional
        Number of components to extract.
        If None, no dimension reduction is performed
    dimension_reduction: str, optional
        if srm: use srm to reduce the data
        if pca: use group specific pca to reduce the data
    max_iter : int, optional
        Maximum number of iterations to perform
    random_state : int, RandomState instance or None, optional (default=None)
        Used to perform a random initialization. If int, random_state is
        the seed used by the random number generator; If RandomState
        instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance
        used by np.random.
    tol : float, optional
        A positive scalar giving the tolerance at which
        the un-mixing matrices are considered to have converged.
    ortho: bool, optional
        If True, uses Picard-O. Otherwise, uses the standard Picard.
    extended: None or bool, optional
        If True, uses the extended algorithm to separate sub and
        super-Gaussian sources.
        By default, True if ortho == True, False otherwise.

    Returns
    -------
    P : np array of shape (n_groups, n_components, n_features)
        P is the projection matrix that projects data in reduced space
    W : np array of shape (n_groups, n_components, n_components)
        Estimated un-mixing matrices
    S : np array of shape (n_components, n_samples)
        Estimated source


    See also
    --------
    permica
    multiviewica
    """
    P, X = reduce_data(
        X, n_components=n_components, dimension_reduction=dimension_reduction
    )
    n_pb, p, n = X.shape
    X_concat = np.vstack(X)

    U, S, V = randomized_svd(X_concat, n_components=p)
    if c is not None:
        U, S, V = randomized_svd(X_concat, n_components=2*p-c)
    X_reduced = np.diag(S).dot(V)
    U = np.split(U, n_pb, axis=0)
    K, W, S = picard(
        X_reduced,
        ortho=False,
        extended=extended,
        centering=False,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    scale = np.linalg.norm(S, axis=1)
    S = S / scale[:, None]
    W = np.array([S.dot(np.linalg.pinv(x)) for x in X])
    return P, W, S
