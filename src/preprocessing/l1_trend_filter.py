"""
L1 trend filtering used for accelerometer preprocessing.

Adapted from the preprocessing code provided with the
Parkinson's Disease Smartwatch (PADS) dataset.

The original PADS implementation uses an L1 trend filter with
lambda = 50 to estimate the slowly varying accelerometer trend.
"""

import numpy as np
import scipy
import cvxpy as cp


def l1_trend_filter(
    signal: np.ndarray,
    vlambda: float = 50,
    verbose: bool = False,
) -> np.ndarray:
    """
    Estimate the slowly varying trend of a one-dimensional signal.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional sensor signal.
    vlambda : float, default=50
        L1 trend-filter regularization parameter.
        The default follows the original PADS preprocessing.
    verbose : bool, default=False
        Whether to display solver information.

    Returns
    -------
    np.ndarray
        Estimated signal trend.

    Raises
    ------
    ValueError
        If the signal is not one-dimensional or is too short.
    RuntimeError
        If the optimization solver does not converge.
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("L1 trend filtering requires a one-dimensional signal.")

    n = signal.size

    if n < 3:
        raise ValueError("Signal must contain at least 3 samples.")

    # Second-difference matrix used by the original PADS implementation.
    e = np.ones((1, n))

    difference_matrix = scipy.sparse.spdiags(
        np.vstack((e, -2 * e, e)),
        range(3),
        n - 2,
        n,
    )

    trend = cp.Variable(shape=n)

    objective = cp.Minimize(
        0.5 * cp.sum_squares(signal - trend)
        + vlambda * cp.norm(difference_matrix @ trend, 1)
    )

    problem = cp.Problem(objective)

    problem.solve(
        solver=cp.CVXOPT,
        verbose=verbose,
    )

    if problem.status != cp.OPTIMAL:
        raise RuntimeError(
            f"L1 trend filter did not converge. Solver status: {problem.status}"
        )

    return np.asarray(trend.value)