"""
L1 trend filtering for PADS accelerometer preprocessing.

The optimization formulation and regularization parameter (lambda = 50)
follow the original PADS preprocessing approach.

CLARABEL is used consistently as the numerical solver. Both OPTIMAL and
OPTIMAL_INACCURATE solutions are retained, while solver status and warnings
are recorded for quality auditing.
"""

from __future__ import annotations

import warnings
from typing import Any

import cvxpy as cp
import numpy as np
import scipy.sparse


DEFAULT_LAMBDA = 50.0
SOLVER = "CLARABEL"


def _build_second_difference_matrix(
    n_samples: int,
) -> scipy.sparse.spmatrix:
    """
    Build the second-difference matrix used by the L1 trend filter.

    Parameters
    ----------
    n_samples : int
        Number of signal samples.

    Returns
    -------
    scipy.sparse.spmatrix
        Sparse second-difference matrix.
    """
    e = np.ones((1, n_samples))

    return scipy.sparse.spdiags(
        np.vstack((e, -2 * e, e)),
        range(3),
        n_samples - 2,
        n_samples,
    )


def l1_trend_filter(
    signal: np.ndarray,
    vlambda: float = DEFAULT_LAMBDA,
    verbose: bool = False,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Estimate the slowly varying trend of a one-dimensional signal.

    The optimization problem is:

        0.5 * ||y - x||_2^2 + lambda * ||D x||_1

    where:
        y = observed signal
        x = estimated trend
        D = second-difference matrix

    CLARABEL is used as the sole numerical solver.

    OPTIMAL and OPTIMAL_INACCURATE are accepted. The exact status is
    returned in the solver metadata so reduced-accuracy solutions can
    be audited later.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional input signal.
    vlambda : float, default=50
        L1 trend-filter regularization parameter.
    verbose : bool, default=False
        Whether to display solver output.
    return_info : bool, default=False
        If True, also return solver information.

    Returns
    -------
    np.ndarray
        Estimated slowly varying signal trend.

    tuple[np.ndarray, dict], optional
        Trend and solver metadata when return_info=True.

    Raises
    ------
    ValueError
        If the input signal is invalid.
    RuntimeError
        If CLARABEL does not produce an acceptable solution.
    """
    signal = np.asarray(
        signal,
        dtype=float,
    )

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if signal.ndim != 1:
        raise ValueError(
            "L1 trend filtering requires a one-dimensional signal."
        )

    if signal.size < 3:
        raise ValueError(
            "Signal must contain at least 3 samples."
        )

    if not np.isfinite(signal).all():
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    if vlambda < 0:
        raise ValueError(
            "vlambda must be greater than or equal to zero."
        )

    if SOLVER not in cp.installed_solvers():
        raise RuntimeError(
            f"{SOLVER} is not installed. "
            f"Installed solvers: {cp.installed_solvers()}"
        )

    n_samples = signal.size

    # --------------------------------------------------------
    # Build optimization problem
    # --------------------------------------------------------

    difference_matrix = _build_second_difference_matrix(
        n_samples
    )

    trend = cp.Variable(
        n_samples
    )

    objective = cp.Minimize(
        0.5
        * cp.sum_squares(
            signal - trend
        )
        + vlambda
        * cp.norm(
            difference_matrix @ trend,
            1,
        )
    )

    problem = cp.Problem(
        objective
    )

    # --------------------------------------------------------
    # Solve using CLARABEL
    # --------------------------------------------------------

    with warnings.catch_warnings(
        record=True
    ) as caught:

        warnings.simplefilter(
            "always"
        )

        problem.solve(
            solver=cp.CLARABEL,
            verbose=verbose,
            max_iter=200,
        )

        solver_warnings = [
            str(warning.message)
            for warning in caught
        ]

    status = problem.status

    # --------------------------------------------------------
    # Accept full or reduced-accuracy optimal solutions
    # --------------------------------------------------------

    accepted_statuses = {
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    }

    if status not in accepted_statuses:

        warning_text = (
            " | ".join(
                solver_warnings
            )
            if solver_warnings
            else "None"
        )

        raise RuntimeError(
            "L1 trend filtering did not obtain an acceptable solution. "
            f"Solver: {SOLVER}; "
            f"status: {status}; "
            f"warnings: {warning_text}"
        )

    # --------------------------------------------------------
    # Validate returned trend
    # --------------------------------------------------------

    if trend.value is None:
        raise RuntimeError(
            "CLARABEL returned no trend values."
        )

    estimated_trend = np.asarray(
        trend.value,
        dtype=float,
    ).reshape(-1)

    if (
        estimated_trend.shape[0]
        != n_samples
    ):
        raise RuntimeError(
            "Unexpected trend length returned by CLARABEL."
        )

    if not np.isfinite(
        estimated_trend
    ).all():
        raise RuntimeError(
            "CLARABEL generated non-finite trend values."
        )

    # --------------------------------------------------------
    # Solver metadata
    # --------------------------------------------------------

    info = {
        "solver": SOLVER,
        "status": status,
        "warnings": solver_warnings,
        "lambda": vlambda,
        "samples": n_samples,
        "objective_value": problem.value,
        "solve_time": (
            problem.solver_stats.solve_time
            if problem.solver_stats is not None
            else None
        ),
        "iterations": (
            problem.solver_stats.num_iters
            if problem.solver_stats is not None
            else None
        ),
    }

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    if return_info:
        return estimated_trend, info

    return estimated_trend