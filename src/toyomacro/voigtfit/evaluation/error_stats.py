"""Error statistics for parameter estimation evaluation.

Computes bias, variance, RMSE, and other diagnostics from
estimated vs true parameter arrays.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ErrorStats:
    """Error statistics for a single parameter across all pixels.

    Attributes:
        bias: mean(est - true) — systematic error
        std: std(est - true) — random error (variance^0.5)
        rmse: sqrt(mean((est - true)^2)) — total error
        mae: median(|est - true|) — robust central error
        p95: 95th percentile of |est - true| — tail error
        errors: (n_pixels,) raw signed errors for histograms
    """
    bias: float
    std: float
    rmse: float
    mae: float
    p95: float
    errors: np.ndarray  # (n_pixels,) signed errors


def compute_error_stats(
    estimated: np.ndarray,
    true_values: np.ndarray,
    component: int | None = None,
) -> ErrorStats:
    """Compute error statistics between estimated and true parameter values.

    Args:
        estimated: (n_pixels,) or (n_pixels, n_comp) estimated values
        true_values: same shape as estimated
        component: If multi-component, select this component index.
            None means flatten all components together.

    Returns:
        ErrorStats with bias, std, rmse, mae, p95, and raw errors
    """
    est = np.asarray(estimated, dtype=np.float64).ravel() if component is None else \
        np.asarray(estimated, dtype=np.float64)
    true = np.asarray(true_values, dtype=np.float64).ravel() if component is None else \
        np.asarray(true_values, dtype=np.float64)

    if component is not None:
        if est.ndim == 2:
            est = est[:, component]
        if true.ndim == 2:
            true = true[:, component]
        est = est.ravel()
        true = true.ravel()

    errors = est - true
    abs_errors = np.abs(errors)

    bias = float(np.mean(errors))
    std = float(np.std(errors, ddof=0))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.median(abs_errors))
    p95 = float(np.percentile(abs_errors, 95)) if len(abs_errors) > 0 else 0.0

    return ErrorStats(
        bias=bias,
        std=std,
        rmse=rmse,
        mae=mae,
        p95=p95,
        errors=errors.astype(np.float32),
    )
