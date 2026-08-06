"""The iterative solvers must not warn on their own first iteration.

Both refiners seed `prev_residual_norm` with an infinity sentinel. Until
2026-07-31 they divided by it on iteration 0, producing inf/inf = nan and
a RuntimeWarning on the first line of output — including from the README
quick start, which is the first code a new reader runs.

The nan never reached control flow (`nan < tol` is False, and an
`iteration > 0` guard suppressed the break independently), so removing it
changed no number. That is exactly what makes the warning easy to
reintroduce: nothing else fails when it comes back. These tests fail
instead.

The scalar and vectorized paths are separate implementations and are
tested separately; neither covers the other.
"""

import warnings

import numpy as np
import pytest

from toyomacro.voigtfit import voigt_profile
from toyomacro.voigtfit.gauss_newton import GaussNewtonRefiner
from toyomacro.voigtfit.stage2 import Stage2Config, Stage2Refiner

ENERGY = np.linspace(97, 107, 128)
CENTERS = np.array([99.3, 103.4])
SIGMAS = np.array([0.45, 0.60])
GAMMAS = np.array([0.10, 0.10])
# Offset the starting centers so the solver has real work to do: a fit
# that converges immediately would not exercise later iterations.
START_OFFSET = 0.2


def _basis():
    return np.stack(
        [voigt_profile(ENERGY, c, s, g) for c, s, g in zip(CENTERS, SIGMAS, GAMMAS)]
    )


def _numeric_warnings(record):
    """Warnings from the numeric stack, which is what a nan would raise."""
    return sorted(
        {str(w.message) for w in record if issubclass(w.category, RuntimeWarning)}
    )


# `max_iter=1` stops at iteration 0, the pass that used to warn, and is
# also the case where `rel_change` is read after the loop without ever
# having been assigned — so it guards the fix's one fragile edge.
ITERATION_LIMITS = (1, 2, 15)


@pytest.mark.parametrize("max_iter", ITERATION_LIMITS)
def test_scalar_refiner_is_silent(max_iter):
    rng = np.random.default_rng(7)
    y = _basis().T @ np.array([1.0, 0.6]) + rng.normal(0, 0.003, size=ENERGY.size)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = GaussNewtonRefiner(max_iter=max_iter, tol=1e-6).refine(
            ENERGY, y, CENTERS + START_OFFSET, SIGMAS, GAMMAS
        )

    assert _numeric_warnings(record) == []
    # The result is still well formed — silence achieved by returning
    # early or by swallowing the warning would not count.
    assert np.isfinite(result.residual_norm)
    assert np.all(np.isfinite(result.centers))
    assert result.n_iter == max_iter or result.n_iter < max_iter


@pytest.mark.parametrize("max_iter", ITERATION_LIMITS)
def test_vectorized_refiner_is_silent(max_iter):
    rng = np.random.default_rng(7)
    n = 32
    amps = rng.uniform(0.4, 1.2, size=(2, n))
    Y = _basis().T @ amps + rng.normal(0, 0.003, size=(ENERGY.size, n))

    refiner = Stage2Refiner(Stage2Config(max_iter=max_iter, tol=1e-5))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = refiner.refine_batch_vectorized(
            Y, ENERGY, CENTERS + START_OFFSET, SIGMAS, GAMMAS, np.arange(n)
        )

    assert _numeric_warnings(record) == []
    assert np.all(np.isfinite(np.asarray(result.centers)))


def test_the_infinity_sentinel_is_still_what_makes_this_necessary():
    """Pin the premise, so the guard is removed if the seed ever changes.

    If someone replaces the sentinel with a finite value the tests above
    would pass for a different reason — and a finite seed can spuriously
    satisfy the tolerance on the first pass, which is why it was rejected
    as the fix. Reading the source keeps that trade-off visible.
    """
    import inspect

    for module_fn, name in (
        (GaussNewtonRefiner.refine, "gauss_newton"),
        (Stage2Refiner.refine_batch_vectorized, "stage2"),
    ):
        source = inspect.getsource(module_fn)
        assert "np.inf" in source, f"{name}: no infinity sentinel found"
        assert "if iteration > 0:" in source, (
            f"{name}: the first-iteration guard is gone while the "
            "infinity sentinel remains"
        )
