"""
Monte Carlo check of the identifiability diagnostics
====================================================

Simulated Poisson spectra, fitted by exact constrained maximum
likelihood (``_poisson_mle``), against the inverse Fisher matrix of
``toyomacro.voigtfit.identifiability``.

Two regimes, treated differently on purpose:

- *Interior.* Where the module itself says the Gaussian variance is
  identified and away from its boundary, the replica covariance must
  match the inverse Fisher matrix. Pass/fail.
- *Near the boundary.* There the inverse Fisher matrix is not the
  variance of the constrained estimator (Self & Liang 1987) and no
  agreement is expected. The distribution of the estimate -- including
  the share of replicas held at the floor -- is *recorded*, printed and
  attached to the test report, and nothing about the disagreement is
  asserted. Run with ``-rP`` to read it.

Everything here is a simulation from the model; none of it is a
measurement. Seeds are fixed.
"""

import numpy as np
import pytest

from tests._poisson_mle import I_VAR, deviance, fit_in_chunks, fit_reference, model
from toyomacro.voigtfit.identifiability import (
    VoigtPeak,
    assess_identifiability,
    constant_background,
    poisson_fisher,
)

GAMMA = 0.3
AREA = 2.0e4
LEVEL = 300.0
ENERGY = np.arange(-3.0, 3.0 + 1e-9, 0.02)


def _simulate(sigma, n_replicas, seed, variance_floor=0.0):
    """Counts for one peak on an estimated flat background, and their fit."""
    peak = VoigtPeak(AREA, 0.0, sigma, GAMMA)
    background = constant_background(ENERGY, LEVEL)
    fisher = poisson_fisher(ENERGY, [peak], background)
    counts = np.random.default_rng(seed).poisson(
        fisher.expected_counts, size=(n_replicas, ENERGY.size)
    ).astype(np.float64)
    fit = fit_in_chunks(
        counts, ENERGY, fisher.param_values, background.basis, variance_floor=variance_floor
    )
    report = assess_identifiability(
        ENERGY, [peak], background, variance_floor=variance_floor
    )
    return fisher, counts, fit, report


class TestFitter:
    """The helper has to be right before anything is compared with it."""

    @pytest.mark.parametrize("sigma", [GAMMA, 0.0])
    def test_reaches_the_constrained_maximum_of_the_exact_likelihood(self, sigma):
        """Against L-BFGS-B on the same deviance: a different algorithm,
        the same optimum, in the interior and on the boundary."""
        fisher, counts, fit, _ = _simulate(sigma, 12, seed=11)
        sd = np.sqrt(np.diag(np.linalg.inv(fisher.fisher)))
        assert fit.converged.all()
        if sigma == 0.0:
            assert 0 < fit.at_floor.sum() < 12  # both kinds of replica are exercised
        for k in range(12):
            reference, value = fit_reference(counts[k], ENERGY, fisher.param_values,
                                             np.ones((1, ENERGY.size)))
            assert np.all(np.abs(reference - fit.params[k]) < 1e-4 * sd)
            assert fit.deviance[k] <= value + 1e-7
            assert (fit.params[k, I_VAR] == 0.0) == bool(fit.at_floor[k])

    def test_held_replicas_satisfy_the_kkt_condition(self):
        """At the floor the score must point out of the feasible set."""
        fisher, counts, fit, _ = _simulate(0.0, 400, seed=12)
        held = fit.at_floor
        assert 0.3 < held.mean() < 0.7
        mean, jac = model(ENERGY, fit.params, np.ones((1, ENERGY.size)))
        score = np.einsum("nep,ne->np", jac, counts / mean - 1.0)
        scale = np.sqrt(np.einsum("nep,nep->np", jac / mean[:, :, None], jac))
        assert np.all(score[held, I_VAR] <= 0.0)
        free = np.ones_like(score, dtype=bool)
        free[held, I_VAR] = False
        assert np.max(np.abs(score / scale)[free]) < 1e-6
        assert np.all(np.isfinite(deviance(counts, mean)))


class TestInterior:
    def test_replica_covariance_is_the_inverse_fisher_matrix(self):
        """sigma/gamma = 1, 10^4 replicas, tolerance 5 %.

        The sampling error of a standard deviation from 10^4 replicas is
        1/sqrt(2e4) = 0.7 %, so 5 % is seven of those; a first-order bias
        of the estimate would show as a shift, bounded here at 0.05 sd
        (its own sampling error is 0.01 sd).
        """
        fisher, _, fit, report = _simulate(GAMMA, 10_000, seed=1)
        assert fit.converged.all() and not fit.at_floor.any()
        # the regime is interior by the module's own account, not by assumption
        assert report.widths[0].status == "identified" and not report.widths[0].near_boundary

        bound = np.linalg.inv(fisher.fisher)
        sd = np.sqrt(np.diag(bound))
        empirical = np.cov(fit.params.T)
        empirical_sd = np.sqrt(np.diag(empirical))
        assert np.all(np.abs(empirical_sd / sd - 1.0) < 0.05)
        assert np.max(np.abs(
            empirical / np.outer(empirical_sd, empirical_sd) - bound / np.outer(sd, sd)
        )) < 0.05
        assert np.all(np.abs(fit.params.mean(axis=0) - fisher.param_values) < 0.05 * sd)


class TestBoundary:
    def test_distribution_near_the_boundary_is_recorded(self, record_property):
        """True variance at 0, 1, 2 and 4 Fisher standard deviations from
        the floor. Recorded, not judged: only the bookkeeping is asserted.

        For orientation, not as a criterion: with one parameter on the
        boundary and the rest interior, the asymptotic law of the
        estimate is half a point mass at the floor and half a half-normal
        (Self & Liang 1987, case 5), whose standard deviation is
        sqrt(1/2 - 1/(2 pi)) = 0.584 of the Fisher one.
        """
        sd_at_zero = assess_identifiability(
            ENERGY, [VoigtPeak(AREA, 0.0, 0.0, GAMMA)], constant_background(ENERGY, LEVEL)
        ).widths[0].sd_variance
        lines = ["v_true/sd  near_boundary  share_at_floor  sd_emp/sd_fisher(v)  bias(v)/sd  "
                 "sd_emp/sd_fisher(gamma)"]
        for k, distance in enumerate((0.0, 1.0, 2.0, 4.0)):
            sigma = float(np.sqrt(distance * sd_at_zero))
            fisher, _, fit, report = _simulate(sigma, 4000, seed=20 + k)
            sd = np.sqrt(np.diag(np.linalg.inv(fisher.fisher)))
            estimates = fit.params
            row = (
                distance, bool(report.widths[0].near_boundary), float(fit.at_floor.mean()),
                float(estimates[:, I_VAR].std(ddof=1) / sd[I_VAR]),
                float((estimates[:, I_VAR].mean() - sigma**2) / sd[I_VAR]),
                float(estimates[:, 3].std(ddof=1) / sd[3]),
            )
            record_property(f"boundary_{distance:g}_sd", row)
            lines.append("{:9.1f}  {!s:13}  {:14.4f}  {:19.4f}  {:10.4f}  {:23.4f}".format(*row))

            assert fit.converged.all()
            assert np.all(estimates[:, I_VAR] >= 0.0)
            assert np.all((estimates[:, I_VAR] == 0.0) == fit.at_floor)
        print("\n".join(lines))

    def test_a_calibrated_floor_is_a_boundary_like_any_other(self, record_property):
        """The same record with the boundary at sigma_inst**2 instead of 0:
        sample broadening absent, instrument width 0.15 eV."""
        floor = 0.15**2
        _, _, fit, report = _simulate(0.15, 4000, seed=30, variance_floor=floor)
        share = float(fit.at_floor.mean())
        record_property("share_at_calibrated_floor", share)
        print(f"variance_floor = 0.15**2, true excess 0: share at floor = {share:.4f}")
        assert report.widths[0].near_boundary and report.widths[0].variance_excess == 0.0
        assert fit.converged.all()
        assert np.all(fit.params[:, I_VAR] >= floor)
        assert np.all((fit.params[:, I_VAR] == floor) == fit.at_floor)
