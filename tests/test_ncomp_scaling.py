"""Smoke tests for n_comp scaling benchmark.

Verifies the benchmark runners execute without error at small scale.
No throughput assertions — purely functional validation.
"""

import importlib.util

import numpy as np
import pytest

HAS_MLX = importlib.util.find_spec("mlx") is not None
from toyomacro.voigtfit.benchmarks.bench_ncomp_scaling import (
    _make_energy_axis,
    _peak_config,
    _singlet_profile,
    bench_4step,
    bench_amp_only,
    bench_dict2d,
)
from toyomacro.voigtfit.synthetic import make_synthetic_configs, make_synthetic_spectra

N_SPECTRA = 1_000
SNR_DB = 40.0


def _prepare(n_comp: int):
    """Generate synthetic data for a given n_comp."""
    profile = _singlet_profile()
    configs, _ = make_synthetic_configs(profile, n_comp=n_comp)
    energy = _make_energy_axis(configs, profile.sigma)
    spectra, _ = make_synthetic_spectra(
        configs, profile, n_spectra=N_SPECTRA,
        E_range=(energy[0], energy[-1]),
        n_channels=len(energy),
        snr_db=SNR_DB,
    )
    return spectra, energy, configs


class TestNcompScalingSmoke:
    """Smoke tests: each solver × n_comp=1,2,3."""

    @pytest.mark.parametrize("n_comp", [1, 2, 3])
    def test_amp_only(self, n_comp: int):
        spectra, energy, configs = _prepare(n_comp)
        r = bench_amp_only(spectra, energy, _peak_config(configs), n_warmup=100)
        assert r.solver == "amp_only"
        assert r.n_comp == n_comp
        assert r.rate > 0
        assert r.elapsed > 0

    @pytest.mark.parametrize("n_comp", [1, 2, 3])
    def test_4step(self, n_comp: int):
        spectra, energy, configs = _prepare(n_comp)
        r = bench_4step(spectra, energy, _peak_config(configs), n_warmup=100)
        assert r.solver == "4-step"
        assert r.n_comp == n_comp
        assert r.rate > 0

    @pytest.mark.parametrize("n_comp", [1, 2, 3])
    def test_dict2d(self, n_comp: int):
        spectra, energy, configs = _prepare(n_comp)
        r = bench_dict2d(spectra, energy, configs, n_warmup=100)
        assert r.solver == "dict2d"
        assert r.n_comp == n_comp
        assert r.rate > 0
