"""
Tests for Memory Management Utilities
===================================================

Covers:
1. Memory detection (available/total)
2. Optimal chunk size calculation
3. Dict3D LRU cache auto-sizing
4. MemoryProfiler context manager
5. Chunked parabola solver (low-memory path)
6. fp16 scores option
7. FastFitConfig memory modes
8. Memory mode auto-selection
"""


import numpy as np
import pytest

from toyomacro.voigtfit.memory import (
    MemoryProfiler,
    get_available_memory_gb,
    get_total_memory_gb,
    optimal_chunk_size,
    optimal_dict3d_cache_size,
    select_memory_mode,
)

# ============================================================================
# Part 1: Memory Detection
# ============================================================================

class TestMemoryDetection:
    """Test memory detection utilities."""

    def test_available_memory_positive(self):
        """Available memory should be a positive number."""
        avail = get_available_memory_gb()
        assert avail > 0.0
        assert isinstance(avail, float)

    def test_total_memory_positive(self):
        """Total memory should be positive and >= available."""
        total = get_total_memory_gb()
        assert total > 0.0
        assert total >= get_available_memory_gb()


class TestOptimalChunkSize:
    """Test chunk size auto-determination."""

    def test_128gb_environment(self):
        """128 GB available → 10M (max_chunk cap)."""
        chunk = optimal_chunk_size(n_dict=310, available_gb=128.0)
        assert chunk == 10_000_000

    def test_16gb_environment(self):
        """16 GB available → constrained chunk size."""
        chunk = optimal_chunk_size(n_dict=310, available_gb=16.0)
        # 16 * 1e9 * 0.3 / (310 * 4) ≈ 3.87M
        assert 100_000 <= chunk <= 10_000_000
        assert chunk < 5_000_000  # should be well below max

    def test_4gb_environment(self):
        """4 GB available → small but viable chunk."""
        chunk = optimal_chunk_size(n_dict=310, available_gb=4.0)
        # 4 * 1e9 * 0.3 / (310 * 4) ≈ 967K
        assert chunk >= 100_000
        assert chunk < 2_000_000

    def test_min_chunk_enforced(self):
        """Very low memory → clipped to min_chunk."""
        chunk = optimal_chunk_size(n_dict=310, available_gb=0.01)
        assert chunk == 100_000

    def test_fp16_doubles_chunk(self):
        """fp16 halves dtype_bytes → roughly doubles chunk size."""
        chunk_fp32 = optimal_chunk_size(n_dict=310, available_gb=16.0, dtype_bytes=4)
        chunk_fp16 = optimal_chunk_size(n_dict=310, available_gb=16.0, dtype_bytes=2)
        # fp16 chunk should be roughly 2x
        assert chunk_fp16 >= chunk_fp32 * 1.5

    def test_dict3d_larger_dict(self):
        """Dict3D has ~2079 entries → smaller chunk than Dict2D."""
        chunk_2d = optimal_chunk_size(n_dict=310, available_gb=16.0)
        chunk_3d = optimal_chunk_size(n_dict=2079, available_gb=16.0)
        assert chunk_3d < chunk_2d

    def test_custom_fraction(self):
        """Custom memory_fraction should scale linearly."""
        chunk_30 = optimal_chunk_size(n_dict=310, available_gb=16.0, memory_fraction=0.3)
        chunk_10 = optimal_chunk_size(n_dict=310, available_gb=16.0, memory_fraction=0.1)
        # Should be roughly 3:1 ratio (within clipping)
        assert chunk_30 >= chunk_10


class TestDict3DCacheSize:
    """Test Dict3D LRU cache auto-sizing."""

    def test_16gb(self):
        """16 GB → limited cache (formula: 16*0.2/0.75 = 4)."""
        size = optimal_dict3d_cache_size(available_gb=16.0)
        assert size >= 1
        assert size <= 5

    def test_8gb_available(self):
        """8 GB available (typical 16GB machine) → max_size=2."""
        size = optimal_dict3d_cache_size(available_gb=8.0)
        assert size >= 1
        assert size <= 3

    def test_32gb(self):
        """32 GB → moderate cache."""
        size = optimal_dict3d_cache_size(available_gb=32.0)
        assert size >= 2

    def test_128gb(self):
        """128 GB → large cache."""
        size = optimal_dict3d_cache_size(available_gb=128.0)
        assert size >= 10

    def test_minimum_is_one(self):
        """Even tiny memory → at least 1 cache entry."""
        size = optimal_dict3d_cache_size(available_gb=0.1)
        assert size == 1


# ============================================================================
# Part 5: Memory Profiler
# ============================================================================

class TestMemoryProfiler:
    """Test MemoryProfiler context manager."""

    def test_basic_usage(self):
        """Profiler reports positive baseline and delta."""
        with MemoryProfiler() as prof:
            _ = np.zeros(1_000_000, dtype=np.float32)  # 4 MB
        report = prof.report()
        assert report.baseline_gb > 0
        assert report.peak_gb >= report.baseline_gb
        assert report.delta_gb >= 0

    def test_snapshot(self):
        """Snapshots are recorded with labels."""
        with MemoryProfiler() as prof:
            _ = np.zeros(1_000_000, dtype=np.float32)
            prof.snapshot("after allocation")
            _ = np.zeros(2_000_000, dtype=np.float32)
            prof.snapshot("after second")
        report = prof.report()
        assert len(report.snapshots) == 2
        assert report.snapshots[0].label == "after allocation"
        assert report.snapshots[1].label == "after second"

    def test_report_str(self):
        """String representation should include key metrics."""
        with MemoryProfiler() as prof:
            pass
        text = str(prof.report())
        assert "Peak:" in text
        assert "Baseline:" in text
        assert "Delta:" in text


class TestSelectMemoryMode:
    """Test auto memory mode selection."""

    def test_high_memory(self):
        assert select_memory_mode(available_gb=64.0) == "high"

    def test_low_memory(self):
        assert select_memory_mode(available_gb=16.0) == "low"

    def test_boundary(self):
        assert select_memory_mode(available_gb=32.0) == "high"
        assert select_memory_mode(available_gb=31.9) == "low"


# ============================================================================
# Part 2 + 4: Chunked Parabola Solver + fp16
# ============================================================================

try:
    import mlx.core as mx

    from toyomacro.voigtfit._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND a Metal device works
except ImportError:
    HAS_MLX = False

needs_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def _make_test_spectra(n_spectra=500, n_energy=200, n_comp=3, seed=42):
    """Create synthetic Voigt spectra for memory tests."""
    from toyomacro.voigtfit.voigt_jacobian import voigt_profile

    rng = np.random.default_rng(seed)
    energy = np.linspace(95.0, 105.0, n_energy, dtype=np.float32)
    centers = np.array([98.0, 100.0, 102.0], dtype=np.float32)[:n_comp]
    sigmas = np.array([0.4, 0.5, 0.4], dtype=np.float32)[:n_comp]
    gamma = 0.15

    spectra = np.zeros((n_spectra, n_energy), dtype=np.float32)
    for i in range(n_spectra):
        for j in range(n_comp):
            amp = rng.uniform(50, 200)
            dE = rng.uniform(-0.3, 0.3)
            profile = voigt_profile(
                energy, centers[j] + dE, sigmas[j], gamma
            )
            spectra[i] += amp * profile
        spectra[i] += rng.normal(0, 0.5, n_energy).astype(np.float32)

    return spectra, energy, centers, sigmas, gamma


@needs_mlx
class TestChunkedParabola:
    """Test that chunked solver produces same results as full solver."""

    def test_chunked_matches_full(self):
        """Chunked parabola should match full solver within tolerance."""
        from toyomacro.voigtfit.dictionary_solver import (
            _PHASE1_MAX_BYTES,
            build_dictionary_2d,
            solve_dict2d_parabola,
        )

        spectra, energy, centers, sigmas, gamma = _make_test_spectra(n_spectra=1000)

        dict_cache = build_dictionary_2d(
            energy=energy, centers=centers, sigmas=sigmas, gamma=gamma,
        )

        # Full solver (large chunk_bytes → no chunking)
        amp_full, chi2_full, dE_full, ds_full, idx_full = solve_dict2d_parabola(
            spectra, dict_cache, chunk_bytes=_PHASE1_MAX_BYTES,
        )

        # Chunked solver (tiny chunk_bytes → force chunking)
        amp_chunk, chi2_chunk, dE_chunk, ds_chunk, idx_chunk = solve_dict2d_parabola(
            spectra, dict_cache, chunk_bytes=1024,  # force tiny chunks
        )

        np.testing.assert_array_equal(idx_full, idx_chunk)
        # GPU chunking introduces minor numerical differences from
        # different matmul partitioning; tolerance ~2e-5 is expected.
        np.testing.assert_allclose(dE_full, dE_chunk, atol=5e-5)
        np.testing.assert_allclose(ds_full, ds_chunk, atol=5e-5)
        np.testing.assert_allclose(amp_full, amp_chunk, atol=1e-3)
        np.testing.assert_allclose(chi2_full, chi2_chunk, atol=1e-4)

    def test_fp16_scores_close_to_fp32(self):
        """fp16 scores should produce nearly identical results."""
        from toyomacro.voigtfit.dictionary_solver import (
            build_dictionary_2d,
            solve_dict2d_parabola,
        )

        spectra, energy, centers, sigmas, gamma = _make_test_spectra(n_spectra=500)

        dict_cache = build_dictionary_2d(
            energy=energy, centers=centers, sigmas=sigmas, gamma=gamma,
        )

        amp_fp32, chi2_fp32, dE_fp32, ds_fp32, idx_fp32 = solve_dict2d_parabola(
            spectra, dict_cache, use_fp16_scores=False,
        )

        amp_fp16, chi2_fp16, dE_fp16, ds_fp16, idx_fp16 = solve_dict2d_parabola(
            spectra, dict_cache, use_fp16_scores=True,
        )

        # fp16 has only ~3.3 decimal digits of precision, so score ranking
        # can differ when nearby dict entries have similar scores.
        # Typical agreement is 60-80% depending on spectral SNR.
        agree_rate = np.mean(idx_fp32 == idx_fp16)
        assert agree_rate > 0.50, f"Index agreement {agree_rate:.3f} too low"

        # Where indices match, parameter shifts should be close
        match = idx_fp32 == idx_fp16
        if np.sum(match) > 10:
            np.testing.assert_allclose(
                dE_fp32[match], dE_fp16[match], atol=0.05,
            )
            np.testing.assert_allclose(
                ds_fp32[match], ds_fp16[match], atol=0.05,
            )

    def test_chunked_with_fp16(self):
        """Combined chunking + fp16 should work."""
        from toyomacro.voigtfit.dictionary_solver import (
            build_dictionary_2d,
            solve_dict2d_parabola,
        )

        spectra, energy, centers, sigmas, gamma = _make_test_spectra(n_spectra=500)

        dict_cache = build_dictionary_2d(
            energy=energy, centers=centers, sigmas=sigmas, gamma=gamma,
        )

        # Should not raise
        amp, chi2, dE, ds, idx = solve_dict2d_parabola(
            spectra, dict_cache,
            chunk_bytes=2048,  # tiny chunks
            use_fp16_scores=True,
        )
        assert amp.shape[1] == 500
        assert chi2.shape[0] == 500


# ============================================================================
# Part 3: Dict3D LRU Auto-sizing
# ============================================================================

class TestDict3DLRUAutoSize:
    """Test Dict3D cache auto-sizes based on memory."""

    def test_auto_size_creates_valid_cache(self):
        """Cache with auto-size should have max_size >= 1."""
        from toyomacro.voigtfit.dictionary_solver_3d import _Dict3DLRUCache
        cache = _Dict3DLRUCache()
        assert cache._max_size >= 1

    def test_explicit_size_preserved(self):
        """Explicit max_size should override auto-detection."""
        from toyomacro.voigtfit.dictionary_solver_3d import _Dict3DLRUCache
        cache = _Dict3DLRUCache(max_size=7)
        assert cache._max_size == 7

    def test_module_singleton_auto_sized(self):
        """Module-level _dict3d_cache should have auto-determined size."""
        from toyomacro.voigtfit.dictionary_solver_3d import _dict3d_cache
        assert _dict3d_cache._max_size >= 1


# ============================================================================
# Part 6: FastFitConfig memory modes
# ============================================================================

class TestFastFitConfigMemory:
    """Test FastFitConfig memory mode parameters."""

    def test_default_memory_mode_is_auto(self):
        from toyomacro.fitting.fast_voigt import FastFitConfig
        config = FastFitConfig()
        assert config.memory_mode == "auto"
        assert config.chunk_bytes is None
        assert config.use_fp16_scores is False

    def test_low_memory_mode(self):
        from toyomacro.fitting.fast_voigt import FastFitConfig
        config = FastFitConfig(memory_mode="low")
        assert config.memory_mode == "low"

    def test_explicit_chunk_bytes(self):
        from toyomacro.fitting.fast_voigt import FastFitConfig
        config = FastFitConfig(chunk_bytes=512 * 1024**2)
        assert config.chunk_bytes == 512 * 1024**2

    def test_fp16_config(self):
        from toyomacro.fitting.fast_voigt import FastFitConfig
        config = FastFitConfig(use_fp16_scores=True)
        assert config.use_fp16_scores is True


class TestFastVoigtFitterMemory:
    """Test FastVoigtFitter memory mode integration."""

    def test_effective_chunk_bytes_auto_returns_none_or_int(self):
        """Auto mode should return None (high memory) or int (low memory)."""
        from toyomacro.fitting.fast_voigt import VOIGTFIT_AVAILABLE, FastFitConfig, FastVoigtFitter
        if not VOIGTFIT_AVAILABLE:
            pytest.skip("voigtfit not available")
        fitter = FastVoigtFitter(FastFitConfig(memory_mode="auto"))
        result = fitter._effective_chunk_bytes()
        assert result is None or isinstance(result, int)

    def test_effective_chunk_bytes_high(self):
        from toyomacro.fitting.fast_voigt import VOIGTFIT_AVAILABLE, FastFitConfig, FastVoigtFitter
        if not VOIGTFIT_AVAILABLE:
            pytest.skip("voigtfit not available")
        fitter = FastVoigtFitter(FastFitConfig(memory_mode="high"))
        assert fitter._effective_chunk_bytes() is None

    def test_effective_chunk_bytes_low(self):
        from toyomacro.fitting.fast_voigt import VOIGTFIT_AVAILABLE, FastFitConfig, FastVoigtFitter
        if not VOIGTFIT_AVAILABLE:
            pytest.skip("voigtfit not available")
        fitter = FastVoigtFitter(FastFitConfig(memory_mode="low"))
        result = fitter._effective_chunk_bytes()
        assert result == 256 * 1024**2

    def test_explicit_chunk_bytes_overrides_mode(self):
        from toyomacro.fitting.fast_voigt import VOIGTFIT_AVAILABLE, FastFitConfig, FastVoigtFitter
        if not VOIGTFIT_AVAILABLE:
            pytest.skip("voigtfit not available")
        fitter = FastVoigtFitter(FastFitConfig(chunk_bytes=42))
        assert fitter._effective_chunk_bytes() == 42
