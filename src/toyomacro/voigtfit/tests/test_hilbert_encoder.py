"""Tests for 3D Hilbert curve and MultiPeakEncoder.

Hilbert space-filling curve correctness and
multi-peak parameter encoding round-trip verification.
"""

import numpy as np
import pytest

from toyomacro.voigtfit.hilbert import (
    _scalar_d_to_xyz,
    _scalar_xyz_to_d,
    coords_to_index,
    index_to_coords,
)
from toyomacro.voigtfit.param_encoder import (
    C1S_MULTIPEAK_PRESET,
    MultiPeakEncoder,
    MultiPeakPreset,
    ParamRange,
    _dequantize,
    _quantize,
)
from toyomacro.voigtfit.spectra_generator import ElementSpec

# -----------------------------------------------------------------------
# Hilbert curve tests
# -----------------------------------------------------------------------

class TestHilbertScalar:
    """Scalar reference implementation correctness."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_roundtrip_exhaustive(self, order):
        """Every point in [0, 2^order)^3 round-trips correctly."""
        n = 1 << order
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    d = _scalar_xyz_to_d(x, y, z, order)
                    x2, y2, z2 = _scalar_d_to_xyz(d, order)
                    assert (x2, y2, z2) == (x, y, z), \
                        f"order={order}: ({x},{y},{z}) -> d={d} -> ({x2},{y2},{z2})"

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_bijection(self, order):
        """Forward mapping produces unique indices covering [0, 2^(3p))."""
        n = 1 << order
        indices = set()
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    indices.add(_scalar_xyz_to_d(x, y, z, order))
        expected = n ** 3
        assert len(indices) == expected
        assert min(indices) == 0
        assert max(indices) == expected - 1

    def test_origin(self):
        """Origin (0,0,0) always maps to index 0."""
        for order in range(1, 6):
            assert _scalar_xyz_to_d(0, 0, 0, order) == 0


class TestHilbertVectorized:
    """Vectorized numpy implementation correctness."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_exhaustive_roundtrip(self, order):
        """All points round-trip through vectorized encode/decode."""
        n = 1 << order
        total = n ** 3
        xs = np.arange(n).repeat(n * n).astype(np.int64)
        ys = np.tile(np.arange(n).repeat(n), n).astype(np.int64)
        zs = np.tile(np.arange(n), n * n).astype(np.int64)

        ds = coords_to_index(xs, ys, zs, order)
        x2, y2, z2 = index_to_coords(ds, order)

        assert np.all(x2 == xs)
        assert np.all(y2 == ys)
        assert np.all(z2 == zs)
        assert len(np.unique(ds)) == total

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_matches_scalar(self, order):
        """Vectorized matches scalar for all points."""
        n = 1 << order
        xs, ys, zs, ds_scalar = [], [], [], []
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    ds_scalar.append(_scalar_xyz_to_d(x, y, z, order))

        xs = np.array(xs, dtype=np.int64)
        ys = np.array(ys, dtype=np.int64)
        zs = np.array(zs, dtype=np.int64)
        ds_vec = coords_to_index(xs, ys, zs, order)
        np.testing.assert_array_equal(ds_vec, ds_scalar)

    def test_scalar_input(self):
        """Single scalar inputs work correctly."""
        d = coords_to_index(np.int64(0), np.int64(0), np.int64(0), 4)
        assert int(d) == 0
        x, y, z = index_to_coords(np.int64(0), 4)
        assert int(x) == 0 and int(y) == 0 and int(z) == 0

    def test_order5_random(self):
        """Order=5 (32^3 = 32768 points) random subset round-trips."""
        order = 5
        n = 1 << order
        rng = np.random.default_rng(123)
        xs = rng.integers(0, n, size=5000)
        ys = rng.integers(0, n, size=5000)
        zs = rng.integers(0, n, size=5000)
        ds = coords_to_index(xs, ys, zs, order)
        x2, y2, z2 = index_to_coords(ds, order)
        assert np.all((x2 == xs) & (y2 == ys) & (z2 == zs))


class TestHilbertLocality:
    """Spatial locality property of Hilbert curve."""

    def test_adjacent_points_nearby(self):
        """Adjacent 3D points should map to relatively nearby indices."""
        order = 4
        n = 1 << order
        rng = np.random.default_rng(42)
        gaps = []
        for _ in range(500):
            x, y, z = rng.integers(0, n - 1, size=3)
            d1 = int(coords_to_index(np.array([x]), np.array([y]), np.array([z]), order)[0])
            # Test all 6 face-adjacent neighbors
            for dx, dy, dz in [(1,0,0),(0,1,0),(0,0,1)]:
                d2 = int(coords_to_index(
                    np.array([x+dx]), np.array([y+dy]), np.array([z+dz]), order)[0])
                gaps.append(abs(d2 - d1))
        gaps = np.array(gaps)
        # Median gap should be much smaller than total range
        assert np.median(gaps) < n ** 3 * 0.05, \
            f"Median gap {np.median(gaps)} too large (range={n**3})"


# -----------------------------------------------------------------------
# Quantize / dequantize tests
# -----------------------------------------------------------------------

class TestQuantization:
    def test_roundtrip(self):
        """Quantize -> dequantize preserves values within step/2."""
        pr = ParamRange(-1.0, 1.0, 'test')
        n_levels = 16
        values = np.linspace(-1, 1, 100).astype(np.float32)
        coords = _quantize(values, n_levels, pr)
        recovered = _dequantize(coords, n_levels, pr)
        step = 2.0 / (n_levels - 1)
        assert np.max(np.abs(values - recovered)) <= step / 2 + 1e-6

    def test_boundary_clamping(self):
        """Values outside range are clamped to boundary."""
        pr = ParamRange(0.0, 1.0, 'test')
        vals = np.array([-0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
        coords = _quantize(vals, 16, pr)
        assert coords[0] == 0
        assert coords[-1] == 15

    def test_center_value(self):
        """Center of range maps to center coordinate."""
        pr = ParamRange(-1.0, 1.0, 'test')
        center = np.array([0.0], dtype=np.float32)
        # n_levels=17 -> center coord = 8
        c = _quantize(center, 17, pr)
        assert c[0] == 8


# -----------------------------------------------------------------------
# MultiPeakEncoder tests
# -----------------------------------------------------------------------

class TestMultiPeakEncoder:
    """Multi-peak Hilbert encoder round-trip tests."""

    def test_decode_encode_roundtrip(self):
        """Parameters -> RGB -> Parameters within quantization error."""
        enc = MultiPeakEncoder(C1S_MULTIPEAK_PRESET)
        rng = np.random.default_rng(42)
        N = 1000
        p = C1S_MULTIPEAK_PRESET

        amps = rng.uniform(p.amplitude_range.min_val,
                           p.amplitude_range.max_val, (N, 2)).astype(np.float32)
        dEs = rng.uniform(p.shift_range.min_val,
                          p.shift_range.max_val, (N, 2)).astype(np.float32)
        dSs = rng.uniform(p.sigma_range.min_val,
                          p.sigma_range.max_val, (N, 2)).astype(np.float32)

        rgb = enc.decode(amps, dEs, dSs, (N, 1))
        amps2, dEs2, dSs2 = enc.encode(rgb)

        da, de, ds = enc.quantization_step()
        np.testing.assert_allclose(amps, amps2, atol=da / 2 + 1e-6)
        np.testing.assert_allclose(dEs, dEs2, atol=de / 2 + 1e-6)
        np.testing.assert_allclose(dSs, dSs2, atol=ds / 2 + 1e-6)

    def test_encode_decode_roundtrip(self):
        """RGB -> Parameters -> RGB is lossless (same quantized image)."""
        enc = MultiPeakEncoder(C1S_MULTIPEAK_PRESET)
        rng = np.random.default_rng(99)
        H, W = 32, 32
        rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)

        amps, dEs, dSs = enc.encode(rgb)
        rgb2 = enc.decode(amps, dEs, dSs, (H, W))

        np.testing.assert_array_equal(rgb, rgb2)

    def test_quantization_step(self):
        """Quantization step sizes are correct."""
        enc = MultiPeakEncoder(C1S_MULTIPEAK_PRESET)
        da, de, ds = enc.quantization_step()
        # 16 levels, range 1.0 -> step = 1/15
        assert abs(da - 1.0 / 15) < 1e-10
        assert abs(de - 1.0 / 15) < 1e-10
        assert abs(ds - 0.2 / 15) < 1e-10

    def test_image_shape(self):
        """Output image has correct shape and dtype."""
        enc = MultiPeakEncoder(C1S_MULTIPEAK_PRESET)
        N = 480
        amps = np.ones((N, 2), dtype=np.float32) * 0.5
        dEs = np.zeros((N, 2), dtype=np.float32)
        dSs = np.zeros((N, 2), dtype=np.float32)
        rgb = enc.decode(amps, dEs, dSs, (20, 24))
        assert rgb.shape == (20, 24, 3)
        assert rgb.dtype == np.uint8

    def test_single_peak_order8(self):
        """Single-peak mode with order=8 (256 levels, full 24-bit)."""
        preset = MultiPeakPreset(
            name='test_1peak',
            elements=[ElementSpec('C', '1s', 284.4)],
            energy_range=(280.0, 290.0), n_energy=101,
            hilbert_order=8,
        )
        enc = MultiPeakEncoder(preset)
        da, de, ds = enc.quantization_step()
        # 256 levels, range 1.0 -> step = 1/255
        assert abs(da - 1.0 / 255) < 1e-10

        # Round-trip
        rng = np.random.default_rng(7)
        N = 500
        amps = rng.uniform(0, 1, (N, 1)).astype(np.float32)
        dEs = rng.uniform(-0.5, 0.5, (N, 1)).astype(np.float32)
        dSs = rng.uniform(-0.1, 0.1, (N, 1)).astype(np.float32)
        rgb = enc.decode(amps, dEs, dSs, (N, 1))
        a2, e2, s2 = enc.encode(rgb)
        np.testing.assert_allclose(amps, a2, atol=da / 2 + 1e-6)
        np.testing.assert_allclose(dEs, e2, atol=de / 2 + 1e-6)

    def test_too_many_bits_raises(self):
        """3 peaks x order 4 = 36 bits > 24 -> ValueError."""
        preset = MultiPeakPreset(
            name='test_3peak',
            elements=[ElementSpec('C', '1s', 284.4)] * 3,
            energy_range=(280.0, 290.0), n_energy=101,
            hilbert_order=4,
        )
        with pytest.raises(ValueError, match="exceeds 24-bit"):
            MultiPeakEncoder(preset)

    def test_zero_parameters(self):
        """All-zero parameters encode and decode correctly."""
        enc = MultiPeakEncoder(C1S_MULTIPEAK_PRESET)
        p = C1S_MULTIPEAK_PRESET
        N = 10
        # Zero is within range for shift and sigma
        amps = np.full((N, 2), 0.5, dtype=np.float32)
        dEs = np.zeros((N, 2), dtype=np.float32)
        dSs = np.zeros((N, 2), dtype=np.float32)
        rgb = enc.decode(amps, dEs, dSs, (N, 1))
        a2, e2, s2 = enc.encode(rgb)
        da, de, ds = enc.quantization_step()
        np.testing.assert_allclose(dEs, e2, atol=de / 2 + 1e-6)
        np.testing.assert_allclose(dSs, s2, atol=ds / 2 + 1e-6)

    def test_psnr_within_expected_range(self):
        """PSNR for 16-level quantization should be ~24-35 dB."""
        enc = MultiPeakEncoder(C1S_MULTIPEAK_PRESET)
        p = C1S_MULTIPEAK_PRESET
        rng = np.random.default_rng(42)
        N = 10000
        amps = rng.uniform(p.amplitude_range.min_val,
                           p.amplitude_range.max_val, (N, 2)).astype(np.float32)
        dEs = rng.uniform(p.shift_range.min_val,
                          p.shift_range.max_val, (N, 2)).astype(np.float32)
        dSs = rng.uniform(p.sigma_range.min_val,
                          p.sigma_range.max_val, (N, 2)).astype(np.float32)
        rgb = enc.decode(amps, dEs, dSs, (N, 1))
        a2, e2, s2 = enc.encode(rgb)
        mse_a = np.mean((amps - a2) ** 2)
        psnr_a = 10 * np.log10(np.max(amps) ** 2 / mse_a)
        # 16 levels -> ~24 dB theoretical minimum, but Hilbert preserves more
        assert 20 < psnr_a < 50, f"Unexpected PSNR: {psnr_a:.1f} dB"
