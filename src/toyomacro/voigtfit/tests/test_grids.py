"""
Tests for non-uniform grid generators and parabola interpolation.

dev-log 56 (Phase 3 Part 1).
"""

import numpy as np

from toyomacro.voigtfit.dictionary_solver import (
    _parabola_vertex,
    _parabola_vertex_nonuniform,
)
from toyomacro.voigtfit.grids import (
    chebyshev_grid,
    is_uniform,
    precompute_grid_spacings,
    quantile_grid,
    sinh_grid,
    uniform_grid,
)

# ============================================================================
# Grid generators
# ============================================================================

class TestUniformGrid:
    def test_symmetry(self):
        g = uniform_grid(11, 2.0)
        assert len(g) == 11
        np.testing.assert_allclose(g[0], -2.0)
        np.testing.assert_allclose(g[-1], 2.0)
        np.testing.assert_allclose(g + g[::-1], 0.0, atol=1e-15)

    def test_center_point(self):
        g = uniform_grid(11, 1.0)
        np.testing.assert_allclose(g[5], 0.0, atol=1e-15)

    def test_is_uniform(self):
        assert is_uniform(uniform_grid(21, 2.0))


class TestSinhGrid:
    def test_symmetry(self):
        g = sinh_grid(29, 2.0, alpha=2.0)
        assert len(g) == 29
        np.testing.assert_allclose(g[0], -2.0, atol=1e-12)
        np.testing.assert_allclose(g[-1], 2.0, atol=1e-12)
        np.testing.assert_allclose(g + g[::-1], 0.0, atol=1e-12)

    def test_center_dense(self):
        g = sinh_grid(29, 2.0, alpha=2.0)
        diffs = np.diff(g)
        center_spacing = diffs[len(diffs) // 2]
        edge_spacing = diffs[0]
        assert center_spacing < edge_spacing, "Center should be denser than edges"

    def test_alpha_zero_is_uniform(self):
        g = sinh_grid(11, 1.0, alpha=0.0)
        assert is_uniform(g)

    def test_alpha_effect(self):
        """Larger alpha → more center concentration."""
        g1 = sinh_grid(21, 2.0, alpha=1.0)
        g2 = sinh_grid(21, 2.0, alpha=3.0)
        center1 = np.diff(g1)[10]
        center2 = np.diff(g2)[10]
        assert center2 < center1, "Higher alpha should compress center more"

    def test_is_not_uniform(self):
        g = sinh_grid(21, 2.0, alpha=2.0)
        assert not is_uniform(g)

    def test_sorted(self):
        g = sinh_grid(29, 2.0, alpha=2.5)
        assert np.all(np.diff(g) > 0), "Grid must be strictly ascending"


class TestChebyshevGrid:
    def test_symmetry(self):
        g = chebyshev_grid(15, 2.0)
        assert len(g) == 15
        np.testing.assert_allclose(g[0], -2.0, atol=1e-12)
        np.testing.assert_allclose(g[-1], 2.0, atol=1e-12)
        np.testing.assert_allclose(g + g[::-1], 0.0, atol=1e-12)

    def test_edge_dense(self):
        """Chebyshev points cluster at edges."""
        g = chebyshev_grid(21, 2.0)
        diffs = np.diff(g)
        edge_spacing = diffs[0]
        center_spacing = diffs[len(diffs) // 2]
        assert edge_spacing < center_spacing, "Edges should be denser"

    def test_sorted(self):
        g = chebyshev_grid(29, 2.0)
        assert np.all(np.diff(g) > 0)


class TestQuantileGrid:
    def test_normal_distribution(self):
        rng = np.random.default_rng(42)
        dist = rng.normal(0, 1, size=10000)
        g = quantile_grid(dist, 21, range_max=3.0)
        assert len(g) == 21
        assert g[0] >= -3.0
        assert g[-1] <= 3.0
        # Center should be dense (normal has most mass at center)
        diffs = np.diff(g)
        center_spacing = diffs[len(diffs) // 2]
        edge_spacing = diffs[0]
        assert center_spacing < edge_spacing

    def test_sorted(self):
        rng = np.random.default_rng(42)
        dist = rng.normal(0, 1, size=1000)
        g = quantile_grid(dist, 15)
        assert np.all(np.diff(g) >= 0)


# ============================================================================
# Grid spacing utilities
# ============================================================================

class TestGridSpacings:
    def test_uniform_spacings(self):
        g = uniform_grid(5, 2.0)  # [-2, -1, 0, 1, 2]
        h_L, h_R = precompute_grid_spacings(g)
        assert h_L[0] == 0.0
        assert h_R[-1] == 0.0
        np.testing.assert_allclose(h_L[1:], 1.0)
        np.testing.assert_allclose(h_R[:-1], 1.0)

    def test_nonuniform_spacings(self):
        g = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
        h_L, h_R = precompute_grid_spacings(g)
        np.testing.assert_allclose(h_L, [0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(h_R, [1.0, 2.0, 3.0, 4.0, 0.0])

    def test_sinh_grid_spacings(self):
        g = sinh_grid(11, 2.0, alpha=2.0)
        h_L, h_R = precompute_grid_spacings(g)
        # h_L[0] and h_R[-1] are sentinel zeros
        assert h_L[0] == 0.0
        assert h_R[-1] == 0.0
        # All interior spacings should be positive
        assert np.all(h_L[1:] > 0)
        assert np.all(h_R[:-1] > 0)
        # Center spacings should be smaller than edge spacings
        assert h_R[5] < h_R[0]


class TestIsUniform:
    def test_uniform(self):
        assert is_uniform(np.linspace(-1, 1, 21))

    def test_nonuniform(self):
        assert not is_uniform(np.array([0, 1, 3, 6]))

    def test_single_point(self):
        assert is_uniform(np.array([0.0]))

    def test_two_points(self):
        assert is_uniform(np.array([-1.0, 1.0]))


# ============================================================================
# Non-uniform parabola vertex
# ============================================================================

class TestParabolaVertexNonuniform:
    def test_reduces_to_uniform(self):
        """When h_L == h_R, should match standard parabola_vertex * h."""
        rng = np.random.default_rng(123)
        n = 100
        # Generate concave-down parabolas: s_0 is max
        s_0 = rng.uniform(5, 10, n)
        delta_true = rng.uniform(-0.4, 0.4, n)  # true offset in grid units
        # s(x) = -a*x^2 + b*x + c with vertex at delta_true
        a = rng.uniform(1, 5, n)
        s_m = s_0 - a * (delta_true + 1.0)**2 + a * delta_true**2
        s_p = s_0 - a * (delta_true - 1.0)**2 + a * delta_true**2
        # Fix: compute from exact parabola
        s_m = -a * 1.0 + a * 2.0 * delta_true + s_0  # s(-1)
        s_p = -a * 1.0 - a * 2.0 * delta_true + s_0  # s(+1)
        # Actually, let's use exact formulation:
        # s(x) = -a*(x - delta)^2 + s_0  (peak at delta with value s_0 + correction)
        # s(-1) = -a*(-1-delta)^2 + (s_0 + a*delta^2)
        # s(0) = s_0
        # s(+1) = -a*(1-delta)^2 + (s_0 + a*delta^2)
        c = s_0 + a * delta_true**2  # vertex value
        s_m = -a * (-1 - delta_true)**2 + c
        s_0_scores = -a * (0 - delta_true)**2 + c  # = s_0
        s_p = -a * (1 - delta_true)**2 + c

        h_L = np.ones(n)
        h_R = np.ones(n)

        shift_nu = _parabola_vertex_nonuniform(s_m, s_0_scores, s_p, h_L, h_R)
        delta_uniform = _parabola_vertex(s_m, s_0_scores, s_p)

        # Non-uniform with h=1 returns shift in real coords = delta_uniform * h = delta_uniform
        np.testing.assert_allclose(shift_nu, delta_uniform, atol=1e-6)

    def test_known_vertex(self):
        """Parabola with known vertex at x=0.3, sampled at (-0.5, 0, +0.8)."""
        x_v = 0.3
        a = -2.0  # concave down
        # s(x) = a*(x - x_v)^2 + 10
        h_L = np.array([0.5])
        h_R = np.array([0.8])
        s_m = a * (-h_L[0] - x_v)**2 + 10
        s_0 = a * (0 - x_v)**2 + 10
        s_p = a * (h_R[0] - x_v)**2 + 10

        shift = _parabola_vertex_nonuniform(s_m, s_0, s_p, h_L, h_R)
        np.testing.assert_allclose(shift, x_v, atol=1e-10)

    def test_known_vertex_negative(self):
        """Vertex at x=-0.2, sampled at (-1.0, 0, +0.5)."""
        x_v = -0.2
        a = -3.0
        h_L = np.array([1.0])
        h_R = np.array([0.5])
        s_m = a * (-h_L[0] - x_v)**2 + 5
        s_0 = a * (0 - x_v)**2 + 5
        s_p = a * (h_R[0] - x_v)**2 + 5

        shift = _parabola_vertex_nonuniform(s_m, s_0, s_p, h_L, h_R)
        np.testing.assert_allclose(shift, x_v, atol=1e-10)

    def test_clamping(self):
        """Vertex outside [-h_L, +h_R] should be clamped."""
        # Concave UP parabola → vertex is a minimum, argmax is at boundary
        h_L = np.array([0.5])
        h_R = np.array([0.8])
        s_m = np.array([10.0])
        s_0 = np.array([1.0])   # minimum
        s_p = np.array([8.0])

        shift = _parabola_vertex_nonuniform(s_m, s_0, s_p, h_L, h_R)
        # Should be clamped to [-0.5, 0.8]
        assert shift[0] >= -h_L[0] - 1e-10
        assert shift[0] <= h_R[0] + 1e-10

    def test_symmetric_nonuniform(self):
        """Symmetric scores → vertex at center regardless of spacing."""
        h_L = np.array([0.3])
        h_R = np.array([0.7])
        s_m = np.array([5.0])
        s_0 = np.array([10.0])
        s_p = np.array([5.0])

        shift = _parabola_vertex_nonuniform(s_m, s_0, s_p, h_L, h_R)
        # Symmetric scores but asymmetric spacing → vertex NOT at 0
        # s(x) passes through (-0.3, 5), (0, 10), (0.7, 5)
        # For a true parabola, the vertex is at the midpoint of the
        # two equal-score points: (0.7 - 0.3)/2 = 0.2
        np.testing.assert_allclose(shift, 0.2, atol=1e-10)

    def test_batch(self):
        """Vectorized over multiple spectra."""
        n = 1000
        rng = np.random.default_rng(99)
        x_v = rng.uniform(-0.3, 0.3, n)
        a = -rng.uniform(1, 5, n)
        h_L = rng.uniform(0.2, 1.0, n)
        h_R = rng.uniform(0.2, 1.0, n)
        # Ensure vertex is within [-h_L, h_R]
        x_v = np.clip(x_v, -h_L * 0.9, h_R * 0.9)

        c = 10.0
        s_m = a * (-h_L - x_v)**2 + c
        s_0 = a * (0 - x_v)**2 + c
        s_p = a * (h_R - x_v)**2 + c

        shift = _parabola_vertex_nonuniform(s_m, s_0, s_p, h_L, h_R)
        np.testing.assert_allclose(shift, x_v, atol=1e-6)
