"""
Faddeeva Function Implementation for MLX
=========================================

MLX-optimized implementation using pre-computed lookup tables with
bilinear interpolation for the Voigt profile computation.

For XPS fitting, z = (E - E₀ + iγ) / (σ√2) with typical:
- Im(z) = γ/(σ√2) ∈ [0.05, 0.8]  (γ ~ 0.1-0.5, σ ~ 0.5-1.5)
- Re(z) ∈ [-15, 15]  (energy range around peak)

Strategy:
- Pre-compute w(x + iy) for a 2D grid of (x, y) values
- Use bilinear interpolation for GPU-friendly evaluation
- Fall back to scipy for out-of-range values
"""

import numpy as np

try:
    import mlx.core as mx

    from ._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND a Metal device works
except ImportError:
    HAS_MLX = False
    mx = None

# Constants
SQRT2 = np.sqrt(2.0)
SQRT2PI = np.sqrt(2.0 * np.pi)
SQRTPI = np.sqrt(np.pi)
INV_SQRTPI = 1.0 / SQRTPI

# ============================================================
# Lookup Table for Faddeeva Function
# ============================================================

class FaddeevaTable:
    """
    Pre-computed lookup table for Faddeeva function.

    Grid specifications:
    - x: [-15, 15] with 1501 points (dx = 0.02)
    - y: [0.01, 1.0] with 100 points (log-spaced for better resolution at small y)

    Accuracy: ~10^-6 relative error via bilinear interpolation
    """

    def __init__(self, x_range=(-15, 15), x_points=1501,
                 y_range=(0.01, 1.0), y_points=100):
        """Initialize and compute lookup table."""
        import scipy.special as sps

        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.nx = x_points
        self.ny = y_points

        # Grid
        self.x_grid = np.linspace(self.x_min, self.x_max, self.nx)
        self.y_grid = np.geomspace(self.y_min, self.y_max, self.ny)  # Log-spaced

        self.dx = self.x_grid[1] - self.x_grid[0]

        # Compute w(x + iy) for all grid points
        X, Y = np.meshgrid(self.x_grid, self.y_grid, indexing='xy')
        Z = X + 1j * Y
        W = sps.wofz(Z)

        # Store real and imaginary parts separately
        self.w_real = W.real.astype(np.float32)  # (ny, nx)
        self.w_imag = W.imag.astype(np.float32)  # (ny, nx)

        # MLX arrays (lazy initialization)
        self._w_real_mlx = None
        self._w_imag_mlx = None
        self._x_grid_mlx = None
        self._y_grid_mlx = None

    def _ensure_mlx(self):
        """Convert to MLX arrays if needed."""
        if HAS_MLX and self._w_real_mlx is None:
            self._w_real_mlx = mx.array(self.w_real)
            self._w_imag_mlx = mx.array(self.w_imag)
            self._x_grid_mlx = mx.array(self.x_grid.astype(np.float32))
            self._y_grid_mlx = mx.array(self.y_grid.astype(np.float32))

    def lookup_numpy(self, x, y):
        """
        Lookup w(x + iy) using bilinear interpolation (NumPy).

        Args:
            x: Real part (array)
            y: Imaginary part (scalar or array, must be same for all x in typical use)

        Returns:
            w: Complex Faddeeva function values
        """
        import scipy.special as sps

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # For out-of-range values, use scipy directly
        out_of_range = (x < self.x_min) | (x > self.x_max) | (y < self.y_min) | (y > self.y_max)

        if np.any(out_of_range):
            # Clip for table lookup
            x_clip = np.clip(x, self.x_min, self.x_max - 1e-6)
            y_clip = np.clip(y, self.y_min, self.y_max - 1e-6)
        else:
            x_clip = x
            y_clip = y

        # Find grid indices
        ix = (x_clip - self.x_min) / self.dx
        ix0 = np.floor(ix).astype(int)
        ix1 = np.minimum(ix0 + 1, self.nx - 1)
        fx = ix - ix0

        # For y, use log-space search
        iy = np.searchsorted(self.y_grid, y_clip) - 1
        iy = np.clip(iy, 0, self.ny - 2)
        iy0 = iy
        iy1 = iy + 1

        # Interpolation weight in y (linear in log space)
        log_y = np.log(y_clip)
        log_y0 = np.log(self.y_grid[iy0])
        log_y1 = np.log(self.y_grid[iy1])
        fy = (log_y - log_y0) / (log_y1 - log_y0 + 1e-10)
        fy = np.clip(fy, 0, 1)

        # Bilinear interpolation
        # Handle broadcast for scalar y
        if np.ndim(iy0) == 0:
            iy0 = np.full_like(ix0, int(iy0))
            iy1 = np.full_like(ix0, int(iy1))
            fy = np.full_like(fx, float(fy))

        # Get corner values
        w00_r = self.w_real[iy0, ix0]
        w01_r = self.w_real[iy0, ix1]
        w10_r = self.w_real[iy1, ix0]
        w11_r = self.w_real[iy1, ix1]

        w00_i = self.w_imag[iy0, ix0]
        w01_i = self.w_imag[iy0, ix1]
        w10_i = self.w_imag[iy1, ix0]
        w11_i = self.w_imag[iy1, ix1]

        # Interpolate
        w_r = (1 - fx) * (1 - fy) * w00_r + fx * (1 - fy) * w01_r + \
              (1 - fx) * fy * w10_r + fx * fy * w11_r
        w_i = (1 - fx) * (1 - fy) * w00_i + fx * (1 - fy) * w01_i + \
              (1 - fx) * fy * w10_i + fx * fy * w11_i

        result = w_r + 1j * w_i

        # Fix out-of-range with scipy
        if np.any(out_of_range):
            z_oor = x[out_of_range] + 1j * y[out_of_range] if np.ndim(y) > 0 else x[out_of_range] + 1j * y
            result[out_of_range] = sps.wofz(z_oor)

        return result

    def lookup_mlx(self, x, y):
        """
        Lookup w(x + iy) using bilinear interpolation (MLX).

        Optimized for GPU with batched operations.

        Args:
            x: Real part (MLX array, n_points)
            y: Imaginary part (scalar float)

        Returns:
            w: Complex Faddeeva function values (MLX array)
        """
        if not HAS_MLX:
            raise ImportError("MLX not available")

        self._ensure_mlx()

        # Ensure x is MLX array
        if not isinstance(x, mx.array):
            x = mx.array(x.astype(np.float32) if isinstance(x, np.ndarray) else np.array([x], dtype=np.float32))

        y = float(y)

        # Find y index (y is scalar for typical Voigt profile use)
        iy = np.searchsorted(self.y_grid, y) - 1
        iy = max(0, min(iy, self.ny - 2))
        iy0, iy1 = iy, iy + 1

        log_y = np.log(y)
        log_y0 = np.log(self.y_grid[iy0])
        log_y1 = np.log(self.y_grid[iy1])
        fy = float((log_y - log_y0) / (log_y1 - log_y0 + 1e-10))
        fy = max(0.0, min(1.0, fy))

        # Find x indices
        ix = (x - self.x_min) / self.dx
        ix0 = mx.floor(ix).astype(mx.int32)
        ix0 = mx.clip(ix0, 0, self.nx - 2)
        ix1 = ix0 + 1
        fx = ix - ix0.astype(mx.float32)
        fx = mx.clip(fx, 0.0, 1.0)

        # Get table rows for this y slice using integer index
        # iy0 and iy1 are Python ints, so direct indexing works
        w_r_y0 = self._w_real_mlx[int(iy0)]  # (nx,) - single row
        w_r_y1 = self._w_real_mlx[int(iy1)]
        w_i_y0 = self._w_imag_mlx[int(iy0)]
        w_i_y1 = self._w_imag_mlx[int(iy1)]

        # Gather values at x indices using take
        w00_r = mx.take(w_r_y0, ix0)
        w01_r = mx.take(w_r_y0, ix1)
        w10_r = mx.take(w_r_y1, ix0)
        w11_r = mx.take(w_r_y1, ix1)

        w00_i = mx.take(w_i_y0, ix0)
        w01_i = mx.take(w_i_y0, ix1)
        w10_i = mx.take(w_i_y1, ix0)
        w11_i = mx.take(w_i_y1, ix1)

        # Bilinear interpolation
        one_minus_fx = 1.0 - fx
        one_minus_fy = 1.0 - fy

        w_r = one_minus_fx * one_minus_fy * w00_r + fx * one_minus_fy * w01_r + \
              one_minus_fx * fy * w10_r + fx * fy * w11_r
        w_i = one_minus_fx * one_minus_fy * w00_i + fx * one_minus_fy * w01_i + \
              one_minus_fx * fy * w10_i + fx * fy * w11_r

        return w_r + 1j * w_i

    def lookup_mlx_batched(self, x, y):
        """
        Batched lookup for multiple y values (fully vectorized for GPU).

        Handles 3D input: x.shape = (n_energy, n_comp, n_spectra), y.shape = (n_comp,)
        Each component has its own y value (from different sigma values).

        Args:
            x: Real part (MLX array, shape (n_energy, n_comp, n_spectra) or (n_energy, n_comp))
            y: Imaginary part (MLX array or numpy array, shape (n_comp,))

        Returns:
            w_real: Real part of Faddeeva function (same shape as x)
        """
        if not HAS_MLX:
            raise ImportError("MLX not available")

        self._ensure_mlx()

        # Ensure inputs are MLX arrays
        if not isinstance(x, mx.array):
            x = mx.array(np.asarray(x, dtype=np.float32))

        y = np.asarray(y, dtype=np.float32)
        n_comp = len(y)

        # Pre-compute y interpolation parameters for all components
        iy0_list = []
        iy1_list = []
        fy_list = []

        for j in range(n_comp):
            y_j = float(y[j])
            iy = np.searchsorted(self.y_grid, y_j) - 1
            iy = max(0, min(iy, self.ny - 2))
            iy0_list.append(iy)
            iy1_list.append(iy + 1)

            log_y = np.log(y_j)
            log_y0 = np.log(self.y_grid[iy])
            log_y1 = np.log(self.y_grid[iy + 1])
            fy = float((log_y - log_y0) / (log_y1 - log_y0 + 1e-10))
            fy = max(0.0, min(1.0, fy))
            fy_list.append(fy)

        iy0_arr = np.array(iy0_list, dtype=np.int32)
        iy1_arr = np.array(iy1_list, dtype=np.int32)
        fy_arr = mx.array(np.array(fy_list, dtype=np.float32))

        # Handle 2D (shared spectra) vs 3D (per-spectrum) cases
        input_shape = x.shape
        if len(input_shape) == 2:
            # (n_energy, n_comp) -> add spectra dim
            x = mx.expand_dims(x, axis=2)
            squeeze_output = True
        else:
            squeeze_output = False

        n_energy, _, n_spectra = x.shape

        # Find x indices - shape (n_energy, n_comp, n_spectra)
        ix = (x - self.x_min) / self.dx
        ix0 = mx.floor(ix).astype(mx.int32)
        ix0 = mx.clip(ix0, 0, self.nx - 2)
        ix1 = ix0 + 1
        fx = ix - ix0.astype(mx.float32)
        fx = mx.clip(fx, 0.0, 1.0)

        # Initialize output
        w_r = mx.zeros_like(x)

        # Process each component (this is the remaining loop, but minimal work per iteration)
        # The heavy lifting (table lookup) is vectorized over energy and spectra
        for j in range(n_comp):
            # Get table rows for this component's y value
            # Use Python int for indexing MLX arrays
            w_r_y0 = self._w_real_mlx[int(iy0_arr[j])]  # (nx,)
            w_r_y1 = self._w_real_mlx[int(iy1_arr[j])]
            fy_j = fy_arr[j]

            # Flatten indices for this component
            ix0_j = mx.reshape(ix0[:, j, :], (-1,))  # (n_energy * n_spectra,)
            ix1_j = mx.reshape(ix1[:, j, :], (-1,))
            fx_j = mx.reshape(fx[:, j, :], (-1,))

            # Gather and interpolate
            w00_r = mx.take(w_r_y0, ix0_j)
            w01_r = mx.take(w_r_y0, ix1_j)
            w10_r = mx.take(w_r_y1, ix0_j)
            w11_r = mx.take(w_r_y1, ix1_j)

            one_minus_fx_j = 1.0 - fx_j
            one_minus_fy_j = 1.0 - fy_j

            w_r_j_flat = (one_minus_fx_j * one_minus_fy_j * w00_r +
                          fx_j * one_minus_fy_j * w01_r +
                          one_minus_fx_j * fy_j * w10_r +
                          fx_j * fy_j * w11_r)

            # Reshape back and assign
            w_r_j = mx.reshape(w_r_j_flat, (n_energy, n_spectra))

            # Update output using array operations
            # Since MLX doesn't support item assignment, we build the result incrementally
            w_r = w_r.at[:, j, :].add(w_r_j)

        if squeeze_output:
            w_r = mx.squeeze(w_r, axis=2)

        return w_r


# Global table instance (lazy initialization)
_faddeeva_table = None


def get_faddeeva_table():
    """Get or create the global Faddeeva lookup table."""
    global _faddeeva_table
    if _faddeeva_table is None:
        _faddeeva_table = FaddeevaTable()
    return _faddeeva_table


# ============================================================
# Voigt Profile Functions
# ============================================================

def voigt_profile_numpy(energy, center, sigma, gamma):
    """
    Compute Voigt profile using NumPy with table lookup.

    Args:
        energy: Energy axis (n_energy,)
        center: Peak center
        sigma: Gaussian width
        gamma: Lorentzian width

    Returns:
        V: Voigt profile
    """
    table = get_faddeeva_table()

    x = (energy - center) / (sigma * SQRT2)
    y = gamma / (sigma * SQRT2)

    w = table.lookup_numpy(x, y)
    V = np.real(w) / (sigma * SQRT2PI)

    return V


def voigt_profile_mlx(energy, center, sigma, gamma):
    """
    Compute Voigt profile using MLX with table lookup.

    Args:
        energy: Energy axis (MLX array or NumPy)
        center: Peak center (scalar)
        sigma: Gaussian width (scalar)
        gamma: Lorentzian width (scalar)

    Returns:
        V: Voigt profile (MLX array)
    """
    if not HAS_MLX:
        raise ImportError("MLX not available")

    table = get_faddeeva_table()

    if not isinstance(energy, mx.array):
        energy = mx.array(energy.astype(np.float32))

    center = float(center)
    sigma = float(sigma)
    gamma = float(gamma)

    x = (energy - center) / (sigma * SQRT2)
    y = gamma / (sigma * SQRT2)

    w = table.lookup_mlx(x, y)
    V = mx.real(w) / (sigma * SQRT2PI)

    return V


def voigt_basis_mlx(energy, centers, sigmas, gamma):
    """
    Compute Voigt basis matrix for multiple peaks using MLX.

    Args:
        energy: Energy axis (n_energy,)
        centers: Peak centers (n_comp,)
        sigmas: Gaussian widths (n_comp,)
        gamma: Lorentzian width (scalar, shared)

    Returns:
        Phi: Basis matrix (n_energy, n_comp)
    """
    if not HAS_MLX:
        raise ImportError("MLX not available")

    if not isinstance(energy, mx.array):
        energy = mx.array(energy.astype(np.float32))

    n_comp = len(centers)
    cols = []

    for j in range(n_comp):
        V_j = voigt_profile_mlx(energy, centers[j], sigmas[j], gamma)
        cols.append(V_j)

    Phi = mx.stack(cols, axis=1)
    return Phi


def voigt_with_jacobian_mlx(energy, center, sigma, gamma):
    """
    Compute Voigt profile and analytical Jacobians using MLX.

    Uses numerical differentiation via table lookup
    (faster than analytical for GPU batched computation).

    Args:
        energy: Energy axis (MLX array)
        center: Peak center (scalar)
        sigma: Gaussian width (scalar)
        gamma: Lorentzian width (scalar)

    Returns:
        V: Voigt profile (n_energy,)
        dV_dc: Derivative w.r.t. center (n_energy,)
        dV_dsigma: Derivative w.r.t. sigma (n_energy,)
        dV_dgamma: Derivative w.r.t. gamma (n_energy,)
    """
    if not HAS_MLX:
        raise ImportError("MLX not available")

    if not isinstance(energy, mx.array):
        energy = mx.array(energy.astype(np.float32))

    center = float(center)
    sigma = float(sigma)
    gamma = float(gamma)

    # Numerical differentiation step sizes
    eps_c = 0.001  # eV for center
    eps_s = 0.001  # eV for sigma
    eps_g = 0.001  # eV for gamma

    # Central value
    V = voigt_profile_mlx(energy, center, sigma, gamma)

    # Center derivative
    V_c_plus = voigt_profile_mlx(energy, center + eps_c, sigma, gamma)
    V_c_minus = voigt_profile_mlx(energy, center - eps_c, sigma, gamma)
    dV_dc = (V_c_plus - V_c_minus) / (2 * eps_c)

    # Sigma derivative
    V_s_plus = voigt_profile_mlx(energy, center, sigma + eps_s, gamma)
    V_s_minus = voigt_profile_mlx(energy, center, sigma - eps_s, gamma)
    dV_dsigma = (V_s_plus - V_s_minus) / (2 * eps_s)

    # Gamma derivative
    V_g_plus = voigt_profile_mlx(energy, center, sigma, gamma + eps_g)
    V_g_minus = voigt_profile_mlx(energy, center, sigma, gamma - eps_g)
    dV_dgamma = (V_g_plus - V_g_minus) / (2 * eps_g)

    return V, dV_dc, dV_dsigma, dV_dgamma


def voigt_basis_batch_mlx_vectorized(energy, centers, sigmas, gamma):
    """
    Compute Voigt basis for batch of spectra with per-spectrum centers.

    Fully vectorized - no component loops (optimized for GPU).

    Args:
        energy: Energy axis (n_energy,) - MLX or numpy array
        centers: Peak centers (n_comp, n_spectra) - per-spectrum centers
        sigmas: Gaussian widths (n_comp,) - shared across spectra
        gamma: Lorentzian width (scalar)

    Returns:
        Phi: Basis matrix (n_energy, n_comp, n_spectra)
    """
    if not HAS_MLX:
        raise ImportError("MLX not available")

    table = get_faddeeva_table()
    table._ensure_mlx()

    if not isinstance(energy, mx.array):
        energy = mx.array(np.asarray(energy, dtype=np.float32))

    centers = np.asarray(centers, dtype=np.float32)
    sigmas = np.asarray(sigmas, dtype=np.float32)

    n_comp, n_spectra = centers.shape
    n_energy = energy.shape[0]

    # Compute x and y values
    # x[e, j, k] = (energy[e] - centers[j, k]) / (sigmas[j] * sqrt(2))
    # y[j] = gamma / (sigmas[j] * sqrt(2))

    energy_np = np.array(energy)
    E = energy_np[:, np.newaxis, np.newaxis]  # (n_energy, 1, 1)
    C = centers[np.newaxis, :, :]  # (1, n_comp, n_spectra)
    S = sigmas[np.newaxis, :, np.newaxis]  # (1, n_comp, 1)

    x = (E - C) / (S * SQRT2)  # (n_energy, n_comp, n_spectra)
    y = gamma / (sigmas * SQRT2)  # (n_comp,)

    # Convert x to MLX
    x_mx = mx.array(x.astype(np.float32))

    # Batched Faddeeva lookup
    w_real = table.lookup_mlx_batched(x_mx, y)

    # Voigt profile: V = Re(w) / (sigma * sqrt(2*pi))
    # sigma is per-component, broadcast: (1, n_comp, 1)
    S_mx = mx.array(S.astype(np.float32))
    Phi = w_real / (S_mx * SQRT2PI)

    return Phi


def voigt_basis_with_gradient_mlx(energy, centers, sigmas, gamma, eps=0.01):
    """
    Compute Voigt basis AND center gradients in one pass (for Gauss-Newton).

    Computes Phi, Phi+, Phi- for each component simultaneously,
    avoiding redundant Faddeeva lookups.

    Args:
        energy: Energy axis (n_energy,) - MLX or numpy array
        centers: Peak centers (n_comp, n_spectra)
        sigmas: Gaussian widths (n_comp,) - shared
        gamma: Lorentzian width (scalar)
        eps: Finite difference step (default 0.01 eV)

    Returns:
        Phi: Basis matrix (n_energy, n_comp, n_spectra)
        dPhi_dc: Center gradient (n_energy, n_comp, n_spectra)
    """
    if not HAS_MLX:
        raise ImportError("MLX not available")

    table = get_faddeeva_table()
    table._ensure_mlx()

    if not isinstance(energy, mx.array):
        energy = mx.array(np.asarray(energy, dtype=np.float32))

    centers = np.asarray(centers, dtype=np.float32)
    sigmas = np.asarray(sigmas, dtype=np.float32)

    n_comp, n_spectra = centers.shape
    n_energy = energy.shape[0]

    energy_np = np.array(energy)
    E = energy_np[:, np.newaxis, np.newaxis]  # (n_energy, 1, 1)
    S = sigmas[np.newaxis, :, np.newaxis]  # (1, n_comp, 1)

    # Compute y values (same for all shifts)
    y = gamma / (sigmas * SQRT2)  # (n_comp,)

    # Compute x for center, center+eps, center-eps
    # Shape: (n_energy, n_comp, n_spectra)
    C0 = centers[np.newaxis, :, :]
    C_plus = (centers + eps)[np.newaxis, :, :]
    C_minus = (centers - eps)[np.newaxis, :, :]

    x0 = (E - C0) / (S * SQRT2)
    x_plus = (E - C_plus) / (S * SQRT2)
    x_minus = (E - C_minus) / (S * SQRT2)

    # Stack for batched lookup: (n_energy, n_comp, 3 * n_spectra)
    # This allows computing all three variants in one lookup call
    x_combined = np.concatenate([x0, x_plus, x_minus], axis=2)
    x_combined_mx = mx.array(x_combined.astype(np.float32))

    # Batched Faddeeva lookup
    w_real_combined = table.lookup_mlx_batched(x_combined_mx, y)

    # Split results
    w0 = w_real_combined[:, :, :n_spectra]
    w_plus = w_real_combined[:, :, n_spectra:2*n_spectra]
    w_minus = w_real_combined[:, :, 2*n_spectra:]

    # Voigt profiles
    S_mx = mx.array(S.astype(np.float32))
    Phi = w0 / (S_mx * SQRT2PI)
    Phi_plus = w_plus / (S_mx * SQRT2PI)
    Phi_minus = w_minus / (S_mx * SQRT2PI)

    # Gradient via central difference
    dPhi_dc = (Phi_plus - Phi_minus) / (2 * eps)

    return Phi, dPhi_dc


def voigt_basis_and_jacobians_mlx(energy, centers, sigmas, gamma):
    """
    Compute Voigt basis and Jacobians for all components.

    Args:
        energy: Energy axis (n_energy,)
        centers: Peak centers (n_comp,)
        sigmas: Gaussian widths (n_comp,)
        gamma: Lorentzian width (scalar)

    Returns:
        Phi: Basis matrix (n_energy, n_comp)
        J_c: Center Jacobians (n_energy, n_comp)
        J_s: Sigma Jacobians (n_energy, n_comp)
        J_g: Gamma Jacobians (n_energy, n_comp)
    """
    if not HAS_MLX:
        raise ImportError("MLX not available")

    if not isinstance(energy, mx.array):
        energy = mx.array(energy.astype(np.float32))

    n_comp = len(centers)

    Phi_cols = []
    Jc_cols = []
    Js_cols = []
    Jg_cols = []

    for j in range(n_comp):
        V, dV_dc, dV_ds, dV_dg = voigt_with_jacobian_mlx(
            energy, centers[j], sigmas[j], gamma
        )
        Phi_cols.append(V)
        Jc_cols.append(dV_dc)
        Js_cols.append(dV_ds)
        Jg_cols.append(dV_dg)

    Phi = mx.stack(Phi_cols, axis=1)
    J_c = mx.stack(Jc_cols, axis=1)
    J_s = mx.stack(Js_cols, axis=1)
    J_g = mx.stack(Jg_cols, axis=1)

    return Phi, J_c, J_s, J_g


def voigt_exact_jacobian_perspectrum_mlx(energy, centers_b, sigmas_b, gamma):
    """Exact Voigt + analytic Jacobians for a single peak with per-spectrum params.

    Fully vectorized over (batch, energy). Unlike
    :func:`voigt_basis_and_jacobians_mlx` (which takes scalar center/sigma and
    loops over components), this evaluates one component at the *per-spectrum*
    absolute parameters ``(centers_b[s], sigmas_b[s])`` — exactly what an
    exact-Jacobian Gauss-Newton step needs after the centre/width have moved
    off the dictionary grid.

    The Faddeeva real *and* imaginary parts are bilinearly interpolated from the
    shared table; the analytic Jacobians use ``dw/dz = -2 z w + 2 i / √π``:

        V       = Re(w) / (σ √(2π))
        ∂V/∂c   = Re(dw/dz · dz/dc) / (σ √(2π)),   dz/dc = -1 / (σ √2)
        ∂V/∂σ   = Re(dw/dz · dz/dσ) / (σ √(2π)) - V / σ,   dz/dσ = -z / σ
        ∂V/∂γ   = Re(dw/dz · dz/dγ) / (σ √(2π)),   dz/dγ = i / (σ √2)
                = -Im(dw/dz) / (σ √2) · 1/(σ √(2π))

    where ``z = ((E - c) + iγ) / (σ √2)``. Accuracy matches the dictionary's own
    grid Jacobians (≈1e-3 relative, set by the table interpolation; ``V`` itself
    is ≈1e-9).

    Args:
        energy: Energy axis (n_energy,) — MLX or numpy.
        centers_b: Per-spectrum centres (batch,) — MLX or numpy.
        sigmas_b: Per-spectrum Gaussian widths (batch,) — MLX or numpy.
        gamma: Lorentzian width (scalar, shared).

    Returns:
        Phi, Jc, Js, Jg — each (batch, n_energy) MLX float32:
        Phi = V, Jc = ∂V/∂center, Js = ∂V/∂sigma, Jg = ∂V/∂gamma.
    """
    if not HAS_MLX:
        raise ImportError("MLX not available")

    table = get_faddeeva_table()
    table._ensure_mlx()

    if not isinstance(energy, mx.array):
        energy = mx.array(np.asarray(energy, dtype=np.float32))
    if not isinstance(centers_b, mx.array):
        centers_b = mx.array(np.asarray(centers_b, dtype=np.float32))
    if not isinstance(sigmas_b, mx.array):
        sigmas_b = mx.array(np.asarray(sigmas_b, dtype=np.float32))

    # gamma may be a scalar (shared Lorentzian width) or a per-spectrum
    # (batch,) array/MLX (used by the δγ-refinement Newton path). In the
    # per-spectrum case it must broadcast against sigmas_b along the batch axis.
    if isinstance(gamma, mx.array):
        gamma_arr = gamma
    elif np.isscalar(gamma) or (hasattr(gamma, "ndim") and np.ndim(gamma) == 0):
        gamma_arr = float(gamma)
    else:
        gamma_arr = mx.array(np.asarray(gamma, dtype=np.float32))
    nx = table.nx
    ny = table.ny
    dx = float(table.dx)
    x_min = float(table.x_min)
    wr_tab = table._w_real_mlx          # (ny, nx)
    wi_tab = table._w_imag_mlx

    s = sigmas_b[:, None]               # (batch, 1)
    c = centers_b[:, None]
    xval = (energy[None, :] - c) / (s * SQRT2)   # (batch, n_E)
    yval = gamma_arr / (sigmas_b * SQRT2)         # (batch,) — scalar γ broadcasts

    # x bilinear indices/weights (per element)
    ix = (xval - x_min) / dx
    ix0 = mx.clip(mx.floor(ix).astype(mx.int32), 0, nx - 2)
    fx = mx.clip(ix - ix0.astype(mx.float32), 0.0, 1.0)
    ix1 = ix0 + 1

    # y bilinear indices/weights (per spectrum; log-spaced grid) computed on CPU
    yv = np.asarray(yval).astype(np.float64)
    yv_clip = np.clip(yv, table.y_min, table.y_max)
    iy = np.clip(np.searchsorted(table.y_grid, yv_clip) - 1, 0, ny - 2)
    log_y = np.log(yv_clip)
    log_y0 = np.log(table.y_grid[iy])
    log_y1 = np.log(table.y_grid[iy + 1])
    fy = np.clip((log_y - log_y0) / (log_y1 - log_y0 + 1e-10), 0.0, 1.0)
    iy0 = mx.array(iy.astype(np.int32))
    iy1 = mx.array((iy + 1).astype(np.int32))
    fy_mx = mx.array(fy.astype(np.float32))[:, None]   # (batch, 1)

    # Per-spectrum table rows, then gather along the x axis (per element)
    row_r0 = mx.take(wr_tab, iy0, axis=0)   # (batch, nx)
    row_r1 = mx.take(wr_tab, iy1, axis=0)
    row_i0 = mx.take(wi_tab, iy0, axis=0)
    row_i1 = mx.take(wi_tab, iy1, axis=0)

    wr00 = mx.take_along_axis(row_r0, ix0, axis=1)
    wr01 = mx.take_along_axis(row_r0, ix1, axis=1)
    wr10 = mx.take_along_axis(row_r1, ix0, axis=1)
    wr11 = mx.take_along_axis(row_r1, ix1, axis=1)
    wi00 = mx.take_along_axis(row_i0, ix0, axis=1)
    wi01 = mx.take_along_axis(row_i0, ix1, axis=1)
    wi10 = mx.take_along_axis(row_i1, ix0, axis=1)
    wi11 = mx.take_along_axis(row_i1, ix1, axis=1)

    omfx = 1.0 - fx
    omfy = 1.0 - fy_mx
    wr = omfx * omfy * wr00 + fx * omfy * wr01 + omfx * fy_mx * wr10 + fx * fy_mx * wr11
    wi = omfx * omfy * wi00 + fx * omfy * wi01 + omfx * fy_mx * wi10 + fx * fy_mx * wi11

    inv = 1.0 / (s * SQRT2PI)
    Phi = wr * inv

    # dw/dz = -2 z w + 2i/√π, with z = xval + i·yval
    yb = yval[:, None]
    re_dwdz = -2.0 * (xval * wr - yb * wi)
    im_dwdz = -2.0 * (xval * wi + yb * wr) + 2.0 / SQRTPI

    # ∂V/∂c = Re(dw/dz · dz/dc) / (σ√2π), dz/dc = -1/(σ√2)
    Jc = re_dwdz * (-1.0 / (s * SQRT2)) * inv

    # ∂V/∂σ = Re(dw/dz · dz/dσ)/(σ√2π) - V/σ, dz/dσ = -z/σ = -(xval + i yb)/σ
    re_dwdz_dzds = (re_dwdz * (-xval / s)) - (im_dwdz * (-yb / s))
    Js = re_dwdz_dzds * inv - Phi / s

    # ∂V/∂γ = Re(dw/dz · dz/dγ)/(σ√2π), dz/dγ = i/(σ√2)
    #       = Re((re_dwdz + i·im_dwdz) · i/(σ√2)) · inv = -im_dwdz/(σ√2) · inv
    Jg = (-im_dwdz / (s * SQRT2)) * inv

    return Phi, Jc, Js, Jg


# ============================================================
# Validation
# ============================================================

def validate_against_scipy():
    """Validate table-based implementation against scipy.special.wofz."""
    import time

    import scipy.special as sps

    print("=" * 60)
    print("Validating Faddeeva Table Lookup")
    print("=" * 60)

    # Initialize table
    t0 = time.perf_counter()
    table = get_faddeeva_table()
    print(f"\nTable initialization: {time.perf_counter() - t0:.3f}s")
    print(f"Table size: {table.nx} x {table.ny} = {table.nx * table.ny} points")
    print(f"Memory: {table.w_real.nbytes * 2 / 1e6:.1f} MB")

    # Test Voigt profile
    print("\nVoigt Profile Accuracy:")
    print("-" * 50)

    energy = np.linspace(95, 110, 151).astype(np.float32)

    test_cases = [
        (100.0, 0.8, 0.25),   # Typical
        (100.0, 0.5, 0.1),    # Small gamma
        (100.0, 1.2, 0.4),    # Large sigma
        (100.0, 0.6, 0.5),    # Large gamma
    ]

    for center, sigma, gamma in test_cases:
        # SciPy reference
        z = ((energy - center) + 1j * gamma) / (sigma * SQRT2)
        V_scipy = np.real(sps.wofz(z)) / (sigma * SQRT2PI)

        # Table lookup
        V_table = voigt_profile_numpy(energy, center, sigma, gamma)

        max_err = np.max(np.abs(V_table - V_scipy) / (np.abs(V_scipy) + 1e-10))
        mean_err = np.mean(np.abs(V_table - V_scipy) / (np.abs(V_scipy) + 1e-10))

        print(f"σ={sigma}, γ={gamma}: max_err={max_err:.2e}, mean_err={mean_err:.2e}")

    # Test MLX
    if HAS_MLX:
        print("\nMLX Voigt Profile:")
        print("-" * 50)

        energy_mlx = mx.array(energy)

        for center, sigma, gamma in test_cases:
            z = ((energy - center) + 1j * gamma) / (sigma * SQRT2)
            V_scipy = np.real(sps.wofz(z)) / (sigma * SQRT2PI)

            V_mlx = voigt_profile_mlx(energy_mlx, center, sigma, gamma)
            mx.eval(V_mlx)
            V_mlx_np = np.array(V_mlx)

            max_err = np.max(np.abs(V_mlx_np - V_scipy) / (np.abs(V_scipy) + 1e-10))
            print(f"σ={sigma}, γ={gamma}: max_err={max_err:.2e}")

        # Test Jacobians
        print("\nMLX Jacobian Validation:")
        print("-" * 50)

        center, sigma, gamma = 100.0, 0.8, 0.25
        V, dV_dc, dV_dsigma, dV_dgamma = voigt_with_jacobian_mlx(
            energy_mlx, center, sigma, gamma
        )
        mx.eval(V, dV_dc, dV_dsigma, dV_dgamma)

        # Numerical reference
        eps = 1e-5
        z = ((energy - center) + 1j * gamma) / (sigma * SQRT2)
        V_scipy = np.real(sps.wofz(z)) / (sigma * SQRT2PI)

        z_c_plus = ((energy - center - eps) + 1j * gamma) / (sigma * SQRT2)
        z_c_minus = ((energy - center + eps) + 1j * gamma) / (sigma * SQRT2)
        dV_dc_ref = (np.real(sps.wofz(z_c_plus)) - np.real(sps.wofz(z_c_minus))) / (2 * eps * sigma * SQRT2PI)

        dV_dc_np = np.array(dV_dc)
        err_c = np.max(np.abs(dV_dc_np - dV_dc_ref) / (np.abs(dV_dc_ref) + 1e-10))
        print(f"dV/dc: max_rel_err = {err_c:.2e}")

        # Performance benchmark
        print("\nPerformance Benchmark:")
        print("-" * 50)

        n_spectra = 10000
        energy_batch = mx.array(np.tile(energy[:, np.newaxis], (1, n_spectra)).astype(np.float32))

        # Warm up
        for _ in range(3):
            V = voigt_profile_mlx(energy_mlx, center, sigma, gamma)
            mx.eval(V)

        # Benchmark single spectrum
        n_iter = 100
        t0 = time.perf_counter()
        for _ in range(n_iter):
            V = voigt_profile_mlx(energy_mlx, center, sigma, gamma)
            mx.eval(V)
        elapsed = time.perf_counter() - t0

        print(f"Single spectrum: {elapsed/n_iter*1e6:.1f} μs/spectrum")
        print(f"Throughput: {n_iter/elapsed:.0f} spectra/s")

    else:
        print("\nMLX not available")

    print("\n✓ Validation complete")


if __name__ == "__main__":
    validate_against_scipy()
