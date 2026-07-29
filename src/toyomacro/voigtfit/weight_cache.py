"""
Weight Matrix Cache for 2-Stage Hybrid Pipeline
================================================

Pre-computes and caches weight matrices for each element/orbital combination.

Theory:
    Linear least squares: min ||Y - Φ·A||²
    Solution: A = (Φ'Φ)⁻¹ Φ' Y = Wt @ Y

    Pre-compute Wt = inv(Φ'Φ + λI) @ Φ' once per element/orbital,
    then runtime is just a single matmul: A = Wt @ Y

Performance:
    - Wt computation: ~10ms per element/orbital (done once)
    - Runtime: single matmul → 500M spectra/s on M3 Max

    Theoretical limit (memory bandwidth):
    - M3 Max: 400 GB/s
    - 100 energy points × 4 bytes = 400 B/spectrum
    - Max: 1000M spec/s
    - Achieved: 500M spec/s (50% efficiency, memory-bound)
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import special as sps
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import mlx.core as mx

    from ._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND the default device can execute work
except ImportError:
    HAS_MLX = False

SQRT2 = np.sqrt(2.0)
SQRT2PI = np.sqrt(2.0 * np.pi)


class WeightMatrixCache:
    """
    Pre-computed weight matrix cache for fast linear fitting.

    Usage:
        cache = WeightMatrixCache()
        Wt, Phi = cache.get_or_create(
            element="Fe", orbital="2p3/2",
            energy=energy_axis,
            centers=np.array([707.0, 709.5, 711.0]),
            sigmas=np.array([0.8, 0.9, 0.8]),
            gamma=0.2
        )

        # Fast fitting: A = Wt @ Y
        amplitudes = Wt @ spectra  # (n_components, n_spectra)
    """

    def __init__(self, cache_dir: str = "~/.voigtfit/cache"):
        """
        Initialize weight matrix cache.

        Args:
            cache_dir: Directory for disk cache (default: ~/.voigtfit/cache)
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, tuple[Any, Any]] = {}

    def get_key(
        self,
        element: str,
        orbital: str,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float,
        alphas: np.ndarray | None = None,
        lineshape_types: list[str] | None = None,
    ) -> str:
        """
        Generate unique cache key from parameters.

        Uses hash of all parameters to ensure cache invalidation
        when any parameter changes.
        """
        # Create deterministic hash
        params = {
            "element": element,
            "orbital": orbital,
            "e_start": float(energy[0]),
            "e_end": float(energy[-1]),
            "e_len": len(energy),
            "centers": centers.tolist(),
            "sigmas": sigmas.tolist(),
            "gamma": gamma,
        }
        if alphas is not None:
            params["alphas"] = alphas.tolist()
        if lineshape_types is not None:
            params["lineshape_types"] = lineshape_types
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
        # Replace / with _ for safe filename
        safe_orbital = orbital.replace("/", "_")
        return f"{element}_{safe_orbital}_{param_hash}"

    def build_basis_voigt(
        self,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float = 0.2,
    ) -> np.ndarray:
        """
        Build Voigt basis matrix using Faddeeva function.

        The Voigt profile V(x) is the real part of the Faddeeva function:
            V(x; σ, γ) = Re[w(z)] / (σ√(2π))
            where z = (x + iγ) / (σ√2)

        Args:
            energy: Energy axis (n_energy,)
            centers: Peak center positions (n_components,)
            sigmas: Gaussian widths (n_components,)
            gamma: Lorentzian width (scalar, shared)

        Returns:
            Phi: Basis matrix (n_energy, n_components)
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for Faddeeva function")

        n_energy = len(energy)
        n_components = len(centers)
        Phi = np.zeros((n_energy, n_components), dtype=np.float32)

        for k in range(n_components):
            # Complex argument for Faddeeva function
            z = ((energy - centers[k]) + 1j * gamma) / (sigmas[k] * SQRT2)
            # Voigt profile = Re[w(z)] / (σ√(2π))
            Phi[:, k] = np.real(sps.wofz(z)).astype(np.float32) / (sigmas[k] * SQRT2PI)

        return Phi

    def build_basis_ds(
        self,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float,
        alphas: np.ndarray,
    ) -> np.ndarray:
        """
        Build Doniach-Sunjic basis matrix.

        DS(x) = Γ(1-α) cos(πα/2 + (1-α)arctan(Δ/γ)) / (Δ²+γ²)^((1-α)/2)
        Optionally convolved with Gaussian (sigma > 0).

        Args:
            energy: Energy axis (n_energy,)
            centers: Peak center positions (n_components,)
            sigmas: Gaussian widths for broadening (n_components,)
            gamma: Lorentzian half-width (scalar, shared)
            alphas: Asymmetry indices (n_components,)

        Returns:
            Phi: Basis matrix (n_energy, n_components), area-normalized, float32
        """
        from .ds_profile import ds_with_gaussian

        n_energy = len(energy)
        n_components = len(centers)
        Phi = np.zeros((n_energy, n_components), dtype=np.float32)

        for k in range(n_components):
            Phi[:, k] = ds_with_gaussian(
                energy, centers[k], alphas[k], gamma, sigmas[k],
            )

        return Phi

    def build_basis_mixed(
        self,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float,
        lineshape_types: list[str],
        alphas: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build mixed Voigt + DS basis matrix.

        Args:
            energy: Energy axis (n_energy,)
            centers: Peak center positions (n_components,)
            sigmas: Gaussian widths (n_components,)
            gamma: Lorentzian half-width (scalar, shared)
            lineshape_types: "voigt" or "ds" per component
            alphas: Asymmetry indices for DS components (required if any "ds")

        Returns:
            Phi: Combined basis matrix (n_energy, n_components), float32
        """
        n_energy = len(energy)
        n_components = len(centers)
        Phi = np.zeros((n_energy, n_components), dtype=np.float32)

        voigt_idx = [k for k, t in enumerate(lineshape_types) if t == "voigt"]
        ds_idx = [k for k, t in enumerate(lineshape_types) if t == "ds"]

        if ds_idx and alphas is None:
            raise ValueError("alphas required for DS components")

        # Build Voigt columns
        if voigt_idx:
            Phi_v = self.build_basis_voigt(
                energy,
                centers[voigt_idx],
                sigmas[voigt_idx],
                gamma,
            )
            for i, k in enumerate(voigt_idx):
                Phi[:, k] = Phi_v[:, i]

        # Build DS columns
        if ds_idx:
            Phi_ds = self.build_basis_ds(
                energy,
                centers[ds_idx],
                sigmas[ds_idx],
                gamma,
                alphas[ds_idx],
            )
            for i, k in enumerate(ds_idx):
                Phi[:, k] = Phi_ds[:, i]

        return Phi

    def compute_weight_matrix(
        self,
        Phi: np.ndarray,
        reg_lambda: float = 1e-8,
    ) -> np.ndarray:
        """
        Compute weight matrix Wt = inv(Φ'Φ + λI) @ Φ'.

        This is the pre-computed linear least squares solver.
        At runtime: A = Wt @ Y (single matmul)

        Args:
            Phi: Basis matrix (n_energy, n_components)
            reg_lambda: Tikhonov regularization (default 1e-8)

        Returns:
            Wt: Weight matrix (n_components, n_energy)
        """
        n_components = Phi.shape[1]
        AtA = Phi.T @ Phi + reg_lambda * np.eye(n_components, dtype=np.float32)
        Wt = np.linalg.solve(AtA, Phi.T)  # More stable than inv()
        return Wt.astype(np.float32)

    def get_or_create(
        self,
        element: str,
        orbital: str,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float = 0.2,
        reg_lambda: float = 1e-8,
        use_mlx: bool = True,
        alphas: np.ndarray | None = None,
        lineshape_types: list[str] | None = None,
    ) -> tuple[Any, Any]:
        """
        Get weight matrix from cache, or create if not exists.

        Args:
            element: Element symbol (e.g., "Fe")
            orbital: Orbital name (e.g., "2p3/2")
            energy: Energy axis (n_energy,)
            centers: Peak center positions (n_components,)
            sigmas: Gaussian widths (n_components,)
            gamma: Lorentzian width (scalar)
            reg_lambda: Tikhonov regularization
            use_mlx: Convert to MLX arrays if available
            alphas: DS asymmetry per component (None = all Voigt)
            lineshape_types: "voigt"/"ds" per component (None = all Voigt)

        Returns:
            (Wt, Phi): Weight matrix and basis matrix
                       Returns MLX arrays if use_mlx=True and MLX available
        """
        # Ensure numpy arrays
        energy = np.asarray(energy, dtype=np.float32)
        centers = np.asarray(centers, dtype=np.float32)
        sigmas = np.asarray(sigmas, dtype=np.float32)
        if alphas is not None:
            alphas = np.asarray(alphas, dtype=np.float32)

        has_ds = lineshape_types is not None and "ds" in lineshape_types

        key = self.get_key(
            element, orbital, energy, centers, sigmas, gamma,
            alphas=alphas, lineshape_types=lineshape_types,
        )

        # Check memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            Wt = data["Wt"]
            Phi = data["Phi"]

            # Convert to MLX if requested
            if use_mlx and HAS_MLX:
                Wt = mx.array(Wt)
                Phi = mx.array(Phi)

            self._memory_cache[key] = (Wt, Phi)
            return Wt, Phi

        # Create new basis and weight matrix
        if has_ds:
            Phi = self.build_basis_mixed(
                energy, centers, sigmas, gamma, lineshape_types, alphas,
            )
        else:
            Phi = self.build_basis_voigt(energy, centers, sigmas, gamma)
        Wt = self.compute_weight_matrix(Phi, reg_lambda)

        # Save to disk
        np.savez(
            cache_file,
            Wt=Wt,
            Phi=Phi,
            energy=energy,
            centers=centers,
            sigmas=sigmas,
            gamma=np.array([gamma]),
            reg_lambda=np.array([reg_lambda]),
        )

        # Convert to MLX if requested
        if use_mlx and HAS_MLX:
            Wt = mx.array(Wt)
            Phi = mx.array(Phi)

        self._memory_cache[key] = (Wt, Phi)
        return Wt, Phi

    def build_basis_voigt_with_jacobian(
        self,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build Voigt basis matrix and analytical center Jacobian.

        Uses the Faddeeva function derivative for exact dΦ/dc.
        Energy shift derivative: dΦ/dE = -dΦ/dc (shift E ↔ shift c oppositely).

        Args:
            energy: Energy axis (n_energy,)
            centers: Peak center positions (n_components,)
            sigmas: Gaussian widths (n_components,)
            gamma: Lorentzian width (scalar, shared)

        Returns:
            (Phi, J_c): Basis matrix and center Jacobian, both (n_energy, n_components)
        """
        from .voigt_jacobian import voigt_jacobian_batch

        Phi, J_c, _, _ = voigt_jacobian_batch(
            energy, centers, sigmas, gamma
        )
        return Phi.astype(np.float32), J_c.astype(np.float32)

    def get_or_create_with_jacobian(
        self,
        element: str,
        orbital: str,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float = 0.2,
        reg_lambda: float = 1e-8,
        use_mlx: bool = True,
    ) -> tuple[Any, Any, Any]:
        """
        Get weight matrix, basis, AND center Jacobian from cache.

        Used by the extended SVD shift correction pipeline.
        The Jacobian J_c = ∂Φ/∂c is needed for residual projection:
            δE = -Σ(r · (J_c @ a)) / Σ(J_c @ a)²

        Args:
            element: Element symbol
            orbital: Orbital name
            energy: Energy axis (n_energy,)
            centers: Peak center positions (n_components,)
            sigmas: Gaussian widths (n_components,)
            gamma: Lorentzian width (scalar)
            reg_lambda: Tikhonov regularization
            use_mlx: Convert to MLX arrays if available

        Returns:
            (Wt, Phi, J_c): Weight matrix, basis, and center Jacobian
        """
        energy = np.asarray(energy, dtype=np.float32)
        centers = np.asarray(centers, dtype=np.float32)
        sigmas = np.asarray(sigmas, dtype=np.float32)

        key = self.get_key(element, orbital, energy, centers, sigmas, gamma) + "_jac"

        if key in self._memory_cache:
            return self._memory_cache[key]

        cache_file = self.cache_dir / f"{key}.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            Wt = data["Wt"]
            Phi = data["Phi"]
            J_c = data["J_c"]

            if use_mlx and HAS_MLX:
                Wt = mx.array(Wt)
                Phi = mx.array(Phi)
                J_c = mx.array(J_c)

            self._memory_cache[key] = (Wt, Phi, J_c)
            return Wt, Phi, J_c

        # Build with analytical Jacobian
        Phi, J_c = self.build_basis_voigt_with_jacobian(energy, centers, sigmas, gamma)
        Wt = self.compute_weight_matrix(Phi, reg_lambda)

        np.savez(
            cache_file,
            Wt=Wt,
            Phi=Phi,
            J_c=J_c,
            energy=energy,
            centers=centers,
            sigmas=sigmas,
            gamma=np.array([gamma]),
            reg_lambda=np.array([reg_lambda]),
        )

        if use_mlx and HAS_MLX:
            Wt = mx.array(Wt)
            Phi = mx.array(Phi)
            J_c = mx.array(J_c)

        self._memory_cache[key] = (Wt, Phi, J_c)
        return Wt, Phi, J_c

    def build_basis_voigt_with_full_jacobian(
        self,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build Voigt basis matrix with center AND sigma Jacobians.

        Returns:
            (Phi, J_c, J_sigma): all (n_energy, n_components)
        """
        from .voigt_jacobian import voigt_jacobian_batch

        Phi, J_c, J_sigma, _ = voigt_jacobian_batch(
            energy, centers, sigmas, gamma
        )
        return (Phi.astype(np.float32), J_c.astype(np.float32),
                J_sigma.astype(np.float32))

    def get_or_create_with_full_jacobian(
        self,
        element: str,
        orbital: str,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float = 0.2,
        reg_lambda: float = 1e-8,
        use_mlx: bool = True,
    ) -> tuple[Any, Any, Any, Any]:
        """
        Get weight matrix, basis, center Jacobian, AND sigma Jacobian.

        Used by the 3-step residual projection pipeline (shift + width).

        Returns:
            (Wt, Phi, J_c, J_sigma)
        """
        energy = np.asarray(energy, dtype=np.float32)
        centers = np.asarray(centers, dtype=np.float32)
        sigmas = np.asarray(sigmas, dtype=np.float32)

        key = self.get_key(element, orbital, energy, centers, sigmas, gamma) + "_fulljac"

        if key in self._memory_cache:
            return self._memory_cache[key]

        cache_file = self.cache_dir / f"{key}.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            Wt = data["Wt"]
            Phi = data["Phi"]
            J_c = data["J_c"]
            J_sigma = data["J_sigma"]

            if use_mlx and HAS_MLX:
                Wt = mx.array(Wt)
                Phi = mx.array(Phi)
                J_c = mx.array(J_c)
                J_sigma = mx.array(J_sigma)

            self._memory_cache[key] = (Wt, Phi, J_c, J_sigma)
            return Wt, Phi, J_c, J_sigma

        Phi, J_c, J_sigma = self.build_basis_voigt_with_full_jacobian(
            energy, centers, sigmas, gamma
        )
        Wt = self.compute_weight_matrix(Phi, reg_lambda)

        np.savez(
            cache_file,
            Wt=Wt, Phi=Phi, J_c=J_c, J_sigma=J_sigma,
            energy=energy, centers=centers, sigmas=sigmas,
            gamma=np.array([gamma]), reg_lambda=np.array([reg_lambda]),
        )

        if use_mlx and HAS_MLX:
            Wt = mx.array(Wt)
            Phi = mx.array(Phi)
            J_c = mx.array(J_c)
            J_sigma = mx.array(J_sigma)

        self._memory_cache[key] = (Wt, Phi, J_c, J_sigma)
        return Wt, Phi, J_c, J_sigma

    def build_basis_voigt_with_hessian(
        self,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build Voigt basis matrix with Jacobians AND d²V/dc² Hessian.

        Returns:
            (Phi, J_c, J_sigma, H_cc): all (n_energy, n_components)
        """
        from .voigt_jacobian import voigt_hessian_batch

        Phi, J_c, J_sigma, _, H_cc = voigt_hessian_batch(
            energy, centers, sigmas, gamma
        )
        return (Phi.astype(np.float32), J_c.astype(np.float32),
                J_sigma.astype(np.float32), H_cc.astype(np.float32))

    def get_or_create_with_hessian(
        self,
        element: str,
        orbital: str,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float = 0.2,
        reg_lambda: float = 1e-8,
        use_mlx: bool = True,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """
        Get weight matrix, basis, Jacobians, AND d²V/dc² Hessian.

        Used by the 6-step residual projection pipeline.

        Returns:
            (Wt, Phi, J_c, J_sigma, H_cc)
        """
        energy = np.asarray(energy, dtype=np.float32)
        centers = np.asarray(centers, dtype=np.float32)
        sigmas = np.asarray(sigmas, dtype=np.float32)

        key = self.get_key(element, orbital, energy, centers, sigmas, gamma) + "_hess"

        if key in self._memory_cache:
            return self._memory_cache[key]

        cache_file = self.cache_dir / f"{key}.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            Wt = data["Wt"]
            Phi = data["Phi"]
            J_c = data["J_c"]
            J_sigma = data["J_sigma"]
            H_cc = data["H_cc"]

            if use_mlx and HAS_MLX:
                Wt = mx.array(Wt)
                Phi = mx.array(Phi)
                J_c = mx.array(J_c)
                J_sigma = mx.array(J_sigma)
                H_cc = mx.array(H_cc)

            self._memory_cache[key] = (Wt, Phi, J_c, J_sigma, H_cc)
            return Wt, Phi, J_c, J_sigma, H_cc

        Phi, J_c, J_sigma, H_cc = self.build_basis_voigt_with_hessian(
            energy, centers, sigmas, gamma
        )
        Wt = self.compute_weight_matrix(Phi, reg_lambda)

        np.savez(
            cache_file,
            Wt=Wt, Phi=Phi, J_c=J_c, J_sigma=J_sigma, H_cc=H_cc,
            energy=energy, centers=centers, sigmas=sigmas,
            gamma=np.array([gamma]), reg_lambda=np.array([reg_lambda]),
        )

        if use_mlx and HAS_MLX:
            Wt = mx.array(Wt)
            Phi = mx.array(Phi)
            J_c = mx.array(J_c)
            J_sigma = mx.array(J_sigma)
            H_cc = mx.array(H_cc)

        self._memory_cache[key] = (Wt, Phi, J_c, J_sigma, H_cc)
        return Wt, Phi, J_c, J_sigma, H_cc

    def clear_memory_cache(self):
        """Clear in-memory cache (disk cache remains)."""
        self._memory_cache.clear()

    def clear_all(self):
        """Clear both memory and disk cache."""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.npz"):
            f.unlink()

    def info(self) -> dict:
        """Get cache statistics."""
        disk_files = list(self.cache_dir.glob("*.npz"))
        total_size = sum(f.stat().st_size for f in disk_files)
        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": len(disk_files),
            "disk_size_mb": total_size / 1024 / 1024,
            "cache_dir": str(self.cache_dir),
        }
