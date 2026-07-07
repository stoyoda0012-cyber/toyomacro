"""
Cramer-Rao Lower Bound (CRLB) for Multi-Peak Voigt Spectra
===========================================================

Computes the theoretical minimum variance for parameter estimation
of overlapping Voigt peaks under Poisson noise.

For n_comp peaks with parameters theta_k = (A_k, dE_k, dsigma_k [, dgamma_k]),
the total parameter vector is theta = (theta_0, theta_1, ..., theta_{n-1}).

The Fisher Information Matrix is:

    g_ij = sum_E (df/dtheta_i)(df/dtheta_j) / f(E; theta)

where f(E) = sum_k A_k V_k(E) is the composite spectrum (Poisson mean).

The CRLB states: Var(theta_hat_i) >= [g^{-1}]_ii

Key insight: overlapping peaks share the Poisson denominator f(E),
creating off-diagonal blocks that couple parameters of different peaks.
As overlap increases, the Fisher matrix becomes ill-conditioned and
CRLB diverges -- quantifying the fundamental limit of peak separation.

References:
    Rao (1945), Cramer (1946)
    Kay (1993) "Fundamentals of Statistical Signal Processing: Estimation Theory"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

import numpy as np

from .voigt_jacobian import voigt_with_jacobian

# FWHM <-> sigma conversion
FWHM_TO_SIGMA = 1.0 / (2 * np.sqrt(2 * np.log(2)))


# ===================================================================
# Solvability classification
# ===================================================================


class SolvabilityLevel(Enum):
    """Classification of peak separation difficulty based on CRLB.

    Based on sqrt(CRLB[dE]) relative to Gaussian width sigma:
        EASY:       < 2% of sigma  — meV precision, well-resolved
        HARD:       2-10% of sigma — tens of meV, valley visible but shallow
        SHOULDER:   10-40% of sigma — shoulder visible, position uncertain
        IMPOSSIBLE: > 40% of sigma — peaks merge into one feature
    """
    EASY = "easy"
    HARD = "hard"
    SHOULDER = "shoulder"
    IMPOSSIBLE = "impossible"


@dataclass
class SolvabilityInfo:
    """Solvability assessment for a multi-peak configuration.

    Attributes:
        level: Overall solvability classification
        crlb_dE_meV: sqrt(CRLB[dE]) in meV, averaged over components
        crlb_ds_meV: sqrt(CRLB[dsigma]) in meV, averaged over components
        condition_number: Fisher matrix condition number
        min_overlap_fwhm: Minimum inter-peak spacing in FWHM units
        n_comp: Number of components
        per_component: Per-component (level, crlb_dE_meV) list
        detail: CRLBResult for full access
    """
    level: SolvabilityLevel
    crlb_dE_meV: float
    crlb_ds_meV: float
    condition_number: float
    min_overlap_fwhm: float
    n_comp: int
    per_component: list[tuple[SolvabilityLevel, float]]
    detail: Optional['CRLBResult'] = field(default=None, repr=False)

    def __str__(self) -> str:
        badge = {
            SolvabilityLevel.EASY: "Easy",
            SolvabilityLevel.HARD: "Hard",
            SolvabilityLevel.SHOULDER: "Shoulder",
            SolvabilityLevel.IMPOSSIBLE: "Impossible",
        }[self.level]
        return (f"[{badge}] n_comp={self.n_comp}, "
                f"CRLB(dE)={self.crlb_dE_meV:.1f} meV, "
                f"cond={self.condition_number:.1e}, "
                f"min_sep={self.min_overlap_fwhm:.2f} FWHM")


def _classify_crlb_dE(crlb_dE_eV: float, sigma: float) -> SolvabilityLevel:
    """Classify solvability from sqrt(CRLB[dE]) relative to sigma."""
    std_dE = np.sqrt(max(crlb_dE_eV, 0))
    ratio = std_dE / sigma if sigma > 0 else np.inf

    if ratio < 0.02:
        return SolvabilityLevel.EASY
    elif ratio < 0.10:
        return SolvabilityLevel.HARD
    elif ratio < 0.40:
        return SolvabilityLevel.SHOULDER
    else:
        return SolvabilityLevel.IMPOSSIBLE


def classify_solvability(
    centers: np.ndarray,
    sigmas: np.ndarray,
    gammas: np.ndarray,
    amplitudes: np.ndarray | None = None,
    energy: np.ndarray | None = None,
    snr: float = 100.0,
    noise_model: Literal['poisson', 'gaussian'] = 'gaussian',
) -> SolvabilityInfo:
    """Classify solvability of a multi-peak configuration.

    Lightweight entry point: given peak positions and widths, returns
    a SolvabilityInfo with level (EASY/HARD/SHOULDER/IMPOSSIBLE),
    CRLB bounds in meV, and condition number.

    Args:
        centers: Peak centers (eV), shape (n_comp,)
        sigmas: Gaussian widths (sigma, eV), shape (n_comp,)
        gammas: Lorentzian half-widths (eV), shape (n_comp,)
        amplitudes: Peak amplitudes, shape (n_comp,). Default: all 1.0
        energy: Energy axis (eV). Default: auto-generated.
        snr: Signal-to-noise ratio (for Gaussian noise model)
        noise_model: 'poisson' or 'gaussian'

    Returns:
        SolvabilityInfo with classification and CRLB bounds
    """
    centers = np.asarray(centers, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    gammas = np.asarray(gammas, dtype=np.float64)
    n_comp = len(centers)

    if amplitudes is None:
        amplitudes = np.ones(n_comp)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)

    # Auto energy axis
    if energy is None:
        fwhm_max = max(_voigt_fwhm(s, g) for s, g in zip(sigmas, gammas))
        e_min = centers.min() - 5 * fwhm_max
        e_max = centers.max() + 5 * fwhm_max
        energy = np.linspace(e_min, e_max, 256)

    # Noise std for Gaussian model
    noise_std = None
    if noise_model == 'gaussian':
        # Estimate composite peak height for noise_std = max(f) / SNR
        from .voigt_jacobian import voigt_profile
        f_max = sum(
            a * voigt_profile(np.array([c]), c, s, g)[0]
            for a, c, s, g in zip(amplitudes, centers, sigmas, gammas)
        )
        noise_std = max(f_max / snr, 1e-30)

    # Compute multi-peak CRLB
    cr = compute_multipeak_fisher(
        amplitudes, centers, sigmas, gammas, energy,
        mode='3d',
        noise_model=noise_model,
        noise_std=noise_std,
    )

    # Per-component classification
    avg_sigma = float(np.mean(sigmas))
    per_component = []
    crlb_dE_list = []
    crlb_ds_list = []

    for k in range(n_comp):
        crlb_dE_k = cr.crlb_per_component[k]['dE']
        crlb_ds_k = cr.crlb_per_component[k]['dsigma']
        level_k = _classify_crlb_dE(crlb_dE_k, sigmas[k])
        std_dE_meV = np.sqrt(max(crlb_dE_k, 0)) * 1e3
        per_component.append((level_k, std_dE_meV))
        crlb_dE_list.append(crlb_dE_k)
        crlb_ds_list.append(crlb_ds_k)

    # Average CRLB
    avg_crlb_dE_meV = float(np.mean([np.sqrt(max(v, 0)) for v in crlb_dE_list])) * 1e3
    avg_crlb_ds_meV = float(np.mean([np.sqrt(max(v, 0)) for v in crlb_ds_list])) * 1e3

    # Overall level = worst component
    level_order = [SolvabilityLevel.EASY, SolvabilityLevel.HARD,
                   SolvabilityLevel.SHOULDER, SolvabilityLevel.IMPOSSIBLE]
    overall_level = max(
        (lev for lev, _ in per_component),
        key=lambda x: level_order.index(x),
    )

    # Min overlap in FWHM units
    if n_comp >= 2:
        sorted_centers = np.sort(centers)
        spacings = np.diff(sorted_centers)
        fwhms = [_voigt_fwhm(s, g) for s, g in zip(sigmas, gammas)]
        avg_fwhm = float(np.mean(fwhms))
        min_overlap_fwhm = float(spacings.min() / avg_fwhm) if avg_fwhm > 0 else 0.0
    else:
        min_overlap_fwhm = float('inf')

    return SolvabilityInfo(
        level=overall_level,
        crlb_dE_meV=avg_crlb_dE_meV,
        crlb_ds_meV=avg_crlb_ds_meV,
        condition_number=cr.condition_number,
        min_overlap_fwhm=min_overlap_fwhm,
        n_comp=n_comp,
        per_component=per_component,
        detail=cr,
    )


@dataclass
class CRLBResult:
    """Result of multi-peak CRLB computation.

    Attributes:
        fisher: Fisher Information Matrix (n_params_total, n_params_total)
        crlb: CRLB per parameter = diag(inv(fisher)), shape (n_params_total,)
        crlb_per_component: Nested dict {comp_idx: {param_name: crlb_value}}
        condition_number: Condition number of Fisher matrix
        eigenvalues: Eigenvalues in ascending order
        eigenvectors: Corresponding eigenvectors (columns)
        param_names: Flat list of parameter names (e.g. ['amp_0', 'dE_0', ...])
        n_comp: Number of components
        n_params_per_comp: Parameters per component (3 or 4)
        correlation: Correlation matrix r_ij = g_ij / sqrt(g_ii * g_jj)
        config: Dictionary of configuration used
    """
    fisher: np.ndarray
    crlb: np.ndarray
    crlb_per_component: dict[int, dict[str, float]]
    condition_number: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    param_names: list[str]
    n_comp: int
    n_params_per_comp: int
    correlation: np.ndarray
    config: dict[str, Any] = field(default_factory=dict)


# Parameter name templates
_PARAM_NAMES_3D = ('amp', 'dE', 'dsigma')
_PARAM_NAMES_4D = ('amp', 'dE', 'dsigma', 'dgamma')


def compute_multipeak_fisher(
    amplitudes: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gammas: np.ndarray,
    energy: np.ndarray,
    mode: Literal['3d', '4d'] = '3d',
    floor_eps: float = 1e-30,
    noise_model: Literal['poisson', 'gaussian'] = 'poisson',
    noise_std: float | None = None,
) -> CRLBResult:
    """Compute Fisher Information Matrix and CRLB for multi-peak Voigt spectrum.

    For n_comp peaks, builds the full (n_comp * n_params_per) x (n_comp * n_params_per)
    Fisher matrix including inter-peak coupling.

    Noise models:
        - 'poisson': g_ij = sum_E (df/dθ_i)(df/dθ_j) / f(E)
          Physically correct for photon counting. Weight = 1/f(E).
        - 'gaussian': g_ij = (1/σ²) sum_E (df/dθ_i)(df/dθ_j)
          Matches bench_multipeak_sweep convention. Requires noise_std.

    Args:
        amplitudes: Peak amplitudes, shape (n_comp,)
        centers: Peak centers (eV), shape (n_comp,)
        sigmas: Gaussian widths (sigma, eV), shape (n_comp,)
        gammas: Lorentzian half-widths (eV), shape (n_comp,)
        energy: Energy axis (eV), shape (n_energy,)
        mode: '3d' for (amp, dE, dsigma) per comp, '4d' adds dgamma
        floor_eps: Floor for Poisson denominator to avoid 1/0
        noise_model: 'poisson' or 'gaussian'
        noise_std: Standard deviation of Gaussian noise (required if noise_model='gaussian')

    Returns:
        CRLBResult with Fisher matrix, CRLB, and diagnostics
    """
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    gammas = np.asarray(gammas, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)

    n_comp = len(amplitudes)
    n_energy = len(energy)
    base_names = _PARAM_NAMES_4D if mode == '4d' else _PARAM_NAMES_3D
    n_per = len(base_names)
    n_total = n_comp * n_per

    # --- Step 1: Compute per-peak Voigt profiles and Jacobians ---
    profiles = []  # V_k(E)
    jacobians_per_peak = []  # list of (n_per, n_energy) arrays

    for k in range(n_comp):
        V_k, dV_dc_k, dV_dsigma_k, dV_dgamma_k = voigt_with_jacobian(
            energy, centers[k], sigmas[k], gammas[k]
        )
        profiles.append(V_k)

        # Build Jacobian rows for peak k:
        #   df/d(amp_k) = V_k
        #   df/d(dE_k) = A_k * dV_dc_k
        #   df/d(dsigma_k) = A_k * dV_dsigma_k
        #   df/d(dgamma_k) = A_k * dV_dgamma_k  (4d only)
        A_k = amplitudes[k]
        if mode == '3d':
            J_k = np.stack([V_k, A_k * dV_dc_k, A_k * dV_dsigma_k])
        else:
            J_k = np.stack([V_k, A_k * dV_dc_k, A_k * dV_dsigma_k, A_k * dV_dgamma_k])
        jacobians_per_peak.append(J_k)

    # --- Step 2: Composite spectrum (Poisson mean) ---
    f_composite = np.zeros(n_energy, dtype=np.float64)
    for k in range(n_comp):
        f_composite += amplitudes[k] * profiles[k]
    f_safe = np.maximum(f_composite, floor_eps)

    # --- Step 3: Full Jacobian matrix (n_total, n_energy) ---
    J_full = np.zeros((n_total, n_energy), dtype=np.float64)
    for k in range(n_comp):
        i0 = k * n_per
        J_full[i0:i0 + n_per, :] = jacobians_per_peak[k]

    # --- Step 4: Fisher matrix ---
    # Poisson: g_ij = sum_E J_i J_j / f(E)
    # Gaussian: g_ij = (1/sigma^2) sum_E J_i J_j
    if noise_model == 'poisson':
        sqrt_w = np.sqrt(1.0 / f_safe)  # (n_energy,)
    elif noise_model == 'gaussian':
        if noise_std is None or noise_std <= 0:
            raise ValueError("noise_std must be positive for Gaussian noise model")
        sqrt_w = np.full(n_energy, 1.0 / noise_std)
    else:
        raise ValueError(f"noise_model must be 'poisson' or 'gaussian', got '{noise_model}'")

    Jw = J_full * sqrt_w[np.newaxis, :]  # (n_total, n_energy)
    fisher = Jw @ Jw.T  # (n_total, n_total)

    # --- Step 5: CRLB = diag(inv(fisher)) ---
    # Use pseudo-inverse for near-singular cases
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    eig_positive = eigenvalues[eigenvalues > 0]

    if len(eig_positive) >= 2:
        condition_number = float(eig_positive[-1] / eig_positive[0])
    else:
        condition_number = np.inf

    # Regularized inverse for CRLB
    # Threshold at machine epsilon * max eigenvalue
    eig_threshold = max(eigenvalues[-1] * 1e-12, 1e-30)
    eig_inv = np.where(eigenvalues > eig_threshold, 1.0 / eigenvalues, 0.0)
    fisher_inv = (eigenvectors * eig_inv[np.newaxis, :]) @ eigenvectors.T
    crlb = np.diag(fisher_inv)

    # --- Step 6: Per-component CRLB and parameter names ---
    param_names = []
    crlb_per_component = {}
    for k in range(n_comp):
        comp_dict = {}
        for p, name in enumerate(base_names):
            full_name = f"{name}_{k}"
            param_names.append(full_name)
            comp_dict[name] = float(crlb[k * n_per + p])
        crlb_per_component[k] = comp_dict

    # --- Step 7: Correlation matrix ---
    diag_sqrt = np.sqrt(np.maximum(np.diag(fisher), 1e-30))
    correlation = fisher / np.outer(diag_sqrt, diag_sqrt)

    return CRLBResult(
        fisher=fisher,
        crlb=crlb,
        crlb_per_component=crlb_per_component,
        condition_number=condition_number,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        param_names=param_names,
        n_comp=n_comp,
        n_params_per_comp=n_per,
        correlation=correlation,
        config={
            'amplitudes': amplitudes.tolist(),
            'centers': centers.tolist(),
            'sigmas': sigmas.tolist(),
            'gammas': gammas.tolist(),
            'n_energy': n_energy,
            'energy_range': (float(energy[0]), float(energy[-1])),
            'mode': mode,
            'noise_model': noise_model,
            'noise_std': noise_std,
        },
    )


def _voigt_fwhm(sigma: float, gamma: float) -> float:
    """Thompson (1987) approximation for Voigt FWHM."""
    f_G = sigma / FWHM_TO_SIGMA
    f_L = 2 * gamma
    return 0.5346 * f_L + np.sqrt(0.2166 * f_L**2 + f_G**2)


def _build_equal_spacing_centers(
    n_comp: int,
    overlap_ratio: float,
    sigma: float,
    gamma: float,
) -> np.ndarray:
    """Build equally-spaced peak centers for given overlap ratio.

    Args:
        n_comp: Number of peaks
        overlap_ratio: delta_center / FWHM (1.0 = FWHM spacing)
        sigma: Gaussian width
        gamma: Lorentzian half-width

    Returns:
        centers array, shape (n_comp,), symmetric around 0
    """
    fwhm = _voigt_fwhm(sigma, gamma)
    spacing = overlap_ratio * fwhm
    # Center the array around 0
    total_span = (n_comp - 1) * spacing
    return np.linspace(-total_span / 2, total_span / 2, n_comp)


def crlb_grid_sweep(
    n_comp_list: list[int],
    overlap_ratios: np.ndarray,
    snr_levels: np.ndarray,
    sigma: float = 0.5,
    gamma: float = 0.3,
    n_energy: int = 256,
    energy_padding: float = 5.0,
    mode: Literal['3d', '4d'] = '3d',
) -> dict[str, Any]:
    """Sweep CRLB over (n_comp, overlap, SNR) grid.

    Generates equal-spacing, equal-amplitude configurations and computes
    CRLB at each grid point.

    SNR is defined as: amplitude = SNR^2 * V_max (so that peak counts ~ SNR^2).
    This gives sqrt(CRLB[amp]) / amp ~ 1/SNR for well-separated peaks.

    Args:
        n_comp_list: List of component counts to sweep
        overlap_ratios: delta_center/FWHM values, shape (n_overlap,)
        snr_levels: SNR values, shape (n_snr,)
        sigma: Gaussian width for all peaks (eV)
        gamma: Lorentzian half-width for all peaks (eV)
        n_energy: Number of energy points
        energy_padding: Extra energy range beyond outermost peaks (in FWHM units)
        mode: '3d' or '4d'

    Returns:
        Dictionary with:
            'crlb_amp': {n_comp: array (n_overlap, n_snr)}
            'crlb_dE': {n_comp: array (n_overlap, n_snr)}
            'crlb_dsigma': {n_comp: array (n_overlap, n_snr)}
            'crlb_dgamma': {n_comp: array (n_overlap, n_snr)} (4d only)
            'condition_number': {n_comp: array (n_overlap, n_snr)}
            'overlap_ratios': overlap_ratios
            'snr_levels': snr_levels
            'n_comp_list': n_comp_list
            'sigma': sigma, 'gamma': gamma
    """
    overlap_ratios = np.asarray(overlap_ratios)
    snr_levels = np.asarray(snr_levels)
    fwhm = _voigt_fwhm(sigma, gamma)

    base_names = _PARAM_NAMES_4D if mode == '4d' else _PARAM_NAMES_3D
    n_per = len(base_names)

    result = {
        f'crlb_{name}': {} for name in base_names
    }
    result['condition_number'] = {}
    result['overlap_ratios'] = overlap_ratios
    result['snr_levels'] = snr_levels
    result['n_comp_list'] = n_comp_list
    result['sigma'] = sigma
    result['gamma'] = gamma
    result['fwhm'] = fwhm
    result['mode'] = mode

    for nc in n_comp_list:
        n_overlap = len(overlap_ratios)
        n_snr = len(snr_levels)

        # Allocate per-parameter CRLB arrays (average over components)
        crlb_arrays = {name: np.full((n_overlap, n_snr), np.nan) for name in base_names}
        cond_array = np.full((n_overlap, n_snr), np.nan)

        for i_ov, ov in enumerate(overlap_ratios):
            centers = _build_equal_spacing_centers(nc, ov, sigma, gamma)

            # Energy axis: cover all peaks + padding
            e_min = centers[0] - energy_padding * fwhm
            e_max = centers[-1] + energy_padding * fwhm
            energy = np.linspace(e_min, e_max, n_energy)

            for i_snr, snr in enumerate(snr_levels):
                # Amplitude from SNR: peak counts ~ amplitude * V_max
                # V_max ~ 1/(sigma*sqrt(2*pi)) for narrow Lorentzian
                # SNR = sqrt(peak_counts) => amplitude = snr^2 / V_max
                # Simplified: use amplitude = snr^2 directly (normalized V)
                from .voigt_jacobian import voigt_profile
                V_max = voigt_profile(
                    np.array([centers[0]]), centers[0], sigma, gamma
                )[0]
                amplitude = snr**2 / V_max if V_max > 1e-30 else snr**2

                amps = np.full(nc, amplitude)
                sigs = np.full(nc, sigma)
                gams = np.full(nc, gamma)

                try:
                    cr = compute_multipeak_fisher(
                        amps, centers, sigs, gams, energy, mode=mode
                    )
                    cond_array[i_ov, i_snr] = cr.condition_number

                    # Average CRLB across components (symmetric config → all equal)
                    for p, name in enumerate(base_names):
                        vals = [cr.crlb[k * n_per + p] for k in range(nc)]
                        crlb_arrays[name][i_ov, i_snr] = np.mean(vals)

                except np.linalg.LinAlgError:
                    # Singular Fisher matrix
                    pass

        for name in base_names:
            result[f'crlb_{name}'][nc] = crlb_arrays[name]
        result['condition_number'][nc] = cond_array

    return result


def print_crlb_summary(cr: CRLBResult, title: str = "") -> None:
    """Print formatted CRLB summary."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    print(f"  n_comp={cr.n_comp}, mode={'4d' if cr.n_params_per_comp==4 else '3d'}")
    print(f"  Condition number: {cr.condition_number:.2e}")
    print(f"  Eigenvalue range: [{cr.eigenvalues[0]:.2e}, {cr.eigenvalues[-1]:.2e}]")
    print()

    # Per-component CRLB (as standard deviation = sqrt(CRLB))
    header = "  Comp |" + "".join(f" sqrt(CRLB[{n}]) |" for n in _PARAM_NAMES_3D[:cr.n_params_per_comp])
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k in range(cr.n_comp):
        vals = cr.crlb_per_component[k]
        line = f"  {k:4d} |"
        for name in list(vals.keys()):
            v = vals[name]
            line += f" {np.sqrt(max(v, 0)):13.4e} |"
        print(line)
    print()


# ===================================================================
# N-component synthetic spectra generation + empirical RMSE
# ===================================================================


def generate_ncomp_spectra(
    n_spectra: int,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gammas: np.ndarray,
    amplitudes: np.ndarray,
    energy: np.ndarray,
    snr: float,
    dE_range: float = 0.3,
    ds_range: float = 0.08,
    seed: int = 42,
    peak_normalize: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate n-component Voigt spectra with controlled noise.

    Each spectrum has per-component random jitter in (dE, dsigma).
    Noise model matches bench_multipeak_sweep: Gaussian with std = max(Y)/SNR.

    Args:
        n_spectra: Number of spectra to generate
        centers: Nominal peak centers (eV), shape (n_comp,)
        sigmas: Gaussian widths (sigma, eV), shape (n_comp,)
        gammas: Lorentzian half-widths (eV), shape (n_comp,)
        amplitudes: Peak amplitudes, shape (n_comp,)
        energy: Energy axis (eV), shape (n_energy,)
        snr: Signal-to-noise ratio (max_signal / noise_std). inf = noiseless.
        dE_range: Random center shift range (uniform ±dE_range)
        ds_range: Random sigma shift range (uniform ±ds_range)
        seed: RNG seed for reproducibility
        peak_normalize: If True, normalize each Voigt profile to peak=1
            before scaling by amplitude (matches benchmark convention)

    Returns:
        Y: (n_spectra, n_energy) float32 spectra
        gt: Ground truth dict with:
            'dE': (n_spectra, n_comp) center shifts
            'ds': (n_spectra, n_comp) sigma shifts
            'amp': (n_spectra, n_comp) amplitudes (constant per spectrum)
    """
    from scipy.special import wofz

    rng = np.random.default_rng(seed)
    n_comp = len(centers)
    n_energy = len(energy)
    energy_f64 = np.asarray(energy, dtype=np.float64)

    SQRT2 = np.sqrt(2.0)
    SQRT2PI = np.sqrt(2.0 * np.pi)

    # Ground truth: random jitter per spectrum per component
    gt_dE = rng.uniform(-dE_range, dE_range, (n_spectra, n_comp)).astype(np.float32)
    gt_ds = rng.uniform(-ds_range, ds_range, (n_spectra, n_comp)).astype(np.float32)

    # Build spectra
    Y = np.zeros((n_spectra, n_energy), dtype=np.float64)

    for k in range(n_comp):
        c_k = (centers[k] + gt_dE[:, k]).astype(np.float64)[:, None]  # (N, 1)
        s_k = np.maximum(sigmas[k] + gt_ds[:, k], 0.01).astype(np.float64)[:, None]
        z_k = (energy_f64[None, :] - c_k + 1j * gammas[k]) / (s_k * SQRT2)
        P_k = np.real(wofz(z_k)) / (s_k * SQRT2PI)

        if peak_normalize:
            P_k /= P_k.max(axis=1, keepdims=True) + 1e-30

        Y += amplitudes[k] * P_k

    Y = Y.astype(np.float32)

    # Gaussian noise (signal-relative)
    if np.isfinite(snr) and snr > 0:
        signal_max = float(Y.max())
        noise_std = signal_max / snr
        Y += rng.normal(0, noise_std, Y.shape).astype(np.float32)

    gt = {
        'dE': gt_dE,
        'ds': gt_ds,
        'amp': np.broadcast_to(
            np.asarray(amplitudes, dtype=np.float32)[None, :],
            (n_spectra, n_comp),
        ).copy(),
    }
    return Y, gt


def _correct_swaps_ncomp(
    result_dE: np.ndarray,
    result_ds: np.ndarray,
    result_amp: np.ndarray,
    gt: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-spectrum greedy swap correction for n-component assignment.

    For n_comp <= 4, tries all permutations. For n_comp > 4, uses
    greedy nearest-neighbor assignment on dE.

    Returns corrected (dE, ds, amp) and number of swapped spectra.
    """
    from itertools import permutations

    n_spectra, n_comp = result_dE.shape
    dE_corr = result_dE.copy()
    ds_corr = result_ds.copy()
    amp_corr = result_amp.copy()
    n_swapped = 0

    gt_dE = gt['dE']

    if n_comp <= 4:
        # Exhaustive permutation search
        perms = list(permutations(range(n_comp)))
        for i in range(n_spectra):
            best_err = np.sum((result_dE[i] - gt_dE[i]) ** 2)
            best_perm = None
            for perm in perms:
                perm_arr = list(perm)
                err = np.sum((result_dE[i, perm_arr] - gt_dE[i]) ** 2)
                if err < best_err:
                    best_err = err
                    best_perm = perm_arr
            if best_perm is not None:
                dE_corr[i] = result_dE[i, best_perm]
                ds_corr[i] = result_ds[i, best_perm]
                amp_corr[i] = result_amp[i, best_perm]
                n_swapped += 1
    else:
        # Greedy nearest-neighbor for large n_comp
        for i in range(n_spectra):
            assigned = set()
            perm = [-1] * n_comp
            # Sort gt components by center for stable assignment
            for k in range(n_comp):
                # Find closest unassigned result component
                best_j = -1
                best_dist = np.inf
                for j in range(n_comp):
                    if j in assigned:
                        continue
                    dist = abs(result_dE[i, j] - gt_dE[i, k])
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
                assigned.add(best_j)
                perm[k] = best_j
            if perm != list(range(n_comp)):
                dE_corr[i] = result_dE[i, perm]
                ds_corr[i] = result_ds[i, perm]
                amp_corr[i] = result_amp[i, perm]
                n_swapped += 1

    return dE_corr, ds_corr, amp_corr, n_swapped


@dataclass
class EfficiencyResult:
    """Result of CRLB vs empirical RMSE comparison at one grid point.

    Attributes:
        n_comp: Number of peaks
        overlap_ratio: Δcenter / FWHM
        snr: Signal-to-noise ratio
        crlb: CRLBResult (theoretical)
        rmse_dE: Empirical RMSE for center shift (eV), averaged over components
        rmse_ds: Empirical RMSE for sigma shift (eV), averaged over components
        rmse_amp: Empirical RMSE for amplitude, averaged over components
        efficiency_dE: CRLB[dE] / RMSE²[dE] (1.0 = theoretical limit)
        efficiency_ds: CRLB[ds] / RMSE²[ds]
        efficiency_amp: CRLB[amp] / RMSE²[amp]
        n_swapped: Number of spectra with component swap correction
        solver_time: Solver wall time (seconds)
    """
    n_comp: int
    overlap_ratio: float
    snr: float
    crlb: CRLBResult
    rmse_dE: float
    rmse_ds: float
    rmse_amp: float
    efficiency_dE: float
    efficiency_ds: float
    efficiency_amp: float
    n_swapped: int
    solver_time: float


def compute_efficiency_point(
    n_comp: int,
    overlap_ratio: float,
    snr: float,
    sigma: float = 0.5,
    gamma: float = 0.3,
    n_energy: int = 256,
    n_spectra: int = 10_000,
    energy_padding: float = 5.0,
    dE_range: float = 0.3,
    ds_range: float = 0.08,
    seed: int = 42,
    solver_kwargs: dict | None = None,
) -> EfficiencyResult:
    """Compute CRLB efficiency at a single (n_comp, overlap, SNR) point.

    1. Compute theoretical CRLB (Gaussian noise model)
    2. Generate synthetic spectra with controlled noise
    3. Run multipeak solver
    4. Compare empirical RMSE with theoretical CRLB

    Args:
        n_comp: Number of peaks
        overlap_ratio: Δcenter / FWHM
        snr: Signal-to-noise ratio (inf = noiseless)
        sigma, gamma: Voigt parameters (eV)
        n_energy: Number of energy points
        n_spectra: Number of spectra for RMSE estimation
        energy_padding: Energy padding in FWHM units
        dE_range, ds_range: Parameter jitter ranges
        seed: RNG seed
        solver_kwargs: Extra kwargs for process_multipeak

    Returns:
        EfficiencyResult with CRLB, RMSE, and efficiency
    """
    import time as _time
    import warnings

    from .multipeak_config import ComponentConfig, MultiPeakConfig
    from .multipeak_solver import process_multipeak

    fwhm = _voigt_fwhm(sigma, gamma)
    centers = _build_equal_spacing_centers(n_comp, overlap_ratio, sigma, gamma)

    # Energy axis
    e_min = centers[0] - energy_padding * fwhm
    e_max = centers[-1] + energy_padding * fwhm
    energy = np.linspace(e_min, e_max, n_energy, dtype=np.float32)

    # Amplitude: use 1.0 (peak-normalized profiles)
    amp_val = 1.0
    amps = np.full(n_comp, amp_val)

    # --- Step 1: Generate spectra ---
    Y, gt = generate_ncomp_spectra(
        n_spectra=n_spectra,
        centers=centers,
        sigmas=np.full(n_comp, sigma),
        gammas=np.full(n_comp, gamma),
        amplitudes=amps,
        energy=energy,
        snr=snr,
        dE_range=dE_range,
        ds_range=ds_range,
        seed=seed,
    )

    # --- Step 2: Theoretical CRLB (Gaussian noise, matching benchmark) ---
    if np.isfinite(snr) and snr > 0:
        noise_std_val = float(Y.max()) / snr
    else:
        noise_std_val = 1e-10  # Near-noiseless

    # Use nominal (un-jittered) parameters for CRLB
    cr = compute_multipeak_fisher(
        amplitudes=amps,
        centers=centers,
        sigmas=np.full(n_comp, sigma),
        gammas=np.full(n_comp, gamma),
        energy=energy.astype(np.float64),
        mode='3d',
        noise_model='gaussian',
        noise_std=noise_std_val,
    )

    # --- Step 3: Run solver ---
    configs = []
    for k in range(n_comp):
        configs.append(ComponentConfig(
            center=float(centers[k]),
            sigma=sigma,
            gamma=gamma,
            dE_range=max(dE_range * 2, 0.5),
            ds_range=max(ds_range * 2, 0.2),
            n_dE=10,
            n_ds=31,
        ))
    mpc = MultiPeakConfig(peaks=configs, energy_axis=energy)

    skw = dict(
        n_iterations=3,
        parabola_dE=True,
        parabola_ds=False,
        auto_constrain=True,
    )
    if solver_kwargs:
        skw.update(solver_kwargs)

    t0 = _time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = process_multipeak(Y, mpc, **skw)
    solver_time = _time.time() - t0

    # --- Step 4: Compute empirical RMSE ---
    dE_corr, ds_corr, amp_corr, n_swapped = _correct_swaps_ncomp(
        result.delta_E, result.delta_sigma, result.amplitudes, gt,
    )

    # Per-component RMSE, then average
    rmse_dE_per = np.sqrt(np.mean((dE_corr - gt['dE']) ** 2, axis=0))
    rmse_ds_per = np.sqrt(np.mean((ds_corr - gt['ds']) ** 2, axis=0))
    rmse_amp_per = np.sqrt(np.mean((amp_corr - gt['amp']) ** 2, axis=0))

    rmse_dE = float(np.mean(rmse_dE_per))
    rmse_ds = float(np.mean(rmse_ds_per))
    rmse_amp = float(np.mean(rmse_amp_per))

    # --- Step 5: Efficiency = CRLB / RMSE² ---
    # Average CRLB over components
    avg_crlb_dE = np.mean([cr.crlb_per_component[k]['dE'] for k in range(n_comp)])
    avg_crlb_ds = np.mean([cr.crlb_per_component[k]['dsigma'] for k in range(n_comp)])
    avg_crlb_amp = np.mean([cr.crlb_per_component[k]['amp'] for k in range(n_comp)])

    eff_dE = float(avg_crlb_dE / max(rmse_dE ** 2, 1e-30))
    eff_ds = float(avg_crlb_ds / max(rmse_ds ** 2, 1e-30))
    eff_amp = float(avg_crlb_amp / max(rmse_amp ** 2, 1e-30))

    return EfficiencyResult(
        n_comp=n_comp,
        overlap_ratio=overlap_ratio,
        snr=snr,
        crlb=cr,
        rmse_dE=rmse_dE,
        rmse_ds=rmse_ds,
        rmse_amp=rmse_amp,
        efficiency_dE=eff_dE,
        efficiency_ds=eff_ds,
        efficiency_amp=eff_amp,
        n_swapped=n_swapped,
        solver_time=solver_time,
    )


def efficiency_sweep(
    n_comp_list: list[int],
    overlap_ratios: np.ndarray,
    snr_levels: np.ndarray,
    sigma: float = 0.5,
    gamma: float = 0.3,
    n_energy: int = 256,
    n_spectra: int = 10_000,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Sweep efficiency over (n_comp, overlap, SNR) grid.

    Returns:
        Dictionary with:
            'efficiency_dE': {n_comp: array (n_overlap, n_snr)}
            'efficiency_ds': {n_comp: ...}
            'rmse_dE': {n_comp: ...}
            'rmse_ds': {n_comp: ...}
            'crlb_dE': {n_comp: ...}  (sqrt(CRLB), theoretical std)
            'crlb_ds': {n_comp: ...}
            'overlap_ratios', 'snr_levels', 'n_comp_list'
    """
    import time as _time

    overlap_ratios = np.asarray(overlap_ratios)
    snr_levels = np.asarray(snr_levels)

    keys = ['efficiency_dE', 'efficiency_ds', 'rmse_dE', 'rmse_ds',
            'crlb_dE', 'crlb_ds', 'solver_time']
    result = {k: {} for k in keys}
    result['overlap_ratios'] = overlap_ratios
    result['snr_levels'] = snr_levels
    result['n_comp_list'] = n_comp_list

    total = len(n_comp_list) * len(overlap_ratios) * len(snr_levels)
    done = 0
    t_start = _time.time()

    for nc in n_comp_list:
        shape = (len(overlap_ratios), len(snr_levels))
        arrays = {k: np.full(shape, np.nan) for k in keys}

        for i_ov, ov in enumerate(overlap_ratios):
            for i_snr, snr in enumerate(snr_levels):
                done += 1
                try:
                    er = compute_efficiency_point(
                        n_comp=nc,
                        overlap_ratio=ov,
                        snr=snr,
                        sigma=sigma,
                        gamma=gamma,
                        n_energy=n_energy,
                        n_spectra=n_spectra,
                        seed=seed + done,  # different seed per cell
                    )
                    arrays['efficiency_dE'][i_ov, i_snr] = er.efficiency_dE
                    arrays['efficiency_ds'][i_ov, i_snr] = er.efficiency_ds
                    arrays['rmse_dE'][i_ov, i_snr] = er.rmse_dE
                    arrays['rmse_ds'][i_ov, i_snr] = er.rmse_ds
                    arrays['crlb_dE'][i_ov, i_snr] = np.sqrt(
                        np.mean([er.crlb.crlb_per_component[k]['dE'] for k in range(nc)])
                    )
                    arrays['crlb_ds'][i_ov, i_snr] = np.sqrt(
                        np.mean([er.crlb.crlb_per_component[k]['dsigma'] for k in range(nc)])
                    )
                    arrays['solver_time'][i_ov, i_snr] = er.solver_time

                    if verbose and done % 10 == 0:
                        elapsed = _time.time() - t_start
                        eta = elapsed / done * (total - done)
                        print(f"  [{done}/{total}] nc={nc} ov={ov:.1f} snr={snr:.0f} "
                              f"eff_dE={er.efficiency_dE:.3f} rmse_dE={er.rmse_dE*1e3:.1f}meV "
                              f"({elapsed:.0f}s, ETA {eta:.0f}s)")
                except Exception as e:
                    if verbose:
                        print(f"  [{done}/{total}] nc={nc} ov={ov:.1f} snr={snr:.0f} FAILED: {e}")

        for k in keys:
            result[k][nc] = arrays[k]

    return result
