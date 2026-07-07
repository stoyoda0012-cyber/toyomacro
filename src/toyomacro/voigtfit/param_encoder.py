"""
Spectral Parameter Encoders
=============================

Encode/decode Voigt spectral parameters to/from RGB channels for
roundtrip verification of the fitting pipeline.

SinglePeakEncoder:
  R = amplitude, G = center shift, B = FWHM_G
  Linear [0,255] <-> [min, max] mapping, 8 bits per param.

MultiPeakEncoder:
  2 peaks x 3 params (amp, delta_E, delta_sigma) = 6D
  Dual 3D Hilbert curves at order p (default 4, 16 levels per dim).
  Each peak -> 3D coords -> 12-bit Hilbert index.
  Pack: (peak1_idx << 12) | peak2_idx = 24 bits -> R G B.
  Locality-preserving: nearby parameters -> nearby RGB values.
"""

from dataclasses import dataclass, field

import numpy as np

from .hilbert import coords_to_index, index_to_coords
from .spectra_generator import ElementSpec

FWHM_TO_SIGMA = 1.0 / (2 * np.sqrt(2 * np.log(2)))  # FWHM -> sigma


@dataclass
class ParamRange:
    """Range specification for a single parameter."""
    min_val: float
    max_val: float
    name: str


@dataclass
class SinglePeakPreset:
    """Preset for single-peak 3-parameter roundtrip.

    Defines the element, energy axis, and parameter ranges for
    R (amplitude), G (center shift), B (Gaussian FWHM) encoding.
    """
    name: str
    element: ElementSpec
    energy_range: tuple[float, float]  # (start, end) eV
    n_energy: int                       # number of energy points
    amplitude_range: ParamRange         # R channel
    shift_range: ParamRange             # G channel (delta_E, eV)
    fwhm_range: ParamRange              # B channel (Gaussian FWHM, eV)

    @property
    def energy(self) -> np.ndarray:
        return np.linspace(*self.energy_range, self.n_energy).astype(np.float32)


class SinglePeakEncoder:
    """Encode/decode 3 spectral parameters to/from RGB image.

    R = amplitude [amp_min, amp_max]
    G = center shift delta_E [shift_min, shift_max]
    B = Gaussian FWHM [fwhm_min, fwhm_max]
    """

    def __init__(self, preset: SinglePeakPreset):
        self.preset = preset

    def encode(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RGB image (H,W,3) uint8 -> (amplitude, delta_E, fwhm_g) each (H*W,) float32.

        Maps each channel linearly from [0, 255] to the parameter range.
        """
        H, W = image.shape[:2]
        img = image.reshape(H * W, 3).astype(np.float32)

        amp = self._channel_to_param(img[:, 0], self.preset.amplitude_range)
        dE = self._channel_to_param(img[:, 1], self.preset.shift_range)
        fwhm_g = self._channel_to_param(img[:, 2], self.preset.fwhm_range)

        return amp, dE, fwhm_g

    def decode(
        self,
        amplitude: np.ndarray,
        delta_E: np.ndarray,
        fwhm_g: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """(amplitude, delta_E, fwhm_g) each (N,) -> RGB image (H,W,3) uint8."""
        H, W = image_shape
        r = self._param_to_channel(amplitude, self.preset.amplitude_range)
        g = self._param_to_channel(delta_E, self.preset.shift_range)
        b = self._param_to_channel(fwhm_g, self.preset.fwhm_range)
        rgb = np.stack([r, g, b], axis=1).reshape(H, W, 3)
        return rgb

    def delta_sigma(self, fwhm_g: np.ndarray) -> np.ndarray:
        """Compute delta_sigma = sigma(fwhm_g) - sigma_nominal."""
        sigma_nom = self.preset.element.sigma
        sigma = fwhm_g * FWHM_TO_SIGMA
        return (sigma - sigma_nom).astype(np.float32)

    def fwhm_from_sigma(self, sigma: np.ndarray) -> np.ndarray:
        """Convert sigma to Gaussian FWHM."""
        return sigma / FWHM_TO_SIGMA

    @staticmethod
    def _channel_to_param(channel: np.ndarray, prange: ParamRange) -> np.ndarray:
        """Map [0, 255] -> [min_val, max_val]."""
        t = channel / 255.0
        return (prange.min_val + t * (prange.max_val - prange.min_val)).astype(np.float32)

    @staticmethod
    def _param_to_channel(param: np.ndarray, prange: ParamRange) -> np.ndarray:
        """Map [min_val, max_val] -> [0, 255] uint8."""
        t = (param - prange.min_val) / (prange.max_val - prange.min_val)
        return np.clip(t * 255, 0, 255).astype(np.uint8)


# Built-in preset: C 1s single peak
C1S_SINGLE_PRESET = SinglePeakPreset(
    name='C1s_single',
    element=ElementSpec('C', '1s', 284.4, gaussian_fwhm=1.0, lorentzian_fwhm=0.25),
    energy_range=(280.0, 290.0),
    n_energy=101,
    amplitude_range=ParamRange(0.0, 1.0, 'amplitude'),
    shift_range=ParamRange(-1.0, 1.0, 'delta_E'),
    fwhm_range=ParamRange(0.7, 1.5, 'fwhm_gaussian'),
)


# ---------------------------------------------------------------------------
# Multi-Peak Hilbert Encoder
# ---------------------------------------------------------------------------

@dataclass
class MultiPeakPreset:
    """Preset for multi-peak Hilbert-encoded roundtrip.

    Defines elements, energy axis, Hilbert order, and per-parameter ranges
    shared across all peaks. Uses 3D Hilbert curves (one per peak) to map
    (amplitude, delta_E, delta_sigma) to a locality-preserving 1D index.

    For n_peaks=2, hilbert_order=4:
      Each peak: 3D coords in [0,16)^3 -> 12-bit Hilbert index
      Pack: (idx_0 << 12) | idx_1 = 24 bits -> RGB
    """
    name: str
    elements: list[ElementSpec]
    energy_range: tuple[float, float]
    n_energy: int
    hilbert_order: int = 4  # 2^4 = 16 levels per dimension
    # Shared parameter ranges (same for all peaks)
    amplitude_range: ParamRange = field(
        default_factory=lambda: ParamRange(0.0, 1.0, 'amplitude'))
    shift_range: ParamRange = field(
        default_factory=lambda: ParamRange(-0.5, 0.5, 'delta_E'))
    sigma_range: ParamRange = field(
        default_factory=lambda: ParamRange(-0.1, 0.1, 'delta_sigma'))

    @property
    def n_peaks(self) -> int:
        return len(self.elements)

    @property
    def energy(self) -> np.ndarray:
        return np.linspace(*self.energy_range, self.n_energy).astype(np.float32)

    @property
    def n_levels(self) -> int:
        """Number of quantization levels per dimension."""
        return 1 << self.hilbert_order

    @property
    def bits_per_peak(self) -> int:
        """Bits per peak's Hilbert index (3 * hilbert_order)."""
        return 3 * self.hilbert_order


class MultiPeakEncoder:
    """Encode/decode N-peak x 3-param parameters via dual 3D Hilbert curves.

    For 2 peaks (default):
      Peak k: (amp_k, dE_k, dsigma_k) -> quantize to [0, 2^p)^3
        -> 3D Hilbert index (3p bits)
      Pack: (idx_0 << 3p) | idx_1 = 6p bits -> RGB (24 bits for p=4)

    The Hilbert curve preserves spatial locality: pixels with similar
    physical parameters map to similar RGB values, enabling visual
    inspection and error-map analysis.

    Supports 1 or 2 peaks. For 1 peak with order=8, uses full 24-bit
    precision (256 levels per dimension).
    """

    def __init__(self, preset: MultiPeakPreset):
        self.preset = preset
        total_bits = preset.n_peaks * preset.bits_per_peak
        if total_bits > 24:
            raise ValueError(
                f"Total bits ({total_bits}) exceeds 24-bit RGB capacity. "
                f"Reduce hilbert_order or n_peaks.")

    def encode(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RGB image (H,W,3) uint8 -> per-peak parameter arrays.

        Returns:
            amplitudes: (N, n_peaks) float32
            delta_Es:   (N, n_peaks) float32
            delta_sigmas: (N, n_peaks) float32
        """
        p = self.preset
        H, W = image.shape[:2]
        N = H * W

        # Unpack RGB -> 24-bit integer
        flat = image.reshape(N, 3).astype(np.int64)
        packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]

        # Extract per-peak Hilbert indices
        bpp = p.bits_per_peak
        n_peaks = p.n_peaks
        amps = np.zeros((N, n_peaks), dtype=np.float32)
        dEs = np.zeros((N, n_peaks), dtype=np.float32)
        dSs = np.zeros((N, n_peaks), dtype=np.float32)

        for k in range(n_peaks):
            # Peak 0 is MSB, peak n-1 is LSB
            shift = (n_peaks - 1 - k) * bpp
            mask = (1 << bpp) - 1
            idx = (packed >> shift) & mask

            # Hilbert index -> 3D coordinates
            cx, cy, cz = index_to_coords(idx, p.hilbert_order)

            # Quantized coords -> physical parameters
            amps[:, k] = _dequantize(cx, p.n_levels, p.amplitude_range)
            dEs[:, k] = _dequantize(cy, p.n_levels, p.shift_range)
            dSs[:, k] = _dequantize(cz, p.n_levels, p.sigma_range)

        return amps, dEs, dSs

    def decode(
        self,
        amplitudes: np.ndarray,
        delta_Es: np.ndarray,
        delta_sigmas: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Per-peak parameters -> RGB image (H,W,3) uint8.

        Args:
            amplitudes:    (N, n_peaks) float32
            delta_Es:      (N, n_peaks) float32
            delta_sigmas:  (N, n_peaks) float32
            image_shape:   (H, W)
        """
        p = self.preset
        H, W = image_shape
        N = amplitudes.shape[0]
        n_peaks = p.n_peaks
        bpp = p.bits_per_peak

        packed = np.zeros(N, dtype=np.int64)

        for k in range(n_peaks):
            # Physical parameters -> quantized coordinates
            cx = _quantize(amplitudes[:, k], p.n_levels, p.amplitude_range)
            cy = _quantize(delta_Es[:, k], p.n_levels, p.shift_range)
            cz = _quantize(delta_sigmas[:, k], p.n_levels, p.sigma_range)

            # 3D coordinates -> Hilbert index
            idx = coords_to_index(cx, cy, cz, p.hilbert_order)

            # Pack into 24-bit integer
            shift = (n_peaks - 1 - k) * bpp
            packed |= idx << shift

        # 24-bit integer -> RGB
        r = ((packed >> 16) & 0xFF).astype(np.uint8)
        g = ((packed >> 8) & 0xFF).astype(np.uint8)
        b = (packed & 0xFF).astype(np.uint8)
        return np.stack([r, g, b], axis=1).reshape(H, W, 3)

    def quantization_step(self) -> tuple[float, float, float]:
        """Return the quantization step size for (amp, dE, dsigma)."""
        p = self.preset
        n = p.n_levels
        da = (p.amplitude_range.max_val - p.amplitude_range.min_val) / (n - 1)
        de = (p.shift_range.max_val - p.shift_range.min_val) / (n - 1)
        ds = (p.sigma_range.max_val - p.sigma_range.min_val) / (n - 1)
        return da, de, ds


def _quantize(values: np.ndarray, n_levels: int,
              prange: ParamRange) -> np.ndarray:
    """Map physical values to integer coordinates in [0, n_levels)."""
    t = (values - prange.min_val) / (prange.max_val - prange.min_val)
    coords = np.round(t * (n_levels - 1)).astype(np.int64)
    return np.clip(coords, 0, n_levels - 1)


def _dequantize(coords: np.ndarray, n_levels: int,
                prange: ParamRange) -> np.ndarray:
    """Map integer coordinates in [0, n_levels) to physical values."""
    t = coords.astype(np.float32) / (n_levels - 1)
    return (prange.min_val + t * (prange.max_val - prange.min_val)).astype(np.float32)


# Built-in preset: C 1s two-component (C-C + C-O)
C1S_MULTIPEAK_PRESET = MultiPeakPreset(
    name='C1s_dual',
    elements=[
        ElementSpec('C', '1s', 284.4, gaussian_fwhm=1.0, lorentzian_fwhm=0.25),
        ElementSpec('C', '1s', 286.4, gaussian_fwhm=1.0, lorentzian_fwhm=0.25),
    ],
    energy_range=(280.0, 292.0),
    n_energy=121,
    hilbert_order=4,
    amplitude_range=ParamRange(0.0, 1.0, 'amplitude'),
    shift_range=ParamRange(-0.5, 0.5, 'delta_E'),
    sigma_range=ParamRange(-0.1, 0.1, 'delta_sigma'),
)
