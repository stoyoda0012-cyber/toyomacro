"""
Fisher-Hilbert Encoder
======================

Combines Fisher coordinate transform with 3D Hilbert space-filling curve
for information-optimal parameter encoding.

Encodes (δE, δσ, δγ) into 24-bit RGB where quantization is uniform in
information distance (KL divergence) rather than Euclidean distance.

Key advantages over Euclidean (MultiPeakEncoder) encoding:
    1. σ axis stretched to match information density → finer σ resolution
    2. σ-γ correlation decorrelated → no correlated encoder errors
    3. δγ included → full Voigt parameterization (instead of amplitude
       which is Fisher null space)

dev-log 74 additions:
    - whitened mode: g^{1/2} symmetric transform for better isotropy
    - cubic_bbox: force isotropic Hilbert grid in Fisher space
    - analyze_neighbor_breakdown(): diagnose WHERE neighbors are lost

Reference:
    dev-log 61-62 (Fisher information geometry)
    dev-log 71 (Hilbert encoder)
    dev-log 72 (δσ correlated error, 10 dB gap)
    dev-log 74 (neighbor preservation improvement)
"""

from dataclasses import dataclass

import numpy as np

from .fisher_transform import FisherCoordinateTransform, FisherWhitenedTransform
from .hilbert import coords_to_index, coords_to_index_2d, index_to_coords, index_to_coords_2d


@dataclass
class FisherHilbertConfig:
    """Configuration for Fisher-Hilbert encoder.

    Attributes:
        eta: Voigt mixing ratio (element-dependent). Default 0.15 (C 1s).
        hilbert_order: Hilbert curve order. 2^order levels per dim.
            order=8 → 256 levels, 3×8 = 24-bit → exact RGB.
        dE_range: (min, max) for δE in eV.
        dsigma_range: (min, max) for δσ in eV.
        dgamma_range: (min, max) for δγ in eV.
        amplitude: Nominal amplitude for Fisher computation.
        center: Nominal peak center (eV).
        fwhm_total: Total Voigt FWHM (eV).
        whitened: Use g^{1/2} symmetric transform instead of Cholesky.
        cubic_bbox: Pad bounding box to cubic in Fisher space.
            Makes Hilbert grid isotropic at the cost of PSNR on short axes.
    """
    eta: float = 0.15
    hilbert_order: int = 8
    dE_range: tuple[float, float] = (-2.0, 2.0)
    dsigma_range: tuple[float, float] = (-0.3, 0.3)
    dgamma_range: tuple[float, float] = (-0.1, 0.1)
    amplitude: float = 1000.0
    center: float = 284.4
    fwhm_total: float = 1.0
    whitened: bool = False
    cubic_bbox: bool = False

    @property
    def n_levels(self) -> int:
        return 1 << self.hilbert_order


# Default config for C 1s
C1S_FISHER_CONFIG = FisherHilbertConfig(
    eta=0.15,
    hilbert_order=8,
    dE_range=(-2.0, 2.0),
    dsigma_range=(-0.3, 0.3),
    dgamma_range=(-0.1, 0.1),
)


class FisherHilbertEncoder:
    """Encode Voigt parameters as RGB using Fisher-Hilbert mapping.

    Maps (δE, δσ, δγ) → Fisher coordinates → 3D Hilbert index → RGB.
    Amplitude is not encoded (Fisher null space; solve analytically via LLS).

    With order=8: 256³ grid, 3×8 = 24 bits → exact RGB mapping.

    dev-log 74 options:
        whitened: Use g^{1/2} symmetric transform for better isotropy
        cubic_bbox: Force isotropic Hilbert grid

    Usage::

        enc = FisherHilbertEncoder()  # C 1s defaults
        rgb = enc.encode(dE, dsigma, dgamma)
        rec = enc.decode(rgb)
        # rec['dE'], rec['dsigma'], rec['dgamma']

        # Whitened + cubic for best neighbor preservation:
        cfg = FisherHilbertConfig(whitened=True, cubic_bbox=True)
        enc = FisherHilbertEncoder(config=cfg)
    """

    def __init__(self, config: FisherHilbertConfig | None = None, **kwargs):
        if config is None:
            config = FisherHilbertConfig(**kwargs)
        self.config = config

        total_bits = 3 * config.hilbert_order
        if total_bits > 24:
            raise ValueError(
                f"Total bits ({total_bits}) exceeds 24-bit RGB capacity. "
                f"Reduce hilbert_order (max 8).")

        # Select transform type
        transform_cls = (FisherWhitenedTransform if config.whitened
                         else FisherCoordinateTransform)
        self._transform = transform_cls(
            eta=config.eta,
            fwhm_total=config.fwhm_total,
            amplitude=config.amplitude,
            center=config.center,
        )

        # Bounding box in Fisher space
        xi_min, xi_max = self._transform.fisher_range(
            config.dE_range, config.dsigma_range, config.dgamma_range
        )

        if config.cubic_bbox:
            # Pad bounding box to cubic (equal step in all axes)
            ranges = xi_max - xi_min
            max_range = ranges.max()
            centers = (xi_min + xi_max) / 2.0
            xi_min = centers - max_range / 2.0
            xi_max = centers + max_range / 2.0

        self._xi_min = xi_min
        self._xi_max = xi_max
        self._n_levels = config.n_levels
        self._order = config.hilbert_order

    @property
    def transform(self) -> FisherCoordinateTransform:
        return self._transform

    @property
    def bbox_aspect_ratio(self) -> float:
        """Aspect ratio of the active bounding box (after cubic padding if enabled)."""
        ranges = self._xi_max - self._xi_min
        return float(ranges.max() / max(ranges.min(), 1e-30))

    @property
    def effective_levels(self) -> np.ndarray:
        """Effective quantization levels per Fisher axis (3,).

        For cubic bbox, all axes have n_levels. For default (per-axis),
        each axis uses all n_levels but with different Fisher step sizes.
        """
        # Compute how many levels each param-space axis effectively uses
        # by projecting the Fisher step back to parameter space
        xi_min_orig, xi_max_orig = self._transform.fisher_range(
            self.config.dE_range, self.config.dsigma_range, self.config.dgamma_range
        )
        ranges_orig = xi_max_orig - xi_min_orig
        ranges_active = self._xi_max - self._xi_min
        return (ranges_orig / ranges_active) * self._n_levels

    def quantization_step_fisher(self) -> np.ndarray:
        """Quantization step size in Fisher coordinates (3,)."""
        return (self._xi_max - self._xi_min) / max(self._n_levels - 1, 1)

    def quantization_step_params(self) -> dict[str, float]:
        """Approximate quantization step in parameter space per axis.

        The actual step varies by direction due to the non-diagonal transform.
        Returns the per-axis effective step at θ=0.
        """
        xi_step = self.quantization_step_fisher()
        L_inv = self._transform._L_inv
        names = ['dE', 'dsigma', 'dgamma']
        steps = {}
        for i in range(3):
            e = np.zeros(3)
            e[i] = xi_step[i]
            steps[names[i]] = float(np.linalg.norm(L_inv @ e))
        return steps

    def encode(
        self,
        dE: np.ndarray,
        dsigma: np.ndarray,
        dgamma: np.ndarray,
    ) -> np.ndarray:
        """Encode parameters to RGB.

        Args:
            dE: (N,) center shift deviations
            dsigma: (N,) Gaussian width deviations
            dgamma: (N,) Lorentzian width deviations

        Returns:
            rgb: (N, 3) uint8
        """
        theta = np.stack([
            np.asarray(dE, dtype=np.float64),
            np.asarray(dsigma, dtype=np.float64),
            np.asarray(dgamma, dtype=np.float64),
        ], axis=-1)

        xi = self._transform.to_fisher(theta)

        n = self._n_levels
        coords = np.empty((len(theta), 3), dtype=np.int64)
        for i in range(3):
            t = (xi[:, i] - self._xi_min[i]) / (self._xi_max[i] - self._xi_min[i])
            coords[:, i] = np.clip(
                np.round(t * (n - 1)).astype(np.int64), 0, n - 1
            )

        idx = coords_to_index(coords[:, 0], coords[:, 1], coords[:, 2], self._order)

        r = ((idx >> 16) & 0xFF).astype(np.uint8)
        g = ((idx >> 8) & 0xFF).astype(np.uint8)
        b = (idx & 0xFF).astype(np.uint8)
        return np.stack([r, g, b], axis=1)

    def decode(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Decode RGB to parameters.

        Args:
            rgb: (N, 3) uint8

        Returns:
            Dict with 'dE', 'dsigma', 'dgamma' as (N,) float32 arrays
        """
        flat = rgb.astype(np.int64)
        idx = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]

        cx, cy, cz = index_to_coords(idx, self._order)

        n = self._n_levels
        xi = np.empty((len(idx), 3), dtype=np.float64)
        for i, c in enumerate([cx, cy, cz]):
            t = c.astype(np.float64) / max(n - 1, 1)
            xi[:, i] = self._xi_min[i] + t * (self._xi_max[i] - self._xi_min[i])

        theta = self._transform.from_fisher(xi)

        return {
            'dE': theta[:, 0].astype(np.float32),
            'dsigma': theta[:, 1].astype(np.float32),
            'dgamma': theta[:, 2].astype(np.float32),
        }

    def encode_decode_roundtrip(
        self,
        dE: np.ndarray,
        dsigma: np.ndarray,
        dgamma: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Encode then decode — convenience for PSNR evaluation."""
        rgb = self.encode(dE, dsigma, dgamma)
        return self.decode(rgb)

    def psnr(
        self,
        dE_gt: np.ndarray,
        dsigma_gt: np.ndarray,
        dgamma_gt: np.ndarray,
    ) -> dict[str, float]:
        """Compute per-parameter PSNR for encode-decode round trip.

        PSNR = 10 * log10(range² / MSE)

        Returns:
            Dict with 'dE', 'dsigma', 'dgamma' PSNR values in dB
        """
        rec = self.encode_decode_roundtrip(dE_gt, dsigma_gt, dgamma_gt)
        result = {}
        for name, gt_arr, rng in [
            ('dE', dE_gt, self.config.dE_range),
            ('dsigma', dsigma_gt, self.config.dsigma_range),
            ('dgamma', dgamma_gt, self.config.dgamma_range),
        ]:
            mse = float(np.mean((gt_arr - rec[name]) ** 2))
            signal_range = rng[1] - rng[0]
            if mse < 1e-30:
                result[name] = np.inf
            else:
                result[name] = float(10 * np.log10(signal_range ** 2 / mse))
        return result


# -----------------------------------------------------------------------
# Split Fisher-Hilbert Encoder
# -----------------------------------------------------------------------

@dataclass
class SplitFisherHilbertConfig:
    """Configuration for Split Fisher-Hilbert encoder.

    Exploits δE independence and δσ-δγ correlation (r≈0.70)
    via separate encodings:
        R: δE → 8-bit linear (independent channel)
        G+B: (δσ, δγ) → Fisher 2D → 2D Hilbert → 16-bit

    Attributes:
        eta: Voigt mixing ratio (element-dependent). Default 0.15 (C 1s).
        hilbert_order: 2D Hilbert curve order. 2^order levels per axis.
            order=8 → 256×256 grid → 16-bit → exact G+B mapping.
        dE_range: (min, max) for δE in eV.
        dsigma_range: (min, max) for δσ in eV.
        dgamma_range: (min, max) for δγ in eV.
        amplitude: Nominal amplitude for Fisher computation.
        center: Nominal peak center (eV).
        fwhm_total: Total Voigt FWHM (eV).
    """
    eta: float = 0.15
    hilbert_order: int = 8
    dE_range: tuple[float, float] = (-2.0, 2.0)
    dsigma_range: tuple[float, float] = (-0.3, 0.3)
    dgamma_range: tuple[float, float] = (-0.1, 0.1)
    amplitude: float = 1000.0
    center: float = 284.4
    fwhm_total: float = 1.0

    @property
    def n_levels(self) -> int:
        return 1 << self.hilbert_order

    @property
    def dE_levels(self) -> int:
        return 256  # 8-bit R channel


# Default config for C 1s split encoding
C1S_SPLIT_CONFIG = SplitFisherHilbertConfig(
    eta=0.15,
    hilbert_order=8,
    dE_range=(-2.0, 2.0),
    dsigma_range=(-0.3, 0.3),
    dgamma_range=(-0.1, 0.1),
)


class SplitFisherHilbertEncoder:
    """Encode Voigt parameters using split 2D+1D strategy.

    Exploits dev-log 61 finding that δE is Fisher-independent from (δσ, δγ),
    and dev-log 62 finding that δσ-δγ are strongly correlated (r≈0.70).

    Encoding scheme:
        R: δE → 8-bit linear (256 levels, independent channel)
        G+B: (δσ, δγ) → 2×2 Fisher Cholesky → 2D Hilbert → 16-bit

    Advantages over 3D Hilbert:
        1. δE gets full 8-bit resolution (256 levels vs ~256/aspect_ratio)
        2. δσ-δγ plane gets dedicated 2D Hilbert (256×256 = 65536 bins)
        3. 2D Hilbert has better locality than 3D for the correlated pair
        4. No wasted bits on the δE dimension in the Hilbert curve

    Usage::

        enc = SplitFisherHilbertEncoder()  # C 1s defaults
        rgb = enc.encode(dE, dsigma, dgamma)
        rec = enc.decode(rgb)
    """

    def __init__(self, config: SplitFisherHilbertConfig | None = None, **kwargs):
        if config is None:
            config = SplitFisherHilbertConfig(**kwargs)
        self.config = config

        if config.hilbert_order > 8:
            raise ValueError(
                f"2D Hilbert order {config.hilbert_order} exceeds 8 "
                f"(would need {2 * config.hilbert_order} > 16 bits for G+B).")

        # Full 3D Fisher transform (for diagnostics / distance computation)
        self._transform_3d = FisherCoordinateTransform(
            eta=config.eta,
            fwhm_total=config.fwhm_total,
            amplitude=config.amplitude,
            center=config.center,
        )

        # Extract 2×2 Fisher sub-matrix for (δσ, δγ)
        g_3x3 = self._transform_3d._g  # (δE, δσ, δγ) Fisher matrix
        self._g_sg = g_3x3[1:3, 1:3].copy()  # (δσ, δγ) 2×2 sub-block

        # Cholesky of 2×2 sub-block
        self._L_sg = np.linalg.cholesky(self._g_sg)
        self._L_sg_inv = np.linalg.inv(self._L_sg)

        # δE scaling factor (diagonal element of Fisher)
        self._g_EE = g_3x3[0, 0]
        self._scale_dE = np.sqrt(self._g_EE)

        # σ-γ bounding box in Fisher 2D space
        corners_sg = np.array([
            [ds, dg]
            for ds in config.dsigma_range
            for dg in config.dgamma_range
        ])
        xi_sg = (self._L_sg @ corners_sg.T).T
        self._xi_sg_min = xi_sg.min(axis=0)
        self._xi_sg_max = xi_sg.max(axis=0)

        self._n_levels = config.n_levels  # 2D Hilbert levels per axis
        self._order = config.hilbert_order
        self._dE_levels = config.dE_levels

    @property
    def transform(self) -> FisherCoordinateTransform:
        """Full 3D Fisher transform (for neighbor preservation analysis)."""
        return self._transform_3d

    @property
    def sg_correlation(self) -> float:
        """Normalized σ-γ correlation from Fisher sub-matrix."""
        return float(
            self._g_sg[0, 1] / np.sqrt(self._g_sg[0, 0] * self._g_sg[1, 1])
        )

    @property
    def sg_bbox_aspect_ratio(self) -> float:
        """Aspect ratio of σ-γ bounding box in Fisher 2D space."""
        ranges = self._xi_sg_max - self._xi_sg_min
        return float(ranges.max() / max(ranges.min(), 1e-30))

    @property
    def bbox_aspect_ratio(self) -> float:
        """Overall effective aspect ratio (for compatibility with analysis functions)."""
        return self.sg_bbox_aspect_ratio

    @property
    def effective_levels(self) -> np.ndarray:
        """Effective quantization levels per parameter: (dE, dsigma, dgamma)."""
        return np.array([
            float(self._dE_levels),
            float(self._n_levels),
            float(self._n_levels),
        ])

    def quantization_step_params(self) -> dict[str, float]:
        """Approximate quantization step in parameter space per axis."""
        dE_step = (self.config.dE_range[1] - self.config.dE_range[0]) / max(self._dE_levels - 1, 1)
        xi_step = (self._xi_sg_max - self._xi_sg_min) / max(self._n_levels - 1, 1)
        # Back-transform to parameter space
        e_sg = np.zeros(2)
        steps_sg = {}
        for i, name in enumerate(['dsigma', 'dgamma']):
            e_sg[:] = 0
            e_sg[i] = xi_step[i]
            steps_sg[name] = float(np.linalg.norm(self._L_sg_inv @ e_sg))
        return {'dE': float(dE_step), **steps_sg}

    def quantization_step_fisher(self) -> np.ndarray:
        """Quantization step size in Fisher coordinates (3,) for compatibility."""
        dE_step_param = (self.config.dE_range[1] - self.config.dE_range[0]) / max(self._dE_levels - 1, 1)
        dE_step_fisher = dE_step_param * self._scale_dE
        sg_step = (self._xi_sg_max - self._xi_sg_min) / max(self._n_levels - 1, 1)
        return np.array([dE_step_fisher, sg_step[0], sg_step[1]])

    def encode(
        self,
        dE: np.ndarray,
        dsigma: np.ndarray,
        dgamma: np.ndarray,
    ) -> np.ndarray:
        """Encode parameters to RGB.

        R: δE → 8-bit linear
        G+B: (δσ, δγ) → Fisher 2D → 2D Hilbert → 16-bit

        Args:
            dE: (N,) center shift deviations
            dsigma: (N,) Gaussian width deviations
            dgamma: (N,) Lorentzian width deviations

        Returns:
            rgb: (N, 3) uint8
        """
        dE = np.asarray(dE, dtype=np.float64)
        dsigma = np.asarray(dsigma, dtype=np.float64)
        dgamma = np.asarray(dgamma, dtype=np.float64)

        # R: δE linear
        dE_min, dE_max = self.config.dE_range
        t_dE = (dE - dE_min) / (dE_max - dE_min)
        r = np.clip(np.round(t_dE * 255).astype(np.int64), 0, 255).astype(np.uint8)

        # G+B: (δσ, δγ) → Fisher 2D → 2D Hilbert
        theta_sg = np.stack([dsigma, dgamma], axis=-1)
        xi_sg = (self._L_sg @ theta_sg.T).T

        n = self._n_levels
        coords = np.empty((len(dE), 2), dtype=np.int64)
        for i in range(2):
            t = (xi_sg[:, i] - self._xi_sg_min[i]) / (self._xi_sg_max[i] - self._xi_sg_min[i])
            coords[:, i] = np.clip(np.round(t * (n - 1)).astype(np.int64), 0, n - 1)

        hilbert_idx = coords_to_index_2d(coords[:, 0], coords[:, 1], self._order)

        g = ((hilbert_idx >> 8) & 0xFF).astype(np.uint8)
        b = (hilbert_idx & 0xFF).astype(np.uint8)
        return np.stack([r, g, b], axis=1)

    def decode(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Decode RGB to parameters.

        Args:
            rgb: (N, 3) uint8

        Returns:
            Dict with 'dE', 'dsigma', 'dgamma' as (N,) float32 arrays
        """
        r = rgb[:, 0].astype(np.float64)
        g = rgb[:, 1].astype(np.int64)
        b = rgb[:, 2].astype(np.int64)

        # R → δE
        dE_min, dE_max = self.config.dE_range
        dE = dE_min + (r / 255.0) * (dE_max - dE_min)

        # G+B → 2D Hilbert → (δσ, δγ)
        hilbert_idx = (g << 8) | b
        cx, cy = index_to_coords_2d(hilbert_idx, self._order)

        n = self._n_levels
        xi_sg = np.empty((len(r), 2), dtype=np.float64)
        for i, c in enumerate([cx, cy]):
            t = c.astype(np.float64) / max(n - 1, 1)
            xi_sg[:, i] = self._xi_sg_min[i] + t * (self._xi_sg_max[i] - self._xi_sg_min[i])

        theta_sg = (self._L_sg_inv @ xi_sg.T).T

        return {
            'dE': dE.astype(np.float32),
            'dsigma': theta_sg[:, 0].astype(np.float32),
            'dgamma': theta_sg[:, 1].astype(np.float32),
        }

    def encode_decode_roundtrip(
        self,
        dE: np.ndarray,
        dsigma: np.ndarray,
        dgamma: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Encode then decode — convenience for PSNR evaluation."""
        rgb = self.encode(dE, dsigma, dgamma)
        return self.decode(rgb)

    def psnr(
        self,
        dE_gt: np.ndarray,
        dsigma_gt: np.ndarray,
        dgamma_gt: np.ndarray,
    ) -> dict[str, float]:
        """Compute per-parameter PSNR for encode-decode round trip."""
        rec = self.encode_decode_roundtrip(dE_gt, dsigma_gt, dgamma_gt)
        result = {}
        for name, gt_arr, rng in [
            ('dE', dE_gt, self.config.dE_range),
            ('dsigma', dsigma_gt, self.config.dsigma_range),
            ('dgamma', dgamma_gt, self.config.dgamma_range),
        ]:
            mse = float(np.mean((gt_arr - rec[name]) ** 2))
            signal_range = rng[1] - rng[0]
            if mse < 1e-30:
                result[name] = np.inf
            else:
                result[name] = float(10 * np.log10(signal_range ** 2 / mse))
        return result


# -----------------------------------------------------------------------
# Neighbor preservation analysis
# -----------------------------------------------------------------------

def compute_neighbor_preservation(
    encoder: FisherHilbertEncoder,
    n_samples: int = 10000,
    k: int = 10,
    seed: int = 42,
) -> float:
    """Quantify locality preservation of the encoder.

    Measures what fraction of Fisher-space k-nearest-neighbors
    remain k-nearest-neighbors in the encoded (Hilbert index) space.

    Higher is better. Fisher-Hilbert should outperform Euclidean-Hilbert
    because the Fisher transform aligns the parameter space with
    information geometry before applying the Hilbert curve.

    Args:
        encoder: FisherHilbertEncoder instance
        n_samples: Number of test points
        k: Number of neighbors to check
        seed: Random seed

    Returns:
        Mean preservation rate in [0, 1]
    """
    rng = np.random.default_rng(seed)
    cfg = encoder.config

    dE = rng.uniform(*cfg.dE_range, n_samples).astype(np.float64)
    ds = rng.uniform(*cfg.dsigma_range, n_samples).astype(np.float64)
    dg = rng.uniform(*cfg.dgamma_range, n_samples).astype(np.float64)

    # Fisher-space coordinates
    theta = np.stack([dE, ds, dg], axis=-1)
    xi = encoder.transform.to_fisher(theta)

    # Hilbert indices
    rgb = encoder.encode(dE, ds, dg)
    flat = rgb.astype(np.int64)
    hilbert_idx = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]

    # Subsample for speed (pairwise distance is O(n²))
    n_query = min(500, n_samples)
    query_idx = rng.choice(n_samples, n_query, replace=False)

    preservation = 0.0
    for qi in query_idx:
        # Fisher-space distances
        d_fisher = np.sqrt(np.sum((xi - xi[qi]) ** 2, axis=1))
        d_fisher[qi] = np.inf
        fisher_neighbors = set(np.argsort(d_fisher)[:k])

        # Hilbert-index distances
        d_hilbert = np.abs(hilbert_idx - hilbert_idx[qi])
        d_hilbert[qi] = np.iinfo(np.int64).max
        hilbert_neighbors = set(np.argsort(d_hilbert)[:k])

        preservation += len(fisher_neighbors & hilbert_neighbors) / k

    return preservation / n_query


@dataclass
class NeighborBreakdownResult:
    """Detailed analysis of where neighbor preservation fails.

    Attributes:
        preservation: Overall preservation rate [0, 1]
        direction_histogram: (3,) fraction of broken pairs along each
            eigenvector direction of the Fisher matrix. Largest value
            indicates the principal direction of neighbor breakdown.
        fisher_step_ratios: (3,) quantization step per axis in Fisher
            space, normalized to the minimum step. Ratio > 1 indicates
            anisotropic grid.
        bbox_aspect_ratio: max_range / min_range of the Fisher bounding box.
        broken_distances: (n_broken,) Fisher distances of broken neighbor pairs.
        preserved_distances: (n_preserved,) Fisher distances of preserved pairs.
    """
    preservation: float
    direction_histogram: np.ndarray
    fisher_step_ratios: np.ndarray
    bbox_aspect_ratio: float
    broken_distances: np.ndarray
    preserved_distances: np.ndarray


def compute_grid_neighbor_preservation(
    encoder: FisherHilbertEncoder,
    n_samples: int = 10000,
    k: int = 10,
    seed: int = 42,
) -> float:
    """Measure Hilbert preservation using GRID coordinates.

    Uses quantized integer grid coordinates for k-NN instead of
    continuous Fisher coordinates. This isolates the Hilbert curve's
    intrinsic locality from the grid anisotropy.

    Grid preservation represents the theoretical maximum preservation
    achievable for the given encoding, regardless of how well the
    Fisher metric aligns with the grid.

    Args:
        encoder: FisherHilbertEncoder instance
        n_samples: Number of test points
        k: Number of neighbors to check
        seed: Random seed

    Returns:
        Mean preservation rate in [0, 1]
    """
    rng = np.random.default_rng(seed)
    cfg = encoder.config

    dE = rng.uniform(*cfg.dE_range, n_samples).astype(np.float64)
    ds = rng.uniform(*cfg.dsigma_range, n_samples).astype(np.float64)
    dg = rng.uniform(*cfg.dgamma_range, n_samples).astype(np.float64)

    # Grid coordinates (quantized)
    theta = np.stack([dE, ds, dg], axis=-1)
    xi = encoder.transform.to_fisher(theta)
    n = encoder._n_levels
    coords = np.empty((n_samples, 3), dtype=np.float64)
    for i in range(3):
        t = (xi[:, i] - encoder._xi_min[i]) / (encoder._xi_max[i] - encoder._xi_min[i])
        coords[:, i] = np.clip(np.round(t * (n - 1)), 0, n - 1)

    # Hilbert indices
    rgb = encoder.encode(dE, ds, dg)
    flat = rgb.astype(np.int64)
    hilbert_idx = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]

    n_query = min(500, n_samples)
    query_idx = rng.choice(n_samples, n_query, replace=False)

    preservation = 0.0
    for qi in query_idx:
        d_grid = np.sqrt(np.sum((coords - coords[qi]) ** 2, axis=1))
        d_grid[qi] = np.inf
        grid_neighbors = set(np.argsort(d_grid)[:k])

        d_hilbert = np.abs(hilbert_idx - hilbert_idx[qi])
        d_hilbert[qi] = np.iinfo(np.int64).max
        hilbert_neighbors = set(np.argsort(d_hilbert)[:k])

        preservation += len(grid_neighbors & hilbert_neighbors) / k

    return preservation / n_query


def analyze_neighbor_breakdown(
    encoder: FisherHilbertEncoder,
    n_samples: int = 5000,
    k: int = 10,
    seed: int = 42,
) -> NeighborBreakdownResult:
    """Analyze WHERE neighbor preservation breaks down.

    For each broken neighbor pair (Fisher-close but Hilbert-far),
    identifies the direction in Fisher space along which they differ most.
    Aggregates into a histogram over the 3 Fisher axes.

    This reveals whether the breakdown is dominated by one axis
    (indicating anisotropic grid) or spread uniformly.

    Args:
        encoder: FisherHilbertEncoder instance
        n_samples: Number of test points
        k: Number of neighbors to check
        seed: Random seed

    Returns:
        NeighborBreakdownResult with detailed diagnostics
    """
    rng = np.random.default_rng(seed)
    cfg = encoder.config

    dE = rng.uniform(*cfg.dE_range, n_samples).astype(np.float64)
    ds = rng.uniform(*cfg.dsigma_range, n_samples).astype(np.float64)
    dg = rng.uniform(*cfg.dgamma_range, n_samples).astype(np.float64)

    # Fisher coordinates
    theta = np.stack([dE, ds, dg], axis=-1)
    xi = encoder.transform.to_fisher(theta)

    # Hilbert indices
    rgb = encoder.encode(dE, ds, dg)
    flat = rgb.astype(np.int64)
    hilbert_idx = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]

    # Fisher step ratios
    fisher_steps = encoder.quantization_step_fisher()
    fisher_step_ratios = fisher_steps / fisher_steps.min()

    n_query = min(300, n_samples)
    query_idx = rng.choice(n_samples, n_query, replace=False)

    direction_counts = np.zeros(3, dtype=np.float64)
    broken_dists = []
    preserved_dists = []
    total_preservation = 0.0

    for qi in query_idx:
        d_fisher = np.sqrt(np.sum((xi - xi[qi]) ** 2, axis=1))
        d_fisher[qi] = np.inf
        fisher_nn = np.argsort(d_fisher)[:k]
        fisher_set = set(fisher_nn)

        d_hilbert = np.abs(hilbert_idx - hilbert_idx[qi])
        d_hilbert[qi] = np.iinfo(np.int64).max
        hilbert_nn = np.argsort(d_hilbert)[:k]
        hilbert_set = set(hilbert_nn)

        preserved = fisher_set & hilbert_set
        broken = fisher_set - hilbert_set

        total_preservation += len(preserved) / k

        for bi in broken:
            diff = np.abs(xi[bi] - xi[qi])
            dominant_axis = int(np.argmax(diff))
            direction_counts[dominant_axis] += 1
            broken_dists.append(d_fisher[bi])

        for pi in preserved:
            preserved_dists.append(d_fisher[pi])

    preservation = total_preservation / n_query
    total_broken = direction_counts.sum()
    if total_broken > 0:
        direction_histogram = direction_counts / total_broken
    else:
        direction_histogram = np.ones(3) / 3.0

    return NeighborBreakdownResult(
        preservation=preservation,
        direction_histogram=direction_histogram,
        fisher_step_ratios=fisher_step_ratios,
        bbox_aspect_ratio=encoder.bbox_aspect_ratio,
        broken_distances=np.array(broken_dists),
        preserved_distances=np.array(preserved_dists),
    )


# -----------------------------------------------------------------------
# σ-γ plane neighbor preservation
# -----------------------------------------------------------------------

def compute_sg_neighbor_preservation(
    encoder,
    n_samples: int = 10000,
    k: int = 10,
    seed: int = 42,
) -> float:
    """Measure neighbor preservation in the σ-γ sub-plane only.

    For each test point, finds k nearest neighbors using only (δσ, δγ)
    Fisher coordinates, then checks overlap with k nearest in the
    σ-γ encoding space.

    For SplitFisherHilbertEncoder: uses only G+B channels (16-bit Hilbert
    index) since R encodes independent δE.
    For FisherHilbertEncoder: uses full 24-bit index (σ-γ is mixed into 3D).

    This metric isolates the encoder's ability to preserve the strongly
    correlated σ-γ structure (r≈0.70).

    Works with both FisherHilbertEncoder and SplitFisherHilbertEncoder.

    Args:
        encoder: Encoder with .config, .transform, .encode() methods
        n_samples: Number of test points
        k: Number of neighbors to check
        seed: Random seed

    Returns:
        Mean σ-γ preservation rate in [0, 1]
    """
    rng = np.random.default_rng(seed)
    cfg = encoder.config

    dE = rng.uniform(*cfg.dE_range, n_samples).astype(np.float64)
    ds = rng.uniform(*cfg.dsigma_range, n_samples).astype(np.float64)
    dg = rng.uniform(*cfg.dgamma_range, n_samples).astype(np.float64)

    # Fisher-space σ-γ coordinates (use full 3D transform, take [1:3])
    theta = np.stack([dE, ds, dg], axis=-1)
    xi = encoder.transform.to_fisher(theta)
    xi_sg = xi[:, 1:3]  # σ-γ plane only

    # Encoded σ-γ index
    rgb = encoder.encode(dE, ds, dg)
    flat = rgb.astype(np.int64)
    if isinstance(encoder, SplitFisherHilbertEncoder):
        # Split: G+B channels carry σ-γ Hilbert index (16-bit)
        encoded_idx = (flat[:, 1] << 8) | flat[:, 2]
    else:
        # 3D: full 24-bit index (σ-γ mixed in)
        encoded_idx = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]

    n_query = min(500, n_samples)
    query_idx = rng.choice(n_samples, n_query, replace=False)

    preservation = 0.0
    for qi in query_idx:
        # σ-γ Fisher distances only
        d_sg = np.sqrt(np.sum((xi_sg - xi_sg[qi]) ** 2, axis=1))
        d_sg[qi] = np.inf
        sg_neighbors = set(np.argsort(d_sg)[:k])

        # Encoded-space distances
        d_enc = np.abs(encoded_idx - encoded_idx[qi])
        d_enc[qi] = np.iinfo(np.int64).max
        enc_neighbors = set(np.argsort(d_enc)[:k])

        preservation += len(sg_neighbors & enc_neighbors) / k

    return preservation / n_query


# -----------------------------------------------------------------------
# TwoPeakSplitEncoder
# -----------------------------------------------------------------------


class TwoPeakSplitEncoder:
    """2-peak Split Fisher-Hilbert encoder.

    Encodes two peaks independently using SplitFisherHilbertEncoder,
    producing 6 channels (2 × RGB = 2 × 3 uint8).

    Peak 1: R1G1B1 via SplitFisherHilbertEncoder(order_peak1)
    Peak 2: R2G2B2 via SplitFisherHilbertEncoder(order_peak2)

    Asymmetric orders allow allocating more bits to the primary peak
    (e.g., order_peak1=8 for 256×256, order_peak2=4 for 16×16).

    Usage::

        enc = TwoPeakSplitEncoder(eta=0.15, order_peak1=8, order_peak2=4)
        channels = enc.encode(dE1, ds1, dg1, dE2, ds2, dg2)  # (N, 6) uint8
        rec = enc.decode(channels)
        # rec['dE1'], rec['dsigma1'], rec['dgamma1']
        # rec['dE2'], rec['dsigma2'], rec['dgamma2']
    """

    def __init__(
        self,
        eta: float = 0.15,
        order_peak1: int = 8,
        order_peak2: int = 4,
        dE_range: tuple[float, float] = (-0.5, 0.5),
        dsigma_range: tuple[float, float] = (-0.1, 0.1),
        dgamma_range: tuple[float, float] = (-0.05, 0.05),
    ):
        cfg1 = SplitFisherHilbertConfig(
            eta=eta, hilbert_order=order_peak1,
            dE_range=dE_range, dsigma_range=dsigma_range,
            dgamma_range=dgamma_range,
        )
        cfg2 = SplitFisherHilbertConfig(
            eta=eta, hilbert_order=order_peak2,
            dE_range=dE_range, dsigma_range=dsigma_range,
            dgamma_range=dgamma_range,
        )
        self.enc1 = SplitFisherHilbertEncoder(config=cfg1)
        self.enc2 = SplitFisherHilbertEncoder(config=cfg2)

    def encode(
        self,
        dE1: np.ndarray, dsigma1: np.ndarray, dgamma1: np.ndarray,
        dE2: np.ndarray, dsigma2: np.ndarray, dgamma2: np.ndarray,
    ) -> np.ndarray:
        """Encode two peaks to 6 channels (2 × RGB).

        Args:
            dE1, dsigma1, dgamma1: (N,) peak 1 parameters.
            dE2, dsigma2, dgamma2: (N,) peak 2 parameters.

        Returns:
            channels: (N, 6) uint8 — [R1, G1, B1, R2, G2, B2].
        """
        rgb1 = self.enc1.encode(dE1, dsigma1, dgamma1)  # (N, 3) uint8
        rgb2 = self.enc2.encode(dE2, dsigma2, dgamma2)  # (N, 3) uint8
        return np.concatenate([rgb1, rgb2], axis=1)

    def decode(self, channels: np.ndarray) -> dict[str, np.ndarray]:
        """Decode 6 channels to two peaks' parameters.

        Args:
            channels: (N, 6) uint8 — [R1, G1, B1, R2, G2, B2].

        Returns:
            Dict with keys dE1, dsigma1, dgamma1, dE2, dsigma2, dgamma2.
        """
        rgb1 = channels[:, :3]
        rgb2 = channels[:, 3:6]
        rec1 = self.enc1.decode(rgb1)
        rec2 = self.enc2.decode(rgb2)
        return {
            'dE1': rec1['dE'],
            'dsigma1': rec1['dsigma'],
            'dgamma1': rec1['dgamma'],
            'dE2': rec2['dE'],
            'dsigma2': rec2['dsigma'],
            'dgamma2': rec2['dgamma'],
        }


# ---------------------------------------------------------------------------
# dev-log 134 bit-budget redesign — LinearTwoPeakEncoder
# ---------------------------------------------------------------------------


class LinearTwoPeakEncoder:
    """δE-priority 2-peak encoder.

    dev-log 134's encoder variant sweep showed that the δσ recovery ceiling
    in the AP + Newton pipeline is set by **δE quantization**, not by δσ
    quantization. Increasing the δσ encoder precision from 8 → 16 bits leaves
    δσ PSNR stuck at ~34 dB, while making δE exact lifts δσ PSNR to ~46 dB.

    This encoder reallocates the per-peak 24-bit RGB budget accordingly:

        R + B  →  16-bit linear δE   (joined to recover δE = R*256 + B)
        G      →  8-bit  linear δσ
        (no δγ encoding — assumes δγ = 0)

    Per-peak step sizes for the dev-log 77 default ranges:
        δE step = 1.0 / 65535 ≈ 1.53e-5  (was 3.92e-3 at 8 bits)
        δσ step = 0.2 / 255   ≈ 7.84e-4  (≈ Fisher-Cholesky effective step)

    With this layout the Newton pipeline recovers (per the bypass + Newton
    ceiling): amp ~63 dB → 81 dB at n_iter=5, δE ≈ 74 dB, δσ ≈ 46 dB.
    """

    def __init__(
        self,
        dE_range: tuple[float, float] = (-0.5, 0.5),
        dsigma_range: tuple[float, float] = (-0.1, 0.1),
    ):
        self.dE_range = dE_range
        self.dsigma_range = dsigma_range
        # Levels: 16-bit for δE, 8-bit for δσ
        self._dE_levels = 65536
        self._ds_levels = 256

    @property
    def channels_per_peak(self) -> int:
        return 3

    def quantization_step_params(self) -> dict[str, float]:
        return {
            "dE":     (self.dE_range[1] - self.dE_range[0]) / (self._dE_levels - 1),
            "dsigma": (self.dsigma_range[1] - self.dsigma_range[0]) / (self._ds_levels - 1),
            "dgamma": 0.0,
        }

    # ------------------------------------------------------------------
    # Per-peak encode / decode helpers
    # ------------------------------------------------------------------

    def _encode_peak(
        self,
        dE: np.ndarray,
        dsigma: np.ndarray,
    ) -> np.ndarray:
        """Encode one peak's (δE, δσ) into a uint8 RGB triple.

        Layout: R = high byte of δE quant, B = low byte of δE quant, G = δσ quant.
        """
        dE = np.asarray(dE, dtype=np.float64)
        dsigma = np.asarray(dsigma, dtype=np.float64)
        n = len(dE)

        # 16-bit δE quantization
        dE_lo, dE_hi = self.dE_range
        t_dE = (dE - dE_lo) / (dE_hi - dE_lo)
        q_dE = np.clip(np.round(t_dE * (self._dE_levels - 1)),
                        0, self._dE_levels - 1).astype(np.int64)
        r = ((q_dE >> 8) & 0xFF).astype(np.uint8)
        b = (q_dE & 0xFF).astype(np.uint8)

        # 8-bit δσ quantization
        ds_lo, ds_hi = self.dsigma_range
        t_ds = (dsigma - ds_lo) / (ds_hi - ds_lo)
        g = np.clip(np.round(t_ds * (self._ds_levels - 1)),
                     0, self._ds_levels - 1).astype(np.uint8)

        return np.stack([r, g, b], axis=1)

    def _decode_peak(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct (δE, δσ) from a uint8 RGB triple."""
        rgb = np.asarray(rgb)
        r = rgb[:, 0].astype(np.int64)
        g = rgb[:, 1].astype(np.int64)
        b = rgb[:, 2].astype(np.int64)

        q_dE = (r << 8) | b
        dE_lo, dE_hi = self.dE_range
        dE = (q_dE / (self._dE_levels - 1)) * (dE_hi - dE_lo) + dE_lo

        ds_lo, ds_hi = self.dsigma_range
        dsigma = (g / (self._ds_levels - 1)) * (ds_hi - ds_lo) + ds_lo

        return dE.astype(np.float32), dsigma.astype(np.float32)

    # ------------------------------------------------------------------
    # Two-peak encode / decode (matches TwoPeakSplitEncoder API)
    # ------------------------------------------------------------------

    def encode(
        self,
        dE1: np.ndarray, dsigma1: np.ndarray, dgamma1: np.ndarray,
        dE2: np.ndarray, dsigma2: np.ndarray, dgamma2: np.ndarray,
    ) -> np.ndarray:
        """Encode two peaks to 6 channels (2 × RGB). δγ inputs are ignored.

        Returns: (N, 6) uint8 — [R1, G1, B1, R2, G2, B2].
        """
        rgb1 = self._encode_peak(dE1, dsigma1)
        rgb2 = self._encode_peak(dE2, dsigma2)
        return np.concatenate([rgb1, rgb2], axis=1)

    def decode(self, channels: np.ndarray) -> dict[str, np.ndarray]:
        """Decode 6 channels to two peaks' parameters (δγ is always 0)."""
        dE1, ds1 = self._decode_peak(channels[:, :3])
        dE2, ds2 = self._decode_peak(channels[:, 3:6])
        z = np.zeros_like(dE1)
        return {
            "dE1": dE1, "dsigma1": ds1, "dgamma1": z,
            "dE2": dE2, "dsigma2": ds2, "dgamma2": z.copy(),
        }
