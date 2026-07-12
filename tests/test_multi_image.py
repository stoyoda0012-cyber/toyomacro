"""Smoke tests for multi-image GVRT generalization benchmark.

Uses synthetic RGB images (no external file dependencies) to verify
the round-trip pipeline: encode → exact Voigt → solver → decode → PSNR.

Validates:
- Pipeline runs without error on varied image patterns (4-step & parabola)
- PSNR exceeds minimum threshold (noiseless round-trip)
- PSNR is consistent across different image patterns (spread < 3 dB)
- Parabola solver improves B(FWHM) over 4-step Taylor solver
"""

import os

import numpy as np
import pytest

from toyomacro.voigtfit._mlx_support import mlx_usable as _mlx_usable

IN_CI = os.environ.get("CI") == "true"

from toyomacro.voigtfit.benchmarks.bench_multi_image import (
    compute_channel_psnr,
    roundtrip_single_image,
)
from toyomacro.voigtfit.param_encoder import (
    C1S_SINGLE_PRESET,
    SinglePeakEncoder,
)

# ---------------------------------------------------------------------------
# Synthetic image generators (deterministic, no file I/O)
# ---------------------------------------------------------------------------


def _make_gradient_image(H: int = 32, W: int = 32) -> np.ndarray:
    """Smooth RGB gradient: R varies horizontally, G vertically, B diagonally."""
    x = np.linspace(0, 255, W, dtype=np.float32)
    y = np.linspace(0, 255, H, dtype=np.float32)
    r = np.tile(x, (H, 1))
    g = np.tile(y[:, np.newaxis], (1, W))
    b = (r + g) / 2
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _make_random_image(
    H: int = 32, W: int = 32, seed: int = 42
) -> np.ndarray:
    """Uniformly random RGB values."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def _make_midgray_image(H: int = 32, W: int = 32) -> np.ndarray:
    """Uniform mid-gray (128, 128, 128) — tests center of parameter range."""
    return np.full((H, W, 3), 128, dtype=np.uint8)


def _make_stripe_image(H: int = 32, W: int = 32) -> np.ndarray:
    """Alternating bright/dark vertical stripes in each channel."""
    image = np.zeros((H, W, 3), dtype=np.uint8)
    for ch in range(3):
        stripe_w = max(1, W // (4 + ch * 2))
        for x in range(W):
            if (x // stripe_w) % 2 == 0:
                image[:, x, ch] = 200
            else:
                image[:, x, ch] = 50
    return image


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


SYNTHETIC_IMAGES = {
    "gradient": _make_gradient_image,
    "random": _make_random_image,
    "midgray": _make_midgray_image,
    "stripe": _make_stripe_image,
}


class TestRoundtripSingle:
    """Test round-trip on individual synthetic images."""

    @pytest.mark.parametrize("name", list(SYNTHETIC_IMAGES.keys()))
    def test_roundtrip_runs(self, name: str):
        """Round-trip completes without error for each image pattern."""
        image = SYNTHETIC_IMAGES[name]()
        psnr, gen_time, fit_time = roundtrip_single_image(image)

        assert psnr.amplitude > 0
        assert psnr.shift > 0
        assert psnr.fwhm > 0
        assert gen_time > 0
        assert fit_time > 0

    @pytest.mark.parametrize("name", list(SYNTHETIC_IMAGES.keys()))
    def test_psnr_threshold(self, name: str):
        """Noiseless round-trip per-channel PSNR above known solver floors.

        4-step Taylor solver characteristics (exact Voigt spectra):
        - R(amp): ~23-55 dB depending on δσ range
        - G(δE):  ~22-55 dB depending on perturbation range
        - B(FWHM): ~5-40 dB — known weakness (1st-order Taylor for σ)

        Thresholds are set conservatively below the worst observed case
        for full-range parameter distributions.
        """
        image = SYNTHETIC_IMAGES[name]()
        psnr, _, _ = roundtrip_single_image(image)

        assert psnr.amplitude > 20.0, (
            f"{name}: R(amp)={psnr.amplitude:.1f} dB < 20 dB"
        )
        assert psnr.shift > 20.0, (
            f"{name}: G(δE)={psnr.shift:.1f} dB < 20 dB"
        )
        # B(FWHM) is the weakest channel — Taylor δσ saturation
        assert psnr.fwhm > 3.0, (
            f"{name}: B(FWHM)={psnr.fwhm:.1f} dB < 3 dB"
        )
        # Mean PSNR should be reasonable
        assert psnr.mean > 15.0, (
            f"{name}: mean={psnr.mean:.1f} dB < 15 dB"
        )


class TestConsistency:
    """Test PSNR consistency across different image patterns."""

    # Images with full-range parameter distributions (excludes midgray
    # which is degenerate: all pixels identical → near-zero perturbation)
    FULL_RANGE_IMAGES = ["gradient", "random", "stripe"]

    def test_cross_image_psnr_spread(self):
        """PSNR spread across full-range images should be < 3 dB.

        Solver processes each pixel independently, so PSNR should
        be largely image-independent when parameter distributions
        span similar ranges. Midgray is excluded because it maps
        to near-zero perturbation (degenerate case → 39 dB).
        """
        psnr_means = {}
        for name in self.FULL_RANGE_IMAGES:
            image = SYNTHETIC_IMAGES[name]()
            psnr, _, _ = roundtrip_single_image(image)
            psnr_means[name] = psnr.mean

        arr = np.array(list(psnr_means.values()))
        spread = arr.max() - arr.min()

        assert spread < 3.0, (
            f"PSNR spread {spread:.1f} dB >= 3 dB: {psnr_means}"
        )

    def test_midgray_higher_psnr(self):
        """Midgray (near-zero perturbation) should achieve higher PSNR.

        When all pixels map to the center of the parameter range,
        δE ≈ 0, δσ ≈ 0 → Taylor is exact → PSNR >> full-range case.
        """
        midgray_psnr, _, _ = roundtrip_single_image(_make_midgray_image())
        random_psnr, _, _ = roundtrip_single_image(_make_random_image())

        assert midgray_psnr.mean > random_psnr.mean + 10.0, (
            f"Midgray ({midgray_psnr.mean:.1f} dB) should be >> "
            f"random ({random_psnr.mean:.1f} dB)"
        )


class TestEncoderPSNR:
    """Test PSNR computation itself."""

    def test_identical_images(self):
        """Identical images → PSNR = inf."""
        image = _make_gradient_image(8, 8)
        psnr = compute_channel_psnr(image, image)
        assert psnr.amplitude == float("inf")
        assert psnr.shift == float("inf")
        assert psnr.fwhm == float("inf")

    def test_one_lsb_difference(self):
        """1-LSB difference → PSNR ~ 48 dB."""
        image1 = np.full((8, 8, 3), 128, dtype=np.uint8)
        image2 = image1.copy()
        image2[:, :, 0] = 129  # 1 LSB difference in R only

        psnr = compute_channel_psnr(image1, image2)
        assert 45 < psnr.amplitude < 55, f"R: {psnr.amplitude:.1f} dB"
        assert psnr.shift == float("inf")  # G unchanged
        assert psnr.fwhm == float("inf")  # B unchanged


# ---------------------------------------------------------------------------
# Parabola solver tests
# ---------------------------------------------------------------------------


class TestParabolaRoundtrip:
    """Test round-trip with dict2d_parabola solver (Jacobian-free)."""

    @pytest.mark.parametrize("name", list(SYNTHETIC_IMAGES.keys()))
    def test_parabola_runs(self, name: str):
        """Parabola round-trip completes without error."""
        image = SYNTHETIC_IMAGES[name]()
        psnr, gen_time, fit_time = roundtrip_single_image(
            image, solver="parabola"
        )
        assert psnr.amplitude > 0
        assert psnr.shift > 0
        assert psnr.fwhm > 0

    @pytest.mark.parametrize("name", list(SYNTHETIC_IMAGES.keys()))
    def test_parabola_psnr_threshold(self, name: str):
        """Parabola solver should exceed 4-step thresholds on all channels.

        Parabola eliminates Taylor truncation error for δσ recovery,
        so B(FWHM) threshold is raised significantly vs 4-step (3 dB).
        G(δE) threshold is slightly lower than 4-step (19 vs 20 dB)
        because dictionary grid discretization limits δE precision.
        """
        image = SYNTHETIC_IMAGES[name]()
        psnr, _, _ = roundtrip_single_image(image, solver="parabola")

        assert psnr.amplitude > 20.0, (
            f"{name}: R(amp)={psnr.amplitude:.1f} dB < 20 dB"
        )
        assert psnr.shift > 19.0, (
            f"{name}: G(δE)={psnr.shift:.1f} dB < 19 dB"
        )
        # Parabola should lift B(FWHM) well above 4-step's ~5 dB floor
        assert psnr.fwhm > 15.0, (
            f"{name}: B(FWHM)={psnr.fwhm:.1f} dB < 15 dB (parabola)"
        )
        assert psnr.mean > 20.0, (
            f"{name}: mean={psnr.mean:.1f} dB < 20 dB (parabola)"
        )


class TestParabolaConsistency:
    """Parabola solver consistency across images."""

    FULL_RANGE_IMAGES = ["gradient", "random", "stripe"]

    @pytest.mark.skipif(IN_CI or not _mlx_usable(),
                        reason="PSNR-spread tolerance calibrated for the MLX path")
    def test_parabola_cross_image_spread(self):
        """Parabola PSNR spread across images < 5 dB (finite values).

        Parabola can achieve perfect (inf) recovery on simple patterns
        like stripes; spread is computed on finite values only.
        Parabola spread is wider than 4-step because δE grid
        discretization affects different images differently.
        """
        psnr_means = {}
        for name in self.FULL_RANGE_IMAGES:
            image = SYNTHETIC_IMAGES[name]()
            psnr, _, _ = roundtrip_single_image(image, solver="parabola")
            psnr_means[name] = psnr.mean

        arr = np.array(list(psnr_means.values()))
        finite = arr[np.isfinite(arr)]

        if len(finite) < 2:
            return  # All inf → nothing to compare

        spread = finite.max() - finite.min()

        assert spread < 5.0, (
            f"Parabola spread {spread:.1f} dB >= 5 dB: {psnr_means}"
        )


class TestParabolaImprovesFWHM:
    """Verify parabola improves B(FWHM) over 4-step Taylor solver."""

    def test_fwhm_improvement_gradient(self):
        """Parabola B(FWHM) should be significantly higher than 4-step."""
        image = _make_gradient_image()
        psnr_4step, _, _ = roundtrip_single_image(image, solver="4step")
        psnr_para, _, _ = roundtrip_single_image(image, solver="parabola")

        improvement = psnr_para.fwhm - psnr_4step.fwhm
        assert improvement > 5.0, (
            f"Parabola B(FWHM)={psnr_para.fwhm:.1f} dB vs "
            f"4-step={psnr_4step.fwhm:.1f} dB, "
            f"improvement={improvement:.1f} dB < 5 dB"
        )

    def test_fwhm_improvement_random(self):
        """Parabola B(FWHM) improvement on random image."""
        image = _make_random_image()
        psnr_4step, _, _ = roundtrip_single_image(image, solver="4step")
        psnr_para, _, _ = roundtrip_single_image(image, solver="parabola")

        improvement = psnr_para.fwhm - psnr_4step.fwhm
        assert improvement > 5.0, (
            f"Parabola B(FWHM)={psnr_para.fwhm:.1f} dB vs "
            f"4-step={psnr_4step.fwhm:.1f} dB, "
            f"improvement={improvement:.1f} dB < 5 dB"
        )
