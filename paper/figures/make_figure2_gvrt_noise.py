"""Generate Figure 2: GVRT image round-trip across the shot-noise ladder.

Encodes a synthetic demo image into per-pixel Voigt parameters
(R=amplitude, G=energy shift, B=FWHM), generates one spectrum per
pixel, injects Poisson shot noise at increasing severity, fits every
spectrum with the dict2d_parabola solver, and reconstructs the image
from the fitted parameters.  The per-channel PSNR curves document the
graceful degradation ordering: FWHM collapses first, amplitude second,
and the peak position survives longest.

Noise semantics
---------------
The generator's dimensionless noise-severity parameter ``level``
(``NOISE_LEVELS`` names) fixes the counting statistics at the signal
maximum:

    SNR_peak    = 1e4 / level        (peak-count signal-to-noise ratio)
    lambda_peak = (1e4 / level)^2    (peak Poisson mean, counts)

The x-axis below is the physically interpretable **peak-count SNR**,
computed via ``spectra_generator.level_to_peak_snr``.  Parameter
recovery collapses as SNR_peak approaches unity — the point where the
peak signal equals its own shot noise.  Poisson sampling is exact for
small means and switches to the Gaussian approximation
``N(lambda, sqrt(lambda))`` for lambda > 20 (error < 1%); see
``spectra_generator.add_poisson_noise``.

Fully self-contained — the demo image is generated in code, so no
external image asset (or its license) is required.

Usage:
    python paper/figures/make_figure2_gvrt_noise.py
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from toyomacro.voigtfit.gvrt_service import (
    GVRTConfig,
    GVRTService,
    make_demo_image,
)
from toyomacro.voigtfit.spectra_generator import (
    NOISE_LEVELS,
    level_to_peak_snr,
)

# Noise-severity ladder (spectra_generator.NOISE_LEVELS names).
LADDER = ['None', 'Subtle', 'Weak', 'Small', 'Moderate', 'Strong', 'Intense']

# Columns shown as images in the top row (the visible transition:
# pristine -> width softening -> visible degradation -> collapse)
SHOWN = ['None', 'Small', 'Moderate', 'Strong']

# Curve range: one point past collapse documents the floor; beyond
# that the ladder is flat and adds no information.
CURVE = [n for n in LADDER if n != 'None']

SOLVER = 'dict2d_parabola'
IMAGE_SIZE = 384


def _snr_label(name: str) -> str:
    """Human-readable peak-SNR label for a noise-level name."""
    snr = level_to_peak_snr(NOISE_LEVELS[name])
    if np.isinf(snr):
        return 'noise-free'
    exp = int(round(np.log10(snr)))
    return f'$\\mathrm{{SNR_{{peak}}}}=10^{{{exp}}}$'


def main() -> None:
    service = GVRTService()
    image = make_demo_image(IMAGE_SIZE)

    results = {}
    for name in LADDER:
        config = GVRTConfig(solver=SOLVER, noise_level=name)
        res = service.run(image, config)
        results[name] = res
        p = res.psnr
        snr = level_to_peak_snr(NOISE_LEVELS[name])
        print(f'{name:>9s} (level={NOISE_LEVELS[name]:g}, '
              f'peak SNR={snr:g}): '
              f'PSNR amp={p.amplitude:.1f} dE={p.shift:.1f} '
              f'FWHM={p.fwhm:.1f} dB  '
              f'({res.throughput / 1e6:.2f} M spec/s)')

    n_img = 1 + len(SHOWN)
    fig = plt.figure(figsize=(2.0 * n_img, 4.6))
    gs = fig.add_gridspec(2, n_img, height_ratios=[1, 1.1], hspace=0.32)

    # ── Top row: original + reconstructions ─────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image)
    ax.set_title('Original', fontsize=9)
    ax.axis('off')
    for i, name in enumerate(SHOWN, start=1):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(results[name].reconstructed)
        ax.set_title(_snr_label(name), fontsize=9)
        ax.axis('off')

    # ── Bottom row: PSNR vs peak-count SNR for the three channels ─
    ax = fig.add_subplot(gs[1, :])
    snrs = [level_to_peak_snr(NOISE_LEVELS[n]) for n in CURVE]
    for attr, label, color in [
        ('shift', 'peak position ($\\delta E$)', '#2ca02c'),
        ('amplitude', 'amplitude ($a$)', '#d62728'),
        ('fwhm', 'width (FWHM)', '#1f77b4'),
    ]:
        vals = [getattr(results[name].psnr, attr) for name in CURVE]
        ax.plot(snrs, vals, 'o-', color=color, label=label)
    ax.axvline(1.0, color='gray', ls='--', lw=1)
    ax.text(1.3, 45, 'peak signal = shot noise\n($\\mathrm{SNR_{peak}}=1$)',
            fontsize=8, color='gray')
    ax.set_xscale('log')
    ax.invert_xaxis()  # noise increases to the right
    ax.set_xlabel('peak-count SNR ($=10^4/\\mathrm{level}$; '
                  'noise increases $\\rightarrow$)')
    ax.set_ylabel('parameter-channel PSNR (dB)')
    ax.legend(fontsize=9, loc='center left')
    ax.grid(alpha=0.3)
    ax.set_title('Degradation order under shot noise: '
                 'FWHM $\\rightarrow$ amplitude $\\rightarrow$ position',
                 fontsize=10)

    out = Path(__file__).parent / 'figure2_gvrt_noise.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
