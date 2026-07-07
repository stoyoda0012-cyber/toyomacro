"""Generate Figure 2: GVRT image round-trip across the Poisson noise ladder.

Encodes a synthetic demo image into per-pixel Voigt parameters
(R=amplitude, G=energy shift, B=FWHM), generates one spectrum per
pixel, injects Poisson noise at increasing lambda, fits every spectrum
with the dict2d_parabola solver, and reconstructs the image from the
fitted parameters.  The per-channel PSNR curves document the graceful
degradation ordering: FWHM collapses first, amplitude second, and the
peak position survives longest.

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

# Poisson lambda ladder (spectra_generator.NOISE_LEVELS names)
LADDER = [
    ('None', 0.0),
    ('Subtle', 1e0),
    ('Weak', 1e1),
    ('Small', 1e2),
    ('Moderate', 1e3),
    ('Strong', 1e4),
    ('Intense', 1e5),
    ('Extreme', 1e6),
    ('Maximal', 1e7),
]
# Columns shown as images in the top row (the visible transition:
# pristine -> width softening -> visible degradation -> collapse)
SHOWN = ['None', 'Small', 'Moderate', 'Strong']

# Curve range: one point past collapse documents the floor; beyond
# that the ladder is flat and adds no information.
CURVE = [(n, l) for n, l in LADDER if n not in ('None', 'Extreme', 'Maximal')]

SOLVER = 'dict2d_parabola'
IMAGE_SIZE = 384


def main() -> None:
    service = GVRTService()
    image = make_demo_image(IMAGE_SIZE)

    results = {}
    for name, lam in CURVE + [('None', 0.0)]:
        if name in results:
            continue
        config = GVRTConfig(solver=SOLVER, noise_level=name)
        res = service.run(image, config)
        results[name] = res
        p = res.psnr
        print(f'{name:>9s} (lam={lam:g}): '
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
        lam = dict(LADDER)[name]
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(results[name].reconstructed)
        title = 'noise-free' if lam == 0 else f'$\\lambda=10^{{{int(np.log10(lam))}}}$'
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    # ── Bottom row: PSNR vs lambda for the three channels ───────
    ax = fig.add_subplot(gs[1, :])
    lams = [lam for _, lam in CURVE]
    for attr, label, color in [
        ('shift', 'peak position ($\\delta E$)', '#2ca02c'),
        ('amplitude', 'amplitude ($a$)', '#d62728'),
        ('fwhm', 'width (FWHM)', '#1f77b4'),
    ]:
        vals = [getattr(results[name].psnr, attr) for name, _ in CURVE]
        ax.plot(lams, vals, 'o-', color=color, label=label)
    ax.axvline(1e3, color='gray', ls='--', lw=1)
    ax.text(1.15e3, 45, 'encoded peak amplitude\n($10^3$ counts)',
            fontsize=8, color='gray')
    ax.set_xscale('log')
    ax.set_xlabel('Poisson noise level $\\lambda$')
    ax.set_ylabel('parameter-channel PSNR (dB)')
    ax.legend(fontsize=9, loc='center left')
    ax.grid(alpha=0.3)
    ax.set_title('Degradation order under noise: '
                 'FWHM $\\rightarrow$ amplitude $\\rightarrow$ position',
                 fontsize=10)

    out = Path(__file__).parent / 'figure2_gvrt_noise.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
