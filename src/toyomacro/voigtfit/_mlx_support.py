"""MLX capability detection and lazy compilation helpers.

Distinguishes three states:

1. **MLX absent** — ``import mlx`` fails; NumPy backend everywhere.
2. **MLX installed but unusable** — the package imports but the default
   device cannot execute work (headless/virtualized host, a broken
   driver or toolchain, or disabled via ``TOYOMACRO_DISABLE_MLX=1``).
   NumPy backend everywhere.
3. **MLX usable** — a probe kernel actually ran on the device.

The probe is **device-agnostic**: it runs on whatever MLX reports as the
default device, so it answers "can MLX execute work here", not "is this
a Metal device".  Apple Metal is the backend this package distributes
and publishes benchmark results for; MLX's CUDA backend also satisfies
the probe and has been validated once out of tree (see
``docs/CUDA_BACKEND_POC.md``), but it is not a supported installation
target and carries its own caveats.

Modules must consult :func:`mlx_usable` (not a bare ``import mlx``
success) before routing work to the GPU, and must never call
``mx.compile`` at import time — wrap kernels in :class:`LazyCompiled`
instead so compilation happens at the first accelerated call.
"""

from __future__ import annotations

import os

_USABLE: bool | None = None


def mlx_installed() -> bool:
    """True if the ``mlx`` package can be imported at all."""
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def mlx_usable() -> bool:
    """True if MLX can actually execute work on this machine.

    Runs (once, cached) a tiny probe kernel on MLX's **default device**,
    whichever accelerator that is.  An installed MLX that cannot execute
    there — e.g. a headless or virtualized CI runner — returns False,
    and callers fall back to the NumPy backend.  Set
    ``TOYOMACRO_DISABLE_MLX=1`` to force False without uninstalling MLX.
    """
    global _USABLE
    if _USABLE is None:
        if os.environ.get('TOYOMACRO_DISABLE_MLX', '').strip().lower() in (
                '1', 'true', 'yes'):
            _USABLE = False
        else:
            try:
                import mlx.core as mx
                mx.eval(mx.zeros((1,), dtype=mx.float32) + 1)
                _USABLE = True
            except Exception:
                _USABLE = False
    return _USABLE


def require_mlx() -> None:
    """Raise an actionable error when MLX-only behavior was requested."""
    if mlx_usable():
        return
    if not mlx_installed():
        raise RuntimeError(
            'MLX backend requested but the mlx package is not installed. '
            'Install the MLX build for your platform — this package '
            'distributes the Apple Silicon one as `pip install '
            'toyomacro[mlx]` — or use the NumPy backend (use_mlx=False).')
    raise RuntimeError(
        'MLX is installed but could not execute work on the default '
        'device, so the accelerated path is unavailable. Use the NumPy '
        'backend (use_mlx=False), or unset TOYOMACRO_DISABLE_MLX if it '
        'was set.')


class LazyCompiled:
    """Defer ``mx.compile`` until the wrapped kernel is first called.

    Keeps module import free of device/compiler work so that importing
    the package succeeds on machines where MLX is installed but not
    usable.
    """

    def __init__(self, fn):
        self._fn = fn
        self._compiled = None

    def __call__(self, *args, **kwargs):
        if self._compiled is None:
            import mlx.core as mx
            self._compiled = mx.compile(self._fn)
        return self._compiled(*args, **kwargs)
