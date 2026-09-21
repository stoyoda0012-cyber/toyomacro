"""MLX capability detection: absent / installed-but-unusable / usable.

The package must import and fall back to NumPy whenever MLX cannot
actually execute work (not installed, no usable accelerator, or
disabled via ``TOYOMACRO_DISABLE_MLX=1``), and must raise an actionable
error only when the caller explicitly asserts the GPU path via
``require_mlx()``.

The probe runs on MLX's default device and is device-agnostic, so
neither it nor the errors it feeds may claim that Metal is the only
backend MLX can run on.
"""

import os
import subprocess
import sys

import pytest

from toyomacro.voigtfit import _mlx_support
from toyomacro.voigtfit._mlx_support import (
    LazyCompiled,
    mlx_installed,
    mlx_usable,
    require_mlx,
)


@pytest.fixture
def fresh_cache(monkeypatch):
    """Reset the module-level usable cache around each test."""
    monkeypatch.setattr(_mlx_support, '_USABLE', None)
    yield
    # monkeypatch restores _USABLE automatically


class TestCapabilityDetection:
    def test_usable_implies_installed(self, fresh_cache):
        if mlx_usable():
            assert mlx_installed()

    def test_env_var_disables_mlx(self, fresh_cache, monkeypatch):
        # Simulates "installed but must not be used" — the same code
        # path callers hit on a headless Mac after the probe fails.
        monkeypatch.setenv('TOYOMACRO_DISABLE_MLX', '1')
        assert mlx_usable() is False

    def test_probe_failure_reports_unusable(self, fresh_cache, monkeypatch):
        # Simulate MLX importable but the device probe raising (headless
        # or virtualized host, broken driver/toolchain).
        import builtins
        real_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == 'mlx.core' or name.startswith('mlx'):
                raise RuntimeError('no usable device')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', failing_import)
        assert mlx_usable() is False

    def test_result_is_cached(self, fresh_cache, monkeypatch):
        first = mlx_usable()
        # Poison the env: a cached result must not re-probe.
        monkeypatch.setenv('TOYOMACRO_DISABLE_MLX', '1')
        assert mlx_usable() is first


class TestRequireMlx:
    def test_require_mlx_when_disabled_raises_actionable(
            self, fresh_cache, monkeypatch):
        monkeypatch.setenv('TOYOMACRO_DISABLE_MLX', '1')
        with pytest.raises(RuntimeError, match='NumPy backend'):
            require_mlx()

    def test_require_mlx_passes_when_usable(self, fresh_cache):
        if not mlx_usable():
            pytest.skip('MLX not usable on this host')
        require_mlx()  # must not raise

    def test_messages_do_not_claim_metal_is_the_only_backend(
            self, fresh_cache, monkeypatch):
        """Both failure messages must stay device-neutral.

        ``mlx_usable`` probes the default device, whatever it is — MLX
        also has a CUDA backend (``docs/CUDA_BACKEND_POC.md``). Wording
        that tells the caller MLX is Apple-only, or that the probe looks
        for a Metal device specifically, is factually wrong and sends
        non-Apple users down a dead end.
        """
        monkeypatch.setattr(_mlx_support, 'mlx_usable', lambda: False)
        banned = ('apple silicon only', 'metal device', 'apple-only')

        for installed in (True, False):
            monkeypatch.setattr(
                _mlx_support, 'mlx_installed', lambda: installed)
            with pytest.raises(RuntimeError) as excinfo:
                _mlx_support.require_mlx()
            message = str(excinfo.value).lower()
            for phrase in banned:
                assert phrase not in message, (
                    f'installed={installed}: {phrase!r} in {message!r}')
            # Still actionable: it must name the way out.
            assert 'numpy backend' in message


class TestLazyCompiled:
    def test_no_compile_before_first_call(self):
        calls = []
        lc = LazyCompiled(lambda x: calls.append(x) or x)
        assert lc._compiled is None  # nothing happened at wrap time

    def test_executes_and_caches(self):
        if not mlx_usable():
            pytest.skip('MLX not usable on this host')
        import mlx.core as mx
        lc = LazyCompiled(lambda x: x + 1)
        out = lc(mx.array([1.0]))
        assert float(out[0]) == 2.0
        first = lc._compiled
        lc(mx.array([2.0]))
        assert lc._compiled is first  # compiled once


class TestImportWithMlxDisabled:
    def test_package_imports_and_falls_back(self):
        """Import the full package in a subprocess with MLX force-disabled.

        This simulates the headless-Mac scenario end to end: MLX is
        installed, but no accelerated path may be taken, and the import
        plus a small NumPy-backend fit must succeed.
        """
        code = (
            'import numpy as np\n'
            'import toyomacro.voigtfit as vf\n'
            'assert vf.HAS_MLX is False, "HAS_MLX must be False when disabled"\n'
            'assert vf.mlx_usable() is False\n'
            'from toyomacro.voigtfit.weight_cache import WeightMatrixCache\n'
            'from toyomacro.voigtfit.pipeline import HybridPipeline\n'
            'energy = np.linspace(-5, 5, 101)\n'
            'cache = WeightMatrixCache()\n'
            'Wt, Phi = cache.get_or_create("Si", "2p", energy, '
            'np.array([0.0]), np.array([0.6]), 0.2)\n'
            'pipe = HybridPipeline(cache=cache, use_mlx=True, '
            'enable_stage2=False)\n'
            'assert pipe.use_mlx is False\n'
            '# column-major convention: Y is (n_energy, n_spectra)\n'
            'Y = np.asarray(Phi) @ np.array([[7.0]])\n'
            'A, chi2 = pipe.stage1_screening(Y, Wt, Phi)\n'
            'assert abs(float(np.asarray(A)[0, 0]) - 7.0) < 1e-2\n'
            'print("fallback-ok")\n'
        )
        # Inherit the environment rather than building a bare one: on
        # Windows an interpreter started without USERPROFILE/SYSTEMROOT
        # cannot resolve a home directory, and the import aborts before
        # the fallback under test ever runs.
        out = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True,
            env={**os.environ, 'TOYOMACRO_DISABLE_MLX': '1'},
        )
        assert 'fallback-ok' in out.stdout, out.stderr
