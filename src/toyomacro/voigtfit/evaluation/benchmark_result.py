"""BenchmarkResult: per-pixel parameter storage for error map visualization.

Extends the summary-level ParamRoundtripResult with full per-pixel arrays
needed for spatial error analysis.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BenchmarkResult:
    """Per-pixel benchmark result for a single solver × noise combination.

    Stores both estimated and true parameter arrays for spatial error analysis.
    """
    solver_name: str
    noise_level: str

    # Estimated parameters (n_pixels,) for single-component
    amp_est: np.ndarray
    dE_est: np.ndarray
    dsigma_est: np.ndarray

    # True parameters (same shape)
    amp_true: np.ndarray
    dE_true: np.ndarray
    dsigma_true: np.ndarray

    # Image metadata
    image_shape: tuple[int, int]  # (H, W)

    # Performance
    throughput: float = 0.0  # spectra/sec

    # Adaptive-specific: which pixels used 4-step refinement
    phase2_mask: np.ndarray | None = None  # (n_pixels,) bool

    @property
    def n_pixels(self) -> int:
        return self.amp_est.shape[0]

    def save_npz(self, path: str | Path) -> None:
        """Save to compressed NPZ."""
        data = {
            'solver_name': np.array(self.solver_name),
            'noise_level': np.array(self.noise_level),
            'amp_est': self.amp_est.astype(np.float32),
            'dE_est': self.dE_est.astype(np.float32),
            'dsigma_est': self.dsigma_est.astype(np.float32),
            'amp_true': self.amp_true.astype(np.float32),
            'dE_true': self.dE_true.astype(np.float32),
            'dsigma_true': self.dsigma_true.astype(np.float32),
            'image_shape': np.array(self.image_shape),
            'throughput': np.array(self.throughput),
        }
        if self.phase2_mask is not None:
            data['phase2_mask'] = self.phase2_mask.astype(np.bool_)
        np.savez_compressed(str(path), **data)

    @classmethod
    def load_npz(cls, path: str | Path) -> 'BenchmarkResult':
        """Load from NPZ."""
        d = np.load(str(path), allow_pickle=True)
        phase2_mask = None
        if 'phase2_mask' in d:
            phase2_mask = d['phase2_mask'].astype(np.bool_)
        return cls(
            solver_name=str(d['solver_name']),
            noise_level=str(d['noise_level']),
            amp_est=d['amp_est'],
            dE_est=d['dE_est'],
            dsigma_est=d['dsigma_est'],
            amp_true=d['amp_true'],
            dE_true=d['dE_true'],
            dsigma_true=d['dsigma_true'],
            image_shape=tuple(d['image_shape']),
            throughput=float(d['throughput']),
            phase2_mask=phase2_mask,
        )


def save_multi_results(
    results: dict[str, dict[str, BenchmarkResult]],
    path: str | Path,
) -> None:
    """Save nested dict results[noise_level][solver_name] to a single NPZ.

    Keys are encoded as '{noise_level}__{solver_name}__{field}'.
    """
    data = {}
    noise_levels = []
    solver_names = set()

    for noise_level, solvers in results.items():
        noise_levels.append(noise_level)
        for solver_name, br in solvers.items():
            solver_names.add(solver_name)
            prefix = f'{noise_level}__{solver_name}'
            data[f'{prefix}__amp_est'] = br.amp_est.astype(np.float32)
            data[f'{prefix}__dE_est'] = br.dE_est.astype(np.float32)
            data[f'{prefix}__dsigma_est'] = br.dsigma_est.astype(np.float32)
            data[f'{prefix}__amp_true'] = br.amp_true.astype(np.float32)
            data[f'{prefix}__dE_true'] = br.dE_true.astype(np.float32)
            data[f'{prefix}__dsigma_true'] = br.dsigma_true.astype(np.float32)
            data[f'{prefix}__image_shape'] = np.array(br.image_shape)
            data[f'{prefix}__throughput'] = np.array(br.throughput)
            if br.phase2_mask is not None:
                data[f'{prefix}__phase2_mask'] = br.phase2_mask.astype(np.bool_)

    data['__noise_levels'] = np.array(list(dict.fromkeys(noise_levels)))
    data['__solver_names'] = np.array(sorted(solver_names))
    np.savez_compressed(str(path), **data)


def load_multi_results(
    path: str | Path,
) -> dict[str, dict[str, BenchmarkResult]]:
    """Load nested dict results from NPZ."""
    d = np.load(str(path), allow_pickle=True)
    noise_levels = list(d['__noise_levels'])
    solver_names = list(d['__solver_names'])

    results = {}
    for nl in noise_levels:
        nl = str(nl)
        results[nl] = {}
        for sn in solver_names:
            sn = str(sn)
            prefix = f'{nl}__{sn}'
            key = f'{prefix}__amp_est'
            if key not in d:
                continue
            phase2_mask = None
            pm_key = f'{prefix}__phase2_mask'
            if pm_key in d:
                phase2_mask = d[pm_key].astype(np.bool_)
            results[nl][sn] = BenchmarkResult(
                solver_name=sn,
                noise_level=nl,
                amp_est=d[f'{prefix}__amp_est'],
                dE_est=d[f'{prefix}__dE_est'],
                dsigma_est=d[f'{prefix}__dsigma_est'],
                amp_true=d[f'{prefix}__amp_true'],
                dE_true=d[f'{prefix}__dE_true'],
                dsigma_true=d[f'{prefix}__dsigma_true'],
                image_shape=tuple(d[f'{prefix}__image_shape']),
                throughput=float(d[f'{prefix}__throughput']),
                phase2_mask=phase2_mask,
            )
    return results
