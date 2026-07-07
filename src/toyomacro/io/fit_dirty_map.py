"""Dirty map for accumulating interactive fit results before H5 flush.

Accumulates FittingResult objects keyed by (path, spectrum_index).
On flush, writes fitpara/otherpara to H5 files in a single r+ pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from toyomacro.core.fitting_result import FittingResult


@dataclass
class _DirtyEntry:
    """A single pending fit result."""

    fitpara: np.ndarray  # (9, n_components)
    otherpara: np.ndarray  # (10,)


class FitDirtyMap:
    """In-memory accumulator for unsaved single-spectrum fit results.

    Usage::

        dm = FitDirtyMap()
        dm.record(path, idx, fitting_result)  # after ⚡ Fit / LiveFit+AutoSave
        n = dm.dirty_count                    # display "3 unsaved"
        dm.flush(cache)                       # write to H5 & re-open handles
        dm.clear()                            # discard all pending
    """

    def __init__(self) -> None:
        # {path: {idx: _DirtyEntry}}
        self._entries: dict[str, dict[int, _DirtyEntry]] = {}

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record(self, path: str, idx: int, fitting_result: FittingResult) -> None:
        """Add or overwrite a fit result for (path, idx)."""
        fitpara = fitting_result.to_fitpara()  # (9, n_components)
        otherpara = fitting_result.to_otherpara()  # (10,)
        entry = _DirtyEntry(
            fitpara=fitpara.astype(np.float32),
            otherpara=otherpara.astype(np.float32),
        )
        self._entries.setdefault(path, {})[idx] = entry

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def dirty_count(self) -> int:
        """Total number of unsaved fit results across all files."""
        return sum(len(v) for v in self._entries.values())

    @property
    def dirty_files(self) -> list[str]:
        """List of file paths with unsaved results."""
        return [p for p, entries in self._entries.items() if entries]

    def is_dirty(self) -> bool:
        """True if any unsaved results exist."""
        return self.dirty_count > 0

    # ------------------------------------------------------------------
    # Flush to H5
    # ------------------------------------------------------------------

    def flush(self, cache=None) -> int:
        """Write all pending results to H5 files.

        Args:
            cache: HDF5Cache instance.  If provided, file handles are
                   closed before writing and re-opened afterward so the
                   read cache stays valid.

        Returns:
            Number of spectra written.
        """
        if not self.is_dirty():
            return 0

        import h5py

        # Close read handles so we can open r+
        if cache is not None:
            cache.close_files_only()

        total_written = 0

        for path, entries in self._entries.items():
            if not entries:
                continue
            if not Path(path).exists():
                print(f"[DirtyMap] File not found, skipping: {path}")
                continue

            try:
                with h5py.File(path, "r+") as f:
                    has_fitpara = "fitpara" in f
                    has_otherpara = "otherpara" in f

                    if not has_fitpara and "fitpara_compressed" in f:
                        # Decompress fitpara_compressed → fitpara
                        from toyomacro.io.compression import decode_fitpara_from_h5

                        fitpara_full = decode_fitpara_from_h5(f)
                        del f["fitpara_compressed"]
                        f.create_dataset(
                            "fitpara", data=fitpara_full, dtype=np.float32,
                        )
                        has_fitpara = True
                        print(
                            f"[DirtyMap] Decompressed fitpara in {Path(path).name}",
                            flush=True,
                        )
                    elif not has_fitpara:
                        # Auto-create fitpara/otherpara from specdata shape
                        if "specdata" not in f:
                            continue
                        n_spectra = f["specdata"].shape[0] - 1
                        maxcomp = 6
                        f.create_dataset(
                            "fitpara",
                            shape=(n_spectra, maxcomp, 9),
                            dtype=np.float32,
                            fillvalue=np.nan,
                        )
                        if not has_otherpara:
                            f.create_dataset(
                                "otherpara",
                                shape=(10, n_spectra),
                                dtype=np.float32,
                            )
                            has_otherpara = True
                        print(
                            f"[DirtyMap] Created fitpara in {Path(path).name}",
                            flush=True,
                        )

                    fp_ds = f["fitpara"]
                    maxcomp = fp_ds.shape[1]  # (n_spectra, maxcomp, 9)

                    op_ds = f["otherpara"] if has_otherpara else None

                    for idx, entry in entries.items():
                        # Pad fitpara to maxcomp
                        fp = entry.fitpara  # (9, n_comp)
                        n_comp = fp.shape[1]
                        if n_comp < maxcomp:
                            padded = np.full((9, maxcomp), np.nan, dtype=np.float32)
                            padded[:, :n_comp] = fp
                            fp = padded
                        elif n_comp > maxcomp:
                            fp = fp[:, :maxcomp]

                        # fitpara layout: (n_spectra, maxcomp, 9) — need (maxcomp, 9)
                        fp_ds[idx, :, :] = fp.T  # (9, maxcomp).T = (maxcomp, 9)

                        if op_ds is not None and op_ds.shape[0] >= 10:
                            op_ds[:, idx] = entry.otherpara

                        total_written += 1

                print(
                    f"[DirtyMap] Wrote {len(entries)} fits to {Path(path).name}",
                    flush=True,
                )
            except Exception as e:
                print(f"[DirtyMap] Error writing {Path(path).name}: {e}", flush=True)

        # Re-open read handles
        if cache is not None:
            for path in self._entries:
                if Path(path).exists():
                    try:
                        cache._get_file(path)
                    except Exception:
                        pass

        self._entries.clear()
        return total_written

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear(self, path: str | None = None) -> None:
        """Discard pending results.

        Args:
            path: If given, clear only entries for this file.
                  If None, clear everything.
        """
        if path is not None:
            self._entries.pop(path, None)
        else:
            self._entries.clear()
