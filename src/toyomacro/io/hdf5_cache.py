"""
HDF5 file cache for fast XPS spectrum access.

Provides sub-millisecond read performance by keeping files open
and caching energy axes. Used by both GUI and API data browsers.

Usage:
    cache = HDF5Cache()
    cache.scan_folder('/path/to/folder')

    spectrum = cache.get_spectrum('/path/to/file.h5', index=1000)
    multi = cache.get_multi_spectrum(['/path/to/Si2p.h5', '/path/to/O1s.h5'], index=1000)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

# Try MLX for GPU acceleration
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class FileMetadata:
    """Metadata for a cached HDF5 file."""

    path: str
    filename: str
    element: str
    n_spectra: int
    n_energy: int
    size_mb: float
    has_fitpara: bool = False
    source_file: str = ""  # Source file name (e.g. "1804150013.vms")
    energy_type: str = "BE"  # "BE" or "KE", from misc/bindingenergysign


@dataclass
class Coordinates4D:
    """4D coordinates for spectrum navigation."""

    x: int
    y: int
    t: int
    angle_idx: int
    angle_deg: float = 0.0


@dataclass
class SpectrumData:
    """Single spectrum data with coordinates."""

    energy: NDArray[np.float32]
    intensity: NDArray[np.float32]
    index: int
    coordinates: Coordinates4D
    element: str
    read_time_ms: float
    energy_type: str = "BE"


@dataclass
class MultiSpectrumData:
    """Multi-element spectrum data at same coordinates."""

    spectra: dict[str, NDArray[np.float32]]  # element -> intensity
    energies: dict[str, NDArray[np.float32]]  # element -> energy
    index: int
    coordinates: Coordinates4D
    read_time_ms: float
    energy_type: str = "BE"


@dataclass
class DimensionConfig:
    """4D dimension configuration."""

    n_x: int = 100
    n_y: int = 100
    n_t: int = 200
    n_angles: int = 14
    dim0_is_angle: bool = True  # True→θ label, False→X label


def decode_index(
    idx: int,
    n_angles: int = 14,
    n_y: int = 100,
    n_x: int = 100,
) -> tuple[int, int, int, int]:
    """
    Decode flat index to 4D coordinates (X, Y, T, angle_idx).

    Index structure: idx = angle + n_angles*Y + n_angles*n_y*X + n_angles*n_y*n_x*T

    Returns:
        (x, y, t, angle_idx)
    """
    angle_idx = idx % n_angles
    remainder = idx // n_angles
    y = remainder % n_y
    remainder = remainder // n_y
    x = remainder % n_x
    t = remainder // n_x
    return x, y, t, angle_idx


def encode_index(
    x: int,
    y: int,
    t: int,
    angle_idx: int,
    n_angles: int = 14,
    n_y: int = 100,
    n_x: int = 100,
) -> int:
    """
    Encode 4D coordinates to flat index.

    Index structure: idx = angle + n_angles*Y + n_angles*n_y*X + n_angles*n_y*n_x*T
    """
    return angle_idx + n_angles * y + n_angles * n_y * x + n_angles * n_y * n_x * t


# ---------------------------------------------------------------------------
# Element name utilities
# ---------------------------------------------------------------------------

_ELEMENT_LABEL_RE = re.compile(r"([A-Z][a-z]?)(\d[spdfg]\d?/?[\d/]*)")


@dataclass
class ElementInfo:
    """Element identity parsed from filename/metadata.

    Bridges FileMetadata.element ("Si2p") and deppro ElementConfig
    by splitting the label into symbol + orbital and attaching a
    unique *dedup_key* for multi-file scenarios (e.g. "O1s", "O1s_2").
    """

    label: str  # "Si2p", "O1s" — display name (= FileMetadata.element)
    symbol: str  # "Si", "O"
    orbital: str  # "2p", "1s"
    dedup_key: str  # "Si2p", "O1s_2" — unique key after dedup


def parse_element_label(label: str) -> tuple[str, str]:
    """Parse element label into (symbol, orbital).

    Examples:
        >>> parse_element_label("Si2p")
        ('Si', '2p')
        >>> parse_element_label("O1s")
        ('O', '1s')
        >>> parse_element_label("Ta4f")
        ('Ta', '4f')
        >>> parse_element_label("C1s")
        ('C', '1s')
        >>> parse_element_label("unknown")
        ('unknown', '')
    """
    m = _ELEMENT_LABEL_RE.match(label)
    if m:
        return m.group(1), m.group(2)
    return label, ""


def dedup_names(base_names: list[str]) -> list[str]:
    """Deduplicate a list of names by appending numeric suffixes.

    First occurrence keeps the base name; subsequent ones get a
    numeric suffix: ``O1s``, ``O1s_2``, ``O1s_3``, …

    This is the core algorithm used by :func:`dedup_element_names` and
    can be called directly when an :class:`HDF5Cache` is not available
    (e.g. in the FastAPI routes).

    Args:
        base_names: Ordered list of (possibly duplicate) names.

    Returns:
        List of unique names (same length / order as *base_names*).
    """
    result: list[str] = []
    counts: dict[str, int] = {}
    for name in base_names:
        if name in counts:
            counts[name] += 1
            result.append(f"{name}_{counts[name]}")
        else:
            counts[name] = 1
            result.append(name)
    return result


def dedup_element_names(
    paths: list[str],
    cache: HDF5Cache,
) -> list[str]:
    """Deduplicate element names across multiple HDF5 files.

    Convenience wrapper around :func:`dedup_names` that resolves
    base element names via *cache* metadata.

    Args:
        paths: Ordered list of HDF5 file paths.
        cache: HDF5Cache with metadata already scanned.

    Returns:
        List of unique element names (same length / order as *paths*).
    """
    base_names = []
    for path in paths:
        meta = cache.get_metadata(path)
        base_names.append(meta.element if meta else Path(path).stem)
    return dedup_names(base_names)


def build_element_infos(
    paths: list[str],
    cache: HDF5Cache,
) -> list[ElementInfo]:
    """Build :class:`ElementInfo` list with dedup keys.

    Combines :func:`dedup_element_names` with :func:`parse_element_label`
    to produce rich element metadata suitable for DepthProfiler integration.
    """
    dedup_names = dedup_element_names(paths, cache)
    infos: list[ElementInfo] = []
    for path, dedup_key in zip(paths, dedup_names):
        meta = cache.get_metadata(path)
        label = meta.element if meta else Path(path).stem
        symbol, orbital = parse_element_label(label)
        infos.append(
            ElementInfo(
                label=label,
                symbol=symbol,
                orbital=orbital,
                dedup_key=dedup_key,
            )
        )
    return infos


class HDF5Cache:
    """
    Thread-safe HDF5 file cache for fast spectrum access.

    Keeps file handles open and caches energy axes for sub-ms reads.
    """

    def __init__(self, dims: DimensionConfig | None = None):
        """
        Initialize HDF5 cache.

        Args:
            dims: 4D dimension configuration (default: 100x100x200x14)
        """
        self.dims = dims or DimensionConfig()

        # Cache storage
        self._file_cache: dict[str, h5py.File] = {}
        self._specdata_cache: dict[str, h5py.Dataset] = {}
        self._energy_cache: dict[str, NDArray[np.float32]] = {}
        self._metadata_cache: dict[str, FileMetadata] = {}
        self._dims_cache: dict[str, DimensionConfig] = {}  # per-file dimension config
        self._angle_values_cache: dict[str, NDArray[np.float32]] = {}  # per-file angle values from xytdata[0]
        self._mmap_cache: dict[str, np.memmap] = {}  # path -> memmap of specdata
        self._is_uint16: dict[str, bool] = {}  # path -> True if specdata_uint16 format
        self._energy_type_cache: dict[str, str] = {}  # path -> "BE" or "KE"

    def scan_folder(
        self,
        folder_path: str | Path,
        auto_import: bool = False,
    ) -> list[FileMetadata]:
        """
        Scan a folder for HDF5 files with specdata.

        When *auto_import* is True, raw XPS files (.pxt, .ibw, .txt, .vms,
        .npl) are also discovered and transparently converted to HDF5 via
        ``ensure_h5()`` before being included in the results.

        Args:
            folder_path: Path to folder containing HDF5 / raw files
            auto_import: If True, auto-convert raw files to HDF5

        Returns:
            List of FileMetadata for valid files
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Invalid folder: {folder_path}")

        # Find H5 files (prioritize element-named files)
        h5_files = (
            list(folder.glob("*1s*.h5"))
            + list(folder.glob("*2p*.h5"))
            + list(folder.glob("*.h5"))
        )
        h5_files = sorted(set(h5_files))

        # Auto-import raw files
        if auto_import:
            from toyomacro.io.importer import RAW_EXTENSIONS, ensure_h5

            raw_files: list[Path] = []
            for ext in RAW_EXTENSIONS:
                raw_files.extend(folder.glob(f"*{ext}"))
            raw_files = sorted(set(raw_files))

            for raw_file in raw_files:
                # Skip .txt files that are probably not XPS data
                if raw_file.suffix.lower() == ".txt":
                    from toyomacro.io.readers.ses_reader import SESTxtReader
                    if not SESTxtReader.can_read(raw_file):
                        continue

                try:
                    converted = ensure_h5(raw_file, compress=False)
                    for h5_path in converted:
                        if h5_path not in h5_files:
                            h5_files.append(h5_path)
                except Exception as e:
                    print(f"Auto-import failed for {raw_file}: {e}")

            h5_files = sorted(set(h5_files))

        files = []
        for h5_file in h5_files:
            meta = self._scan_file(h5_file)
            if meta:
                files.append(meta)

        return files

    def _scan_file(self, filepath: Path) -> FileMetadata | None:
        """Scan a single HDF5 file and return metadata."""
        path_str = str(filepath)

        if path_str in self._metadata_cache:
            return self._metadata_cache[path_str]

        try:
            with h5py.File(filepath, "r") as f:
                # Auto-detect format: uint16 compressed or original float32
                if "specdata_uint16" in f:
                    specdata = f["specdata_uint16"]
                    n_spectra = specdata.shape[0]  # No energy row in uint16
                    n_energy = specdata.shape[1]
                elif "specdata" in f:
                    specdata = f["specdata"]
                    n_spectra = specdata.shape[0] - 1  # Row 0 is energy
                    n_energy = specdata.shape[1]
                else:
                    return None
                size_mb = filepath.stat().st_size / 1e6
                has_fitpara = "fitpara" in f or "fitpara_compressed" in f

                # Extract element name from filename
                stem = filepath.stem
                if "_" in stem:
                    element = stem.split("_")[0]
                else:
                    element = stem

                # Read source_file from misc (written by importer)
                source_file = ""
                if "misc" in f and "source_file" in f["misc"]:
                    raw = f["misc"]["source_file"][()]
                    if isinstance(raw, bytes):
                        source_file = raw.decode("utf-8")
                    else:
                        source_file = str(raw)
                if not source_file:
                    # Use parent folder name for grouping (more readable
                    # than the inferred file suffix like "260308OCAlTiSi")
                    source_file = filepath.parent.name

                # Read energy type from misc/bindingenergysign
                energy_type = "BE"
                try:
                    misc = f.get("misc")
                    if misc is not None and "bindingenergysign" in misc:
                        val = int(misc["bindingenergysign"][()])
                        energy_type = "BE" if val == 1 else "KE"
                except Exception:
                    pass

                metadata = FileMetadata(
                    path=path_str,
                    filename=filepath.name,
                    element=element,
                    n_spectra=n_spectra,
                    n_energy=n_energy,
                    size_mb=round(size_mb, 2),
                    has_fitpara=has_fitpara,
                    source_file=source_file,
                    energy_type=energy_type,
                )
                self._metadata_cache[path_str] = metadata
                return metadata

        except Exception as e:
            print(f"Error scanning {filepath}: {e}")
            return None

    @staticmethod
    def _infer_source_file(h5_filename: str) -> str:
        """Infer source file stem from H5 filename (fallback for old files).

        Naming convention: ``{element}_{source_stem}.h5``

        Examples:
            "C1s_20250806_Fixed_Narrow_0001.h5" → "20250806_Fixed_Narrow_0001"
            "XPS1_1804150013.h5"                → "1804150013"
            "O1s.h5"                            → "" (no underscore)

        Same-source H5 files share the suffix, so grouping works even when
        the inferred value differs from the actual source filename.
        """
        stem = Path(h5_filename).stem
        if "_" not in stem:
            return ""
        parts = stem.split("_", 1)
        return parts[1] if len(parts) > 1 else ""

    def _infer_dims_from_xytdata(
        self,
        f: h5py.File,
        path: str,
        misc: h5py.Group,
    ) -> DimensionConfig | None:
        """Infer DimensionConfig from numberofslice + xytdata when dim_shape is absent.

        Uses:
          - ``misc/numberofslice`` → n_angles
          - ``xytdata[2, -1]``    → n_t  (integer time steps starting from 1)
          - n_spatial = n_spectra / (n_angles * n_t), sqrt → n_x = n_y
        """
        try:
            n_angles = int(misc["numberofslice"][0])
            if n_angles <= 0:
                return None

            n_spectra = self._n_spectra(path)
            if n_spectra <= 0:
                return None

            xyt = f.get("xytdata")
            if xyt is None or xyt.shape[1] == 0:
                return None

            # Read last T value (fast — single scalar read)
            n_t = max(1, int(round(float(xyt[2, -1]))))

            n_pos = n_spectra // n_angles
            if n_t > 0 and n_pos % n_t == 0:
                n_spatial = n_pos // n_t
            else:
                n_spatial = n_pos
                n_t = 1

            # Try perfect square for spatial dims
            import math
            n_side = int(round(math.sqrt(n_spatial)))
            if n_side * n_side == n_spatial:
                n_x = n_y = n_side
            else:
                # Non-square: treat as 1D
                n_x = n_spatial
                n_y = 1

            return DimensionConfig(
                n_angles=n_angles, n_y=n_y, n_x=n_x, n_t=n_t
            )
        except Exception:
            return None

    @staticmethod
    def _dims_from_dim_shape(
        dim_shape: list[int], n_spectra: int
    ) -> DimensionConfig:
        """Build DimensionConfig from dim_shape metadata.

        Mapping:
          [n_angle]             -> n_angles=n_angle, n_y=n_spectra//n_angle, n_x=1, n_t=1
          [n_angle, n_pos]      -> n_angles=n_angle, n_y=n_pos, n_x=1, n_t=1
          [n_angle, n_y, n_t]   -> n_angles=n_angle, n_y=n_y,  n_x=1, n_t=n_t
        """
        if len(dim_shape) == 1:
            n_angle = dim_shape[0]
            n_y = max(1, n_spectra // n_angle) if n_angle > 0 else 1
            return DimensionConfig(n_angles=n_angle, n_y=n_y, n_x=1, n_t=1)
        elif len(dim_shape) == 2:
            n_angle, n_pos = dim_shape
            return DimensionConfig(n_angles=n_angle, n_y=n_pos, n_x=1, n_t=1)
        elif len(dim_shape) >= 3:
            n_angle, n_y, n_t = dim_shape[0], dim_shape[1], dim_shape[2]
            return DimensionConfig(n_angles=n_angle, n_y=n_y, n_x=1, n_t=n_t)
        else:
            return DimensionConfig()

    def get_dims(self, path: str) -> DimensionConfig:
        """Get dimension config for a file.

        Returns per-file DimensionConfig from ``/misc/dim_shape`` if
        available, otherwise falls back to the global default.
        """
        if path in self._dims_cache:
            return self._dims_cache[path]
        return self.dims

    def get_angle_values(self, path: str) -> NDArray[np.float32] | None:
        """Get cached dim0 values (angle or spatial) for a file.

        Returns an array of length ``n_angles`` containing the real
        angle values (degrees) when ``dim0_is_angle`` is True, or
        positional indices when False.  Returns None for old files
        without xytdata.
        """
        return self._angle_values_cache.get(path)

    def _resolve_angle_deg(
        self, path: str, angle_idx: int, dims: DimensionConfig
    ) -> float:
        """Resolve angle_deg from cached xytdata values or fallback formula."""
        vals = self._angle_values_cache.get(path)
        if vals is not None and 0 <= angle_idx < len(vals):
            return float(vals[angle_idx])
        # Fallback for old files: hardcoded 60° span
        if dims.n_angles > 1:
            return angle_idx * 60.0 / (dims.n_angles - 1)
        return 0.0

    def _get_file(
        self, path: str
    ) -> tuple[h5py.File, h5py.Dataset, NDArray[np.float32]]:
        """Get file, specdata, and energy from cache or open."""
        if path not in self._file_cache:
            if not Path(path).exists():
                raise FileNotFoundError(f"File not found: {path}")

            self._file_cache[path] = h5py.File(path, "r", swmr=False)
            f = self._file_cache[path]

            # Auto-detect format: uint16 compressed or original float32
            if "specdata_uint16" in f:
                self._specdata_cache[path] = f["specdata_uint16"]
                self._energy_cache[path] = f["specdata_uint16_energy"][:].astype(
                    np.float32
                )
                self._is_uint16[path] = True
                # No mmap for uint16 (chunked layout)
            elif "specdata" in f:
                self._specdata_cache[path] = f["specdata"]
                self._energy_cache[path] = self._specdata_cache[path][0, :].astype(
                    np.float32
                )
                self._is_uint16[path] = False

                # Setup mmap for contiguous float32 datasets (bypasses h5py for speed)
                sd = self._specdata_cache[path]
                if sd.dtype == np.float32 and sd.chunks is None:
                    try:
                        offset = sd.id.get_offset()
                        if offset is not None:
                            self._mmap_cache[path] = np.memmap(
                                path, dtype=np.float32, mode="r",
                                offset=offset, shape=sd.shape,
                            )
                    except Exception:
                        pass  # Fall back to h5py reads
            else:
                self._file_cache[path].close()
                del self._file_cache[path]
                raise ValueError(f"Invalid file format: no specdata in {path}")

            # Read dim_shape + dim0_is_angle for per-file DimensionConfig
            if path not in self._dims_cache:
                try:
                    misc = f.get("misc")
                    if misc is not None and "dim_shape" in misc:
                        dim_shape = list(misc["dim_shape"][:])
                        n_spectra = self._n_spectra(path)
                        dims = self._dims_from_dim_shape(dim_shape, n_spectra)
                        # Read dim0_is_angle flag (default True for backward compat)
                        if "dim0_is_angle" in misc:
                            dims.dim0_is_angle = bool(
                                int(misc["dim0_is_angle"][0])
                            )
                        self._dims_cache[path] = dims
                    elif misc is not None and "numberofslice" in misc:
                        # Infer dimensions from numberofslice + xytdata
                        dims = self._infer_dims_from_xytdata(f, path, misc)
                        if dims is not None:
                            self._dims_cache[path] = dims
                except Exception:
                    pass  # Fall back to global default

            # Read energy type from misc/bindingenergysign
            if path not in self._energy_type_cache:
                try:
                    misc = f.get("misc")
                    if misc is not None and "bindingenergysign" in misc:
                        val = int(misc["bindingenergysign"][()])
                        self._energy_type_cache[path] = "BE" if val == 1 else "KE"
                    else:
                        self._energy_type_cache[path] = "BE"
                except Exception:
                    self._energy_type_cache[path] = "BE"

            # Cache angle values from xytdata[0] for real angle display
            if path not in self._angle_values_cache:
                try:
                    if "xytdata" in f:
                        dims = self._dims_cache.get(path, self.dims)
                        n_dim0 = dims.n_angles
                        if n_dim0 > 0:
                            # First n_dim0 values of xytdata[0] = one cycle of angle/x values
                            raw = f["xytdata"][0, :n_dim0].astype(np.float32)
                            self._angle_values_cache[path] = raw
                except Exception:
                    pass

        return (
            self._file_cache[path],
            self._specdata_cache[path],
            self._energy_cache[path],
        )

    def get_spectrum(self, path: str, index: int) -> SpectrumData:
        """
        Get a single spectrum by flat index.

        Args:
            path: Path to HDF5 file
            index: Flat spectrum index (0-based)

        Returns:
            SpectrumData with energy, intensity, and coordinates
        """
        t0 = time.perf_counter()

        f, specdata, energy = self._get_file(path)
        row_offset = self._row_offset(path)
        n_spectra = self._n_spectra(path)

        if index >= n_spectra:
            raise IndexError(f"Index {index} out of range (max: {n_spectra - 1})")

        # Read spectrum (row offset: +1 for original format, +0 for uint16)
        if path in self._mmap_cache:
            intensity = np.array(self._mmap_cache[path][index + row_offset, :])
        else:
            intensity = specdata[index + row_offset, :].astype(np.float32)

        t1 = time.perf_counter()
        read_time_ms = (t1 - t0) * 1000

        # Decode coordinates using per-file dims
        dims = self.get_dims(path)
        x, y, t, angle_idx = decode_index(
            index, dims.n_angles, dims.n_y, dims.n_x
        )
        angle_deg = self._resolve_angle_deg(path, angle_idx, dims)

        # Get element from metadata
        meta = self._metadata_cache.get(path)
        element = meta.element if meta else Path(path).stem

        return SpectrumData(
            energy=energy,
            intensity=intensity,
            index=index,
            coordinates=Coordinates4D(
                x=x,
                y=y,
                t=t,
                angle_idx=angle_idx,
                angle_deg=round(angle_deg, 1),
            ),
            element=element,
            read_time_ms=round(read_time_ms, 3),
            energy_type=self._energy_type_cache.get(path, "BE"),
        )

    def get_spectrum_by_coord(
        self, path: str, x: int, y: int, t: int, angle_idx: int
    ) -> SpectrumData:
        """
        Get a spectrum by 4D coordinates.

        Args:
            path: Path to HDF5 file
            x, y, t, angle_idx: 4D coordinates

        Returns:
            SpectrumData
        """
        dims = self.get_dims(path)
        index = encode_index(
            x, y, t, angle_idx, dims.n_angles, dims.n_y, dims.n_x
        )
        return self.get_spectrum(path, index)

    def get_multi_spectrum(
        self, paths: list[str], index: int
    ) -> MultiSpectrumData:
        """
        Get spectra from multiple files at the same index.

        Used for multi-element display.

        Args:
            paths: List of HDF5 file paths
            index: Flat spectrum index

        Returns:
            MultiSpectrumData with spectra for all elements
        """
        t0 = time.perf_counter()

        spectra: dict[str, NDArray[np.float32]] = {}
        energies: dict[str, NDArray[np.float32]] = {}
        element_names = dedup_element_names(paths, self)

        for path, element in zip(paths, element_names):
            f, specdata, energy = self._get_file(path)
            row_offset = self._row_offset(path)
            n_spectra = self._n_spectra(path)

            # Clamp index to the file's valid range so files with fewer
            # spectra still display their last valid spectrum.
            file_index = min(index, n_spectra - 1)

            if path in self._mmap_cache:
                intensity = np.array(self._mmap_cache[path][file_index + row_offset, :])
            else:
                intensity = specdata[file_index + row_offset, :].astype(np.float32)

            spectra[element] = intensity
            energies[element] = energy

        t1 = time.perf_counter()
        read_time_ms = (t1 - t0) * 1000

        # Decode coordinates using first file's dims (all files in a
        # multi-element set share the same spatial grid)
        first_path = paths[0] if paths else ""
        dims = self.get_dims(first_path) if paths else self.dims
        x, y, t, angle_idx = decode_index(
            index, dims.n_angles, dims.n_y, dims.n_x
        )
        angle_deg = self._resolve_angle_deg(first_path, angle_idx, dims)

        return MultiSpectrumData(
            spectra=spectra,
            energies=energies,
            index=index,
            coordinates=Coordinates4D(
                x=x,
                y=y,
                t=t,
                angle_idx=angle_idx,
                angle_deg=round(angle_deg, 1),
            ),
            read_time_ms=round(read_time_ms, 3),
            energy_type=self._energy_type_cache.get(first_path, "BE"),
        )

    def get_energy(self, path: str) -> NDArray[np.float32]:
        """Get cached energy axis for a file."""
        _, _, energy = self._get_file(path)
        return energy

    def get_metadata(self, path: str) -> FileMetadata | None:
        """Get cached metadata for a file."""
        return self._metadata_cache.get(path)

    def get_all_metadata(self) -> dict[str, FileMetadata]:
        """Get all cached metadata."""
        return self._metadata_cache.copy()

    def prefetch_range(self, path: str, start: int, end: int) -> list[NDArray[np.float32]]:
        """
        Prefetch a range of spectra for batch processing.

        Args:
            path: Path to HDF5 file
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            List of intensity arrays
        """
        _, specdata, _ = self._get_file(path)
        row_offset = self._row_offset(path)
        n_spectra = self._n_spectra(path)

        end = min(end, n_spectra)
        if start >= end:
            return []

        # Read batch (adjust for energy row in original format)
        if path in self._mmap_cache:
            mm = self._mmap_cache[path]
            batch = np.array(mm[start + row_offset : end + row_offset, :])
        else:
            batch = specdata[start + row_offset : end + row_offset, :].astype(np.float32)
        return [batch[i] for i in range(batch.shape[0])]

    def get_spectra_batch(
        self, path: str, start: int, count: int
    ) -> NDArray[np.float32]:
        """
        Get a batch of spectra as a numpy array.

        Optimized for voigtfit batch processing - returns (n_energy, n_spectra) array
        without Python loop overhead.

        Args:
            path: Path to HDF5 file
            start: Start index (inclusive)
            count: Number of spectra to read

        Returns:
            (n_energy, n_spectra) array of spectral intensities
        """
        _, specdata, _ = self._get_file(path)
        row_offset = self._row_offset(path)
        n_spectra = self._n_spectra(path)

        # Clamp to valid range
        end = min(start + count, n_spectra)
        actual_count = end - start

        if actual_count <= 0:
            return np.empty((specdata.shape[1], 0), dtype=np.float32)

        # Read batch (row offset: +1 for original, +0 for uint16)
        if path in self._mmap_cache:
            mm = self._mmap_cache[path]
            batch = np.array(mm[start + row_offset : end + row_offset, :])
        else:
            batch = specdata[start + row_offset : end + row_offset, :].astype(np.float32)
        return batch.T  # Transpose to (n_energy, n_spectra)

    def get_spectra_batch_rowmajor(
        self, path: str, start: int, count: int
    ) -> NDArray[np.float32]:
        """
        Get a batch of spectra in row-major format (C-contiguous).

        Optimized for MLX GPU processing - avoids transpose overhead.
        Returns (n_spectra, n_energy) array matching HDF5 storage layout.

        Args:
            path: Path to HDF5 file
            start: Start index (inclusive)
            count: Number of spectra to read

        Returns:
            (n_spectra, n_energy) C-contiguous array of spectral intensities
        """
        _, specdata, _ = self._get_file(path)
        row_offset = self._row_offset(path)
        n_spectra = self._n_spectra(path)
        n_energy = specdata.shape[1]

        # Clamp to valid range
        end = min(start + count, n_spectra)
        actual_count = end - start

        if actual_count <= 0:
            return np.empty((0, n_energy), dtype=np.float32)

        # Fastest path: memory-mapped file (bypasses h5py entirely, original format only)
        if path in self._mmap_cache:
            mm = self._mmap_cache[path]
            return np.array(mm[start + row_offset : end + row_offset, :])

        # Read batch (row offset: +1 for original, +0 for uint16)
        # Use read_direct for optimal performance when dtype matches
        if specdata.dtype == np.float32:
            # Pre-allocate buffer and read directly (fastest path)
            batch = np.empty((actual_count, n_energy), dtype=np.float32)
            try:
                specdata.read_direct(
                    batch,
                    source_sel=np.s_[start + row_offset : end + row_offset, :]
                )
            except TypeError:
                # Fallback for older h5py versions
                batch = specdata[start + row_offset : end + row_offset, :].astype(np.float32)
        else:
            # uint16 or other types -> convert to float32
            batch = specdata[start + row_offset : end + row_offset, :].astype(np.float32)

        # batch is already C-contiguous from np.empty
        return batch

    def get_spectra_batch_mlx(
        self, path: str, start: int, count: int
    ):
        """
        Get a batch of spectra directly as MLX array.

        Fastest path for MLX GPU processing - reads HDF5 directly to MLX
        without intermediate transpose. Returns (n_spectra, n_energy) MLX array.

        Args:
            path: Path to HDF5 file
            start: Start index (inclusive)
            count: Number of spectra to read

        Returns:
            MLX array (n_spectra, n_energy) or None if MLX unavailable
        """
        if not HAS_MLX:
            return None

        # Get row-major NumPy array
        batch_np = self.get_spectra_batch_rowmajor(path, start, count)

        # Convert to MLX (single conversion, no transpose)
        return mx.array(batch_np)

    def _row_offset(self, path: str) -> int:
        """Return row offset: 0 for uint16 format, 1 for original format."""
        return 0 if self._is_uint16.get(path, False) else 1

    def _n_spectra(self, path: str) -> int:
        """Return number of spectra for a cached file."""
        sd = self._specdata_cache[path]
        return sd.shape[0] if self._is_uint16.get(path, False) else sd.shape[0] - 1

    def get_energy_type(self, path: str) -> str:
        """Get energy axis type for a file ('BE' or 'KE')."""
        return self._energy_type_cache.get(path, "BE")

    def clear(self) -> None:
        """Close all files and clear caches."""
        for f in self._file_cache.values():
            try:
                f.close()
            except Exception:
                pass

        self._mmap_cache.clear()
        self._is_uint16.clear()
        self._file_cache.clear()
        self._specdata_cache.clear()
        self._energy_cache.clear()
        self._metadata_cache.clear()
        self._dims_cache.clear()
        self._angle_values_cache.clear()
        self._energy_type_cache.clear()

    def close_files_only(self) -> None:
        """Close file handles but preserve metadata and dims caches.

        Use this when you need to release HDF5 file handles temporarily
        (e.g., for inplace writing) but want to re-open them later
        without losing metadata (element names, n_spectra, etc.).
        Call _get_file(path) afterward to re-open handles.
        """
        for f in self._file_cache.values():
            try:
                f.close()
            except Exception:
                pass

        self._mmap_cache.clear()
        self._is_uint16.clear()
        self._file_cache.clear()
        self._specdata_cache.clear()
        self._energy_cache.clear()
        # Note: _metadata_cache and _dims_cache are intentionally preserved

    def close(self) -> None:
        """Alias for clear()."""
        self.clear()

    @property
    def open_files(self) -> int:
        """Number of currently open files."""
        return len(self._file_cache)

    @property
    def cached_metadata_count(self) -> int:
        """Number of cached file metadata entries."""
        return len(self._metadata_cache)

    def __enter__(self) -> HDF5Cache:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.clear()

    def __del__(self) -> None:
        self.clear()
