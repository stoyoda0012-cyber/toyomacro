"""In-test builders for synthetic XPS instrument files.

These generate small but structurally valid PXT (Igor packed experiment),
IBW (standalone Igor wave v5), VAMAS, NPL and SES-text files from scratch,
so the reader tests can run on any machine / public CI without the private
measured-data tree (``TOYOMACRO_TESTDATA``).

The binary layouts mirror exactly what the readers parse (see
``toyomacro/io/readers/pxt_reader.py`` for offsets); they are not full
implementations of the vendor formats.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# --- Igor binary wave builders -----------------------------------------

_WAVE_TYPE_FLOAT32 = 2

# Header sizes as parsed by pxt_reader: bin header 64 bytes + wave header
_V3_HEADER_SIZE = 64 + 328
_V5_HEADER_SIZE = 64 + 320


def _pack_dims(data: np.ndarray) -> list[int]:
    dims = list(data.shape) + [0, 0, 0, 0]
    return dims[:4]


def build_wave_v3(
    data: np.ndarray,
    notes: str,
    x_ini: tuple[float, ...] = (0.0,),
    x_delta: tuple[float, ...] = (1.0,),
) -> bytes:
    """Serialize a v3 Igor binary wave blob (as embedded in .pxt files)."""
    data32 = np.asarray(data, dtype=np.float32)
    notes_b = notes.encode("latin-1")

    hdr = bytearray(_V3_HEADER_SIZE)
    hdr[0:2] = struct.pack("<h", 3)  # bin header version
    hdr[16:20] = struct.pack("<i", 0)  # formula_size
    hdr[20:24] = struct.pack("<i", len(notes_b))  # note_size
    # Wave header fields at wave_start + 88 (v3)
    hdr[88:90] = struct.pack("<H", _WAVE_TYPE_FLOAT32)
    hdr[140:156] = struct.pack("<4i", *_pack_dims(data32))
    sfa = list(x_delta) + [0.0] * (4 - len(x_delta))
    sfb = list(x_ini) + [0.0] * (4 - len(x_ini))
    hdr[156:188] = struct.pack("<4d", *sfa)
    hdr[188:220] = struct.pack("<4d", *sfb)

    return bytes(hdr) + data32.tobytes(order="F") + notes_b


def write_pxt(
    path: Path,
    regions: list[tuple[np.ndarray, str]],
    x_ini: tuple[float, ...] = (100.0,),
    x_delta: tuple[float, ...] = (0.05,),
) -> Path:
    """Write a packed-experiment .pxt file with one wave record per region.

    Args:
        path: Output path.
        regions: (data, wave_notes) per region. Notes use "key=value\\r" lines.
        x_ini / x_delta: Axis scaling shared by all regions.
    """
    blob = b""
    for data, notes in regions:
        wave = build_wave_v3(data, notes, x_ini=x_ini, x_delta=x_delta)
        blob += struct.pack("<HHI", 3, 5, len(wave)) + wave  # record type 3
    blob += struct.pack("<HHI", 0, 0, 0)  # END marker
    path.write_bytes(blob)
    return path


def write_ibw(
    path: Path,
    data: np.ndarray,
    notes: str,
    x_ini: tuple[float, ...] = (100.0,),
    x_delta: tuple[float, ...] = (0.05,),
) -> Path:
    """Write a standalone Igor binary wave v5 (.ibw) file."""
    data32 = np.asarray(data, dtype=np.float32)
    notes_b = notes.encode("latin-1")

    hdr = bytearray(_V5_HEADER_SIZE)
    hdr[0:2] = struct.pack("<h", 5)  # bin header version
    hdr[8:12] = struct.pack("<i", 0)  # formula_size
    hdr[12:16] = struct.pack("<i", len(notes_b))  # note_size
    # Wave header fields at wave_start + 80 (v5)
    hdr[80:82] = struct.pack("<H", _WAVE_TYPE_FLOAT32)
    hdr[132:148] = struct.pack("<4i", *_pack_dims(data32))
    sfa = list(x_delta) + [0.0] * (4 - len(x_delta))
    sfb = list(x_ini) + [0.0] * (4 - len(x_ini))
    hdr[148:180] = struct.pack("<4d", *sfa)
    hdr[180:212] = struct.pack("<4d", *sfb)

    path.write_bytes(bytes(hdr) + data32.tobytes(order="F") + notes_b)
    return path


def make_notes(pairs: dict[str, object]) -> str:
    """Format wave notes as Scienta-style "key=value" CR-separated lines."""
    return "".join(f"{k}={v}\r" for k, v in pairs.items())


# --- Text format builders ----------------------------------------------


def write_vamas(
    path: Path,
    n_energy: int = 16,
    e_ini: float = 380.0,
    e_step: float = 0.1,
    n_sweeps: int = 4,
) -> Path:
    """Write a minimal single-region VAMAS (.vms) file (XPS block layout)."""
    lines: list[str] = []
    # File header: 12 non-empty lines; line[5] = n_regions
    header = ["x"] * 12
    header[0] = "VAMAS Surface Chemical Analysis Standard Data Transfer Format 1988 May 4"
    header[5] = "1"
    lines += header
    # Region header: 9 lines; [0..5] datetime, [8] = "XPS"
    region = ["2026", "1", "15", "10", "30", "0", "0", "0", "XPS"]
    lines += region
    # Extended header: 34 lines for "XPS" blocks
    ext = ["0"] * 34
    ext[18] = str(e_ini)
    ext[19] = str(e_step)
    ext[20] = "1"  # n_slices
    ext[25] = str(n_sweeps)
    ext[31] = str(n_energy)
    lines += ext
    # Data
    lines += [str(1000 + 10 * i) for i in range(n_energy)]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_npl(
    path: Path,
    n_energy: int = 16,
    e_ini: float = 1100.0,
    e_step: float = -0.1,
    excitation_energy: str = "1486.6",
    energy_label: str = "kinetic energy",
    region_base: str = "Au4f",
) -> Path:
    """Write a minimal single-region, named-region NPL (.npl) file.

    Pass a non-numeric ``excitation_energy`` (e.g. ``"none"``) to simulate
    a file where hv is not recorded.
    """
    lines: list[str] = []
    # File header: 14 lines; [7] = n_regions
    header = ["x"] * 14
    header[7] = "1"
    lines += header
    # Region header: 31 lines; [2..4] date, [14] hv, [29] name, [30] flag
    region = ["0"] * 31
    region[2], region[3], region[4] = "2026", "1", "15"
    region[14] = excitation_energy
    region[29] = region_base
    region[30] = "1"  # named region (flag != "-1")
    lines += region
    # Extended header: 19 lines (named region)
    ext = ["0"] * 19
    ext[1] = energy_label
    ext[3] = str(e_ini)
    ext[4] = str(e_step)
    ext[5] = "1"  # n_slices
    ext[10] = "8"  # n_sweeps
    ext[16] = str(n_energy)
    lines += ext
    # Data
    lines += [str(500 + 5 * i) for i in range(n_energy)]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_ses_txt(
    path: Path,
    n_energy: int = 12,
    n_angle: int = 4,
    lens_mode: str = "Angular56",
    extra_info: dict[str, str] | None = None,
) -> Path:
    """Write a minimal SES INI-style text export with one 2D region."""
    energy = 100.0 + 0.1 * np.arange(n_energy)
    angle = -7.0 + 1.0 * np.arange(n_angle)
    rng = np.random.default_rng(0)
    block = rng.integers(100, 200, size=(n_energy, n_angle))

    info = {
        "Region Name": "Si2p",
        "Excitation Energy": "1486.6",
        "Energy Scale": "Kinetic",
        "Lens Mode": lens_mode,
        "Pass Energy": "50",
        "Number of Sweeps": "2",
        "Number of Slices": "1",
    }
    info.update(extra_info or {})

    out = ["[Info]", "Number of Regions=1", "Version=1.3.1", ""]
    out += [
        "[Region 1]",
        f"Dimension 1 scale={' '.join(f'{e:.3f}' for e in energy)}",
        "Dimension 2 name=Y-Scale [deg]",
        f"Dimension 2 scale={' '.join(f'{a:.3f}' for a in angle)}",
        "",
        "[Info 1]",
    ]
    out += [f"{k}={v}" for k, v in info.items()]
    out += ["", "[Data 1]"]
    for i in range(n_energy):
        row = " ".join(str(int(v)) for v in block[i])
        out.append(f"{energy[i]:.3f} {row}")
    path.write_text("\n".join(out) + "\n")
    return path
