"""XPS element database for spin-orbit splitting and branch ratios.

Provides lookup of SO splitting (eV) and branch ratio for common XPS
orbital labels (e.g., "Si2p", "Ta4f", "O1s").

The branch ratio is j_minus / j_plus = (2l) / (2l+2) where l is the
orbital angular momentum quantum number:

    s (l=0): no splitting
    p (l=1): 1/2 (2p1/2 : 2p3/2)
    d (l=2): 2/3 (3d3/2 : 3d5/2)
    f (l=3): 3/4 (4f5/2 : 4f7/2)

SO splitting values are representative values for common chemical
states. They are intended as initial guesses for fitting, not as
definitive references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SOInfo:
    """Spin-orbit coupling information for an XPS orbital."""

    so_split: float  # eV, splitting between j+ and j- components
    branch_ratio: float  # area(j-) / area(j+), e.g. 0.5 for p
    orbital: str  # e.g. "2p", "4f"
    j_plus_label: str  # e.g. "2p3/2"
    j_minus_label: str  # e.g. "2p1/2"


# Branch ratios from quantum mechanics: (2l) / (2l+2)
_BRANCH_RATIOS = {
    "s": 0.0,  # no splitting
    "p": 0.5,  # 1:2
    "d": 2 / 3,  # 2:3
    "f": 3 / 4,  # 3:4
}

# SO splitting values (eV) for common elements.
# Keys: element symbol + orbital (e.g. "Si2p").
# Values: splitting in eV (positive).
# Sources: various XPS handbooks, NIST XPS database.
_SO_SPLITS: dict[str, float] = {
    # 2p orbitals
    "Si2p": 0.6,
    "P2p": 0.84,
    "S2p": 1.16,
    "Cl2p": 1.6,
    "Al2p": 0.44,
    "Mg2p": 0.28,
    # 3p orbitals
    "Ge3p": 4.0,
    # 3d orbitals
    "Ti3d": 6.1,  # actually 2p, but Ti2p is common
    "Ti2p": 5.7,
    "V2p": 7.3,
    "Cr2p": 9.0,
    "Mn2p": 11.2,
    "Fe2p": 13.1,
    "Co2p": 15.0,
    "Ni2p": 17.3,
    "Cu2p": 19.8,
    "Zn2p": 23.0,
    "Ga3d": 0.44,
    "Ge3d": 0.55,
    "As3d": 0.69,
    "Se3d": 0.86,
    "Zr3d": 2.43,
    "Nb3d": 2.72,
    "Mo3d": 3.15,
    "Ru3d": 4.17,
    "Rh3d": 4.74,
    "Pd3d": 5.26,
    "Ag3d": 6.0,
    "Cd3d": 6.75,
    "In3d": 7.56,
    "Sn3d": 8.42,
    # 4d orbitals
    "Hf4d": 16.8,
    "Ta4d": 12.4,
    "W4d": 12.0,
    # 4f orbitals
    "Hf4f": 1.66,
    "Ta4f": 1.91,
    "W4f": 2.18,
    "Re4f": 2.43,
    "Os4f": 2.74,
    "Ir4f": 2.98,
    "Pt4f": 3.33,
    "Au4f": 3.67,
    "Pb4f": 4.86,
    "Bi4f": 5.32,
    "Ce3d": 18.6,
    "La3d": 16.8,
}

# Regex to parse element labels like "Si2p", "Ta4f", "O1s"
_LABEL_RE = re.compile(r"^([A-Z][a-z]?)(\d)([spdf])$")


def parse_element_label(label: str) -> tuple[str, str, str] | None:
    """Parse an XPS element label into (element, n, orbital_letter).

    Examples:
        "Si2p" -> ("Si", "2", "p")
        "Ta4f" -> ("Ta", "4", "f")
        "O1s"  -> ("O", "1", "s")

    Returns None if the label doesn't match.
    """
    m = _LABEL_RE.match(label.strip())
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def get_so_info(label: str) -> SOInfo | None:
    """Look up spin-orbit splitting info for an XPS orbital label.

    Args:
        label: XPS label like "Si2p", "Ta4f", "O1s".

    Returns:
        SOInfo with splitting and branch ratio, or None for s orbitals
        or unknown elements.
    """
    parsed = parse_element_label(label)
    if parsed is None:
        return None

    element, n, orbital_letter = parsed

    # s orbitals have no SO splitting
    if orbital_letter == "s":
        return None

    branch_ratio = _BRANCH_RATIOS.get(orbital_letter, 0.0)
    if branch_ratio == 0.0:
        return None

    key = f"{element}{n}{orbital_letter}"
    so_split = _SO_SPLITS.get(key)
    if so_split is None:
        return None

    # Build j labels
    l_qn = {"p": 1, "d": 2, "f": 3}[orbital_letter]
    j_plus = l_qn + 0.5  # higher j
    j_minus = l_qn - 0.5  # lower j

    def _j_str(j: float) -> str:
        return f"{int(2*j)}/2" if j != int(j) else str(int(j))

    orbital_str = f"{n}{orbital_letter}"
    j_plus_label = f"{orbital_str}{_j_str(j_plus)}"
    j_minus_label = f"{orbital_str}{_j_str(j_minus)}"

    return SOInfo(
        so_split=so_split,
        branch_ratio=branch_ratio,
        orbital=orbital_str,
        j_plus_label=j_plus_label,
        j_minus_label=j_minus_label,
    )
