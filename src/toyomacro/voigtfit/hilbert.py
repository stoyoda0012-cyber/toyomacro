"""
3D Hilbert Space-Filling Curve
===============================

Bijective mapping between 3D integer coordinates and a 1D Hilbert index.
Used for locality-preserving encoding of spectral parameters into RGB.

The 3D Hilbert curve of order p maps [0, 2^p)^3 <-> [0, 2^(3p)).
Nearby points in 3D space map to nearby indices on the curve, making it
ideal for parameter quantization with minimal discontinuities.

Algorithm: Gray-code based rotation method (Skilling 2004).
    T. Skilling, "Programming the Hilbert curve",
    AIP Conf. Proc. 707, 381-387 (2004).

The algorithm has three stages:
  Forward (coords->index): InverseUndo -> GrayEncode -> Interleave
  Inverse (index->coords): Deinterleave -> GrayDecode -> UndoExcess

NOTE: The inverse undo loop includes i=0 (conditional inversion of X[0]),
which is required to reverse the forward's i=0 step. Some transcriptions
of Skilling's algorithm omit i=0 in the inverse, producing incorrect
results when the forward includes i=0.

Vectorized for batch processing (numpy).
"""


import numpy as np

# ---------------------------------------------------------------------------
# Scalar reference implementations (for correctness testing)
# ---------------------------------------------------------------------------

def _scalar_xyz_to_d(x: int, y: int, z: int, order: int) -> int:
    """Scalar 3D->1D Hilbert. Reference implementation for testing."""
    coords = [x, y, z]
    n = 3  # dimensions
    N = 1 << order

    # Stage 1: Inverse undo (rotation + inversion at each level)
    M = N >> 1
    while M > 0:
        P = M - 1
        for i in range(n):
            if coords[i] & M:
                coords[0] ^= P
            else:
                t = (coords[0] ^ coords[i]) & P
                coords[0] ^= t
                coords[i] ^= t
        M >>= 1

    # Stage 2: Gray encode
    for i in range(1, n):
        coords[i] ^= coords[i - 1]
    t = 0
    Q = N >> 1
    while Q > 1:
        if coords[n - 1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(n):
        coords[i] ^= t

    # Stage 3: Interleave bits
    d = 0
    for i in range(order):
        for dim in range(n):
            d |= ((coords[dim] >> i) & 1) << (n * i + dim)
    return d


def _scalar_d_to_xyz(d: int, order: int) -> tuple[int, int, int]:
    """Scalar 1D->3D Hilbert. Reference implementation for testing."""
    n = 3
    N = 1 << order

    # Stage 1: De-interleave bits
    coords = [0, 0, 0]
    for i in range(order):
        for dim in range(n):
            coords[dim] |= ((d >> (n * i + dim)) & 1) << i

    # Stage 2: Gray decode
    t = coords[n - 1] >> 1
    for i in range(n - 1, 0, -1):
        coords[i] ^= coords[i - 1]
    coords[0] ^= t

    # Stage 3: Undo excess work (reverse rotation at each level)
    # NOTE: includes i=0 to undo forward's i=0 conditional inversion
    Q = 2
    while Q < N:
        P = Q - 1
        for i in range(n - 1, -1, -1):
            if coords[i] & Q:
                coords[0] ^= P
            else:
                t = (coords[0] ^ coords[i]) & P
                coords[0] ^= t
                coords[i] ^= t
        Q <<= 1

    return coords[0], coords[1], coords[2]


# ---------------------------------------------------------------------------
# Vectorized implementations (numpy)
# ---------------------------------------------------------------------------

def coords_to_index(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    order: int) -> np.ndarray:
    """Convert 3D coordinates to Hilbert curve index.

    Args:
        x, y, z: Integer coordinates in [0, 2^order), shape (N,) or scalar
        order: Hilbert curve order (bits per dimension)

    Returns:
        index: Hilbert index in [0, 2^(3*order)), same shape as inputs
    """
    X = np.asarray(x, dtype=np.int64).copy()
    Y = np.asarray(y, dtype=np.int64).copy()
    Z = np.asarray(z, dtype=np.int64).copy()
    N = 1 << order

    # Stage 1: Inverse undo
    M = N >> 1
    while M > 0:
        P = np.int64(M - 1)
        mask_x = (X & M) != 0
        mask_y = (Y & M) != 0
        mask_z = (Z & M) != 0

        # i=0: if X bit set -> invert lower bits of X; else no-op
        X = np.where(mask_x, X ^ P, X)

        # i=1: if Y bit set -> invert lower bits of X; else swap X<->Y lower bits
        t = (X ^ Y) & P
        X = np.where(mask_y, X ^ P, X ^ t)
        Y = np.where(mask_y, Y, Y ^ t)

        # i=2: if Z bit set -> invert lower bits of X; else swap X<->Z lower bits
        t = (X ^ Z) & P
        X = np.where(mask_z, X ^ P, X ^ t)
        Z = np.where(mask_z, Z, Z ^ t)

        M >>= 1

    # Stage 2: Gray encode
    Y ^= X
    Z ^= Y
    t = np.zeros_like(X)
    Q = N >> 1
    while Q > 1:
        mask = (Z & Q) != 0
        t = np.where(mask, t ^ np.int64(Q - 1), t)
        Q >>= 1
    X ^= t
    Y ^= t
    Z ^= t

    # Stage 3: Interleave bits
    index = np.zeros_like(X)
    for i in range(order):
        index |= ((X >> i) & 1) << (3 * i)
        index |= ((Y >> i) & 1) << (3 * i + 1)
        index |= ((Z >> i) & 1) << (3 * i + 2)

    return index


def index_to_coords(index: np.ndarray,
                    order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Hilbert curve index to 3D coordinates.

    Args:
        index: Hilbert index in [0, 2^(3*order)), shape (N,) or scalar
        order: Hilbert curve order (bits per dimension)

    Returns:
        x, y, z: Integer coordinates in [0, 2^order)
    """
    index = np.asarray(index, dtype=np.int64)
    N = 1 << order

    # Stage 1: De-interleave bits
    X = np.zeros_like(index)
    Y = np.zeros_like(index)
    Z = np.zeros_like(index)
    for i in range(order):
        X |= ((index >> (3 * i)) & 1) << i
        Y |= ((index >> (3 * i + 1)) & 1) << i
        Z |= ((index >> (3 * i + 2)) & 1) << i

    # Stage 2: Gray decode
    t = Z >> 1
    Z ^= Y
    Y ^= X
    X ^= t

    # Stage 3: Undo excess work (includes i=0)
    Q = 2
    while Q < N:
        P = np.int64(Q - 1)
        mask_z = (Z & Q) != 0
        mask_y = (Y & Q) != 0
        mask_x = (X & Q) != 0

        # i=2 (Z): if Z bit set -> invert X lower; else swap X<->Z lower
        t = (X ^ Z) & P
        X = np.where(mask_z, X ^ P, X ^ t)
        Z = np.where(mask_z, Z, Z ^ t)

        # i=1 (Y): if Y bit set -> invert X lower; else swap X<->Y lower
        t = (X ^ Y) & P
        X = np.where(mask_y, X ^ P, X ^ t)
        Y = np.where(mask_y, Y, Y ^ t)

        # i=0 (X): if X bit set -> invert X lower; else no-op
        X = np.where(mask_x, X ^ P, X)

        Q <<= 1

    return X, Y, Z


# ---------------------------------------------------------------------------
# 2D Hilbert curve (Skilling 2004, n=2)
# ---------------------------------------------------------------------------

def _scalar_xy_to_d_2d(x: int, y: int, order: int) -> int:
    """Scalar 2D->1D Hilbert. Reference implementation for testing."""
    coords = [x, y]
    n = 2  # dimensions
    N = 1 << order

    # Stage 1: Inverse undo
    M = N >> 1
    while M > 0:
        P = M - 1
        for i in range(n):
            if coords[i] & M:
                coords[0] ^= P
            else:
                t = (coords[0] ^ coords[i]) & P
                coords[0] ^= t
                coords[i] ^= t
        M >>= 1

    # Stage 2: Gray encode
    for i in range(1, n):
        coords[i] ^= coords[i - 1]
    t = 0
    Q = N >> 1
    while Q > 1:
        if coords[n - 1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(n):
        coords[i] ^= t

    # Stage 3: Interleave bits
    d = 0
    for i in range(order):
        for dim in range(n):
            d |= ((coords[dim] >> i) & 1) << (n * i + dim)
    return d


def _scalar_d_to_xy_2d(d: int, order: int) -> tuple[int, int]:
    """Scalar 1D->2D Hilbert. Reference implementation for testing."""
    n = 2
    N = 1 << order

    # Stage 1: De-interleave bits
    coords = [0, 0]
    for i in range(order):
        for dim in range(n):
            coords[dim] |= ((d >> (n * i + dim)) & 1) << i

    # Stage 2: Gray decode
    t = coords[n - 1] >> 1
    for i in range(n - 1, 0, -1):
        coords[i] ^= coords[i - 1]
    coords[0] ^= t

    # Stage 3: Undo excess work (includes i=0)
    Q = 2
    while Q < N:
        P = Q - 1
        for i in range(n - 1, -1, -1):
            if coords[i] & Q:
                coords[0] ^= P
            else:
                t = (coords[0] ^ coords[i]) & P
                coords[0] ^= t
                coords[i] ^= t
        Q <<= 1

    return coords[0], coords[1]


def coords_to_index_2d(x: np.ndarray, y: np.ndarray,
                       order: int) -> np.ndarray:
    """Convert 2D coordinates to Hilbert curve index.

    Args:
        x, y: Integer coordinates in [0, 2^order), shape (N,) or scalar
        order: Hilbert curve order (bits per dimension)

    Returns:
        index: Hilbert index in [0, 2^(2*order)), same shape as inputs
    """
    X = np.asarray(x, dtype=np.int64).copy()
    Y = np.asarray(y, dtype=np.int64).copy()
    N = 1 << order

    # Stage 1: Inverse undo
    M = N >> 1
    while M > 0:
        P = np.int64(M - 1)
        mask_x = (X & M) != 0
        mask_y = (Y & M) != 0

        # i=0: if X bit set -> invert lower bits of X; else no-op
        X = np.where(mask_x, X ^ P, X)

        # i=1: if Y bit set -> invert lower bits of X; else swap X<->Y lower bits
        t = (X ^ Y) & P
        X = np.where(mask_y, X ^ P, X ^ t)
        Y = np.where(mask_y, Y, Y ^ t)

        M >>= 1

    # Stage 2: Gray encode
    Y ^= X
    t = np.zeros_like(X)
    Q = N >> 1
    while Q > 1:
        mask = (Y & Q) != 0
        t = np.where(mask, t ^ np.int64(Q - 1), t)
        Q >>= 1
    X ^= t
    Y ^= t

    # Stage 3: Interleave bits
    index = np.zeros_like(X)
    for i in range(order):
        index |= ((X >> i) & 1) << (2 * i)
        index |= ((Y >> i) & 1) << (2 * i + 1)

    return index


def index_to_coords_2d(index: np.ndarray,
                       order: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert Hilbert curve index to 2D coordinates.

    Args:
        index: Hilbert index in [0, 2^(2*order)), shape (N,) or scalar
        order: Hilbert curve order (bits per dimension)

    Returns:
        x, y: Integer coordinates in [0, 2^order)
    """
    index = np.asarray(index, dtype=np.int64)
    N = 1 << order

    # Stage 1: De-interleave bits
    X = np.zeros_like(index)
    Y = np.zeros_like(index)
    for i in range(order):
        X |= ((index >> (2 * i)) & 1) << i
        Y |= ((index >> (2 * i + 1)) & 1) << i

    # Stage 2: Gray decode
    t = Y >> 1
    Y ^= X
    X ^= t

    # Stage 3: Undo excess work (includes i=0)
    Q = 2
    while Q < N:
        P = np.int64(Q - 1)
        mask_y = (Y & Q) != 0
        mask_x = (X & Q) != 0

        # i=1 (Y): if Y bit set -> invert X lower; else swap X<->Y lower
        t = (X ^ Y) & P
        X = np.where(mask_y, X ^ P, X ^ t)
        Y = np.where(mask_y, Y, Y ^ t)

        # i=0 (X): if X bit set -> invert X lower; else no-op
        X = np.where(mask_x, X ^ P, X)

        Q <<= 1

    return X, Y
