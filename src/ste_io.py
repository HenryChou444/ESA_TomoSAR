
from pathlib import Path
import struct
import numpy as np


def rrat(filename):
    """Read a 2D, single-channel RAT v2 file."""
    filename = Path(filename)

    with filename.open("rb") as f:
        header = f.read(52)

    if len(header) != 52:
        raise ValueError("Invalid RAT header")

    magic, version, ndim, channels = struct.unpack_from(
        "<ifii", header, 0
    )

    if magic != 844382546 or version != 2.0:
        raise ValueError("Not a supported RAT v2 file")

    if ndim != 2 or channels != 1:
        raise ValueError("Expected a 2D single-channel RAT file")

    dims = struct.unpack_from("<8i", header, 16)[:ndim]
    var = struct.unpack_from("<i", header, 48)[0]

    types = {
        1: "u1",
        2: "<i2",
        3: "<i4",
        4: "<f4",
        5: "<f8",
        6: "<c8",
        9: "<c16",
    }

    if var not in types:
        raise ValueError(f"Unsupported RAT data type: {var}")

    dtype = np.dtype(types[var])
    shape = tuple(reversed(dims))
    expected_size = 1000 + int(np.prod(shape)) * dtype.itemsize

    if filename.stat().st_size < expected_size:
        raise ValueError("RAT file is smaller than expected")

    return np.memmap(
        filename,
        dtype=dtype,
        mode="r",
        offset=1000,
        shape=shape,
    )
