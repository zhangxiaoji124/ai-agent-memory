"""
TexMex vector file I/O utilities.

Formats:
  fvecs  – float32, record = [dim:u32] + [dim × f32]
  bvecs  – uint8,   record = [dim:u32] + [dim × u8]
  ivecs  – int32,   record = [dim:u32] + [dim × i32]
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

import numpy as np


def read_fvecs(path: str | Path, max_vecs: int | None = None) -> np.ndarray:
    """Return (N, dim) float32 array."""
    return _read_vecs(path, np.float32, 4, max_vecs)


def read_bvecs(path: str | Path, max_vecs: int | None = None) -> np.ndarray:
    """Return (N, dim) uint8 array."""
    return _read_vecs(path, np.uint8, 1, max_vecs)


def read_ivecs(path: str | Path, max_vecs: int | None = None) -> np.ndarray:
    """Return (N, dim) int32 array."""
    return _read_vecs(path, np.int32, 4, max_vecs)


def write_fvecs(path: str | Path, vecs: np.ndarray) -> None:
    """Write (N, dim) float32 array as fvecs."""
    _write_vecs(path, vecs.astype(np.float32), 4)


def write_bvecs(path: str | Path, vecs: np.ndarray) -> None:
    """Write (N, dim) uint8 array as bvecs."""
    _write_vecs(path, vecs.astype(np.uint8), 1)


def write_ivecs(path: str | Path, vecs: np.ndarray) -> None:
    """Write (N, dim) int32 array as ivecs."""
    _write_vecs(path, vecs.astype(np.int32), 4)


def iter_fvecs(path: str | Path, chunk: int = 10_000) -> Iterator[np.ndarray]:
    """Yield (chunk, dim) float32 chunks without loading the whole file."""
    yield from _iter_vecs(path, np.float32, 4, chunk)


def iter_bvecs(path: str | Path, chunk: int = 10_000) -> Iterator[np.ndarray]:
    yield from _iter_vecs(path, np.uint8, 1, chunk)


def probe(path: str | Path) -> tuple[int, int]:
    """Return (dim, num_vectors) without loading data."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read(4)
    if len(raw) < 4:
        raise ValueError(f"Too short: {path}")
    (dim,) = struct.unpack("<I", raw)
    ext = path.suffix.lower()
    elem_bytes = 1 if ext == ".bvecs" else 4
    rec_bytes = 4 + dim * elem_bytes
    n = path.stat().st_size // rec_bytes
    return dim, n


# ── internals ────────────────────────────────────────────────────────────────

def _read_vecs(path: str | Path, dtype: np.dtype, elem_size: int,
               max_vecs: int | None) -> np.ndarray:
    path = Path(path)
    with open(path, "rb") as f:
        raw_dim = f.read(4)
    if len(raw_dim) < 4:
        return np.empty((0, 0), dtype=dtype)
    (dim,) = struct.unpack("<I", raw_dim)
    rec = 4 + dim * elem_size
    total_size = path.stat().st_size
    n_total = total_size // rec
    n = n_total if max_vecs is None else min(n_total, max_vecs)

    out = np.empty((n, dim), dtype=dtype)
    with open(path, "rb") as f:
        for i in range(n):
            f.read(4)  # skip dim field
            raw = f.read(dim * elem_size)
            if len(raw) < dim * elem_size:
                out = out[:i]
                break
            out[i] = np.frombuffer(raw, dtype=dtype)
    return out


def _iter_vecs(path: str | Path, dtype: np.dtype, elem_size: int,
               chunk: int) -> Iterator[np.ndarray]:
    path = Path(path)
    with open(path, "rb") as f:
        raw_dim = f.read(4)
        if len(raw_dim) < 4:
            return
        (dim,) = struct.unpack("<I", raw_dim)
        f.seek(0)
        buf: list[np.ndarray] = []
        while True:
            hdr = f.read(4)
            if not hdr:
                break
            raw = f.read(dim * elem_size)
            if len(raw) < dim * elem_size:
                break
            buf.append(np.frombuffer(raw, dtype=dtype).copy())
            if len(buf) >= chunk:
                yield np.stack(buf)
                buf = []
    if buf:
        yield np.stack(buf)


def _write_vecs(path: str | Path, vecs: np.ndarray, elem_size: int) -> None:
    path = Path(path)
    n, dim = vecs.shape
    with open(path, "wb") as f:
        for i in range(n):
            f.write(struct.pack("<I", dim))
            f.write(vecs[i].tobytes())
