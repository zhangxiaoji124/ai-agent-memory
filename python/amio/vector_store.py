"""
Python ctypes wrapper around the amio_py shared library.

Build the shared library first:
    cmake -B build -DAMIO_BUILD_PYTHON=ON
    cmake --build build --target amio_py

Then use:
    from amio import VectorStore
    with VectorStore("data/sift_base.index", ef_search=192) as vs:
        ids, dists = vs.search_disk(query_vec, k=10)
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import platform
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ── locate shared library ─────────────────────────────────────────────────────

def _find_lib() -> str:
    system = platform.system()
    if system == "Windows":
        names = ["amio_py.dll", "libamio_py.dll"]
    elif system == "Darwin":
        names = ["libamio_py.dylib", "libamio_py.so"]
    else:
        names = ["libamio_py.so"]

    root = pathlib.Path(__file__).parent.parent.parent  # repo root
    search_dirs = [
        root / "build",
        root / "build" / "Release",
        root / "build" / "Debug",
        pathlib.Path(os.environ.get("AMIO_LIB_DIR", ".")),
    ]
    for d in search_dirs:
        for name in names:
            candidate = d / name
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError(
        f"amio_py shared library not found. "
        f"Build with: cmake --build build --target amio_py\n"
        f"Searched in: {[str(d) for d in search_dirs]}"
    )


# ── ctypes structs ────────────────────────────────────────────────────────────

class _AmioMetrics(ctypes.Structure):
    _fields_ = [
        ("cache_hits",          ctypes.c_uint64),
        ("cache_misses",        ctypes.c_uint64),
        ("disk_reads",          ctypes.c_uint64),
        ("prefetch_submitted",  ctypes.c_uint64),
        ("theta_phase_switches",ctypes.c_uint64),
        ("useful_prefetches",   ctypes.c_uint64),
        ("wasted_prefetches",   ctypes.c_uint64),
        ("static_pins",         ctypes.c_uint64),
        ("uring_active",        ctypes.c_int),
    ]


@dataclass
class Metrics:
    cache_hits: int
    cache_misses: int
    disk_reads: int
    prefetch_submitted: int
    theta_phase_switches: int
    useful_prefetches: int
    wasted_prefetches: int
    static_pins: int
    uring_active: bool

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def prefetch_utilization(self) -> float:
        total = self.useful_prefetches + self.wasted_prefetches
        return self.useful_prefetches / total if total > 0 else 0.0


# ── library singleton ─────────────────────────────────────────────────────────

_lib: ctypes.CDLL | None = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(_find_lib())

        lib.amio_store_open.restype = ctypes.c_void_p
        lib.amio_store_open.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        lib.amio_store_close.restype = None
        lib.amio_store_close.argtypes = [ctypes.c_void_p]

        lib.amio_store_search_disk.restype = ctypes.c_int
        lib.amio_store_search_disk.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float),
        ]
        lib.amio_store_search_mem.restype = ctypes.c_int
        lib.amio_store_search_mem.argtypes = lib.amio_store_search_disk.argtypes

        lib.amio_store_insert.restype = ctypes.c_int
        lib.amio_store_insert.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        lib.amio_store_metrics.restype = None
        lib.amio_store_metrics.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_AmioMetrics)
        ]
        lib.amio_store_reset_metrics.restype = None
        lib.amio_store_reset_metrics.argtypes = [ctypes.c_void_p]

        lib.amio_store_partition_profile.restype = ctypes.c_char_p
        lib.amio_store_partition_profile.argtypes = [ctypes.c_void_p]

        lib.amio_detect_vector_kind.restype = ctypes.c_int
        lib.amio_detect_vector_kind.argtypes = [ctypes.c_char_p]

        lib.amio_probe_vector_file.restype = ctypes.c_int
        lib.amio_probe_vector_file.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        _lib = lib
    return _lib


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    High-level Python wrapper around amio VectorStore.

    Parameters
    ----------
    index_path : str
        Path to the .index file built by ``build_index``.
    ef_search : int
        Beam width for search (higher = better recall, slower).
    cache_mb : int
        Dynamic (hot) cache in MiB.
    static_cache_mb : int
        Static (pinned navigation subgraph) cache in MiB.
    """

    def __init__(
        self,
        index_path: str,
        *,
        ef_search: int = 128,
        cache_mb: int = 512,
        static_cache_mb: int = 128,
    ) -> None:
        lib = _get_lib()
        handle = lib.amio_store_open(
            index_path.encode(), ef_search, cache_mb, static_cache_mb
        )
        if not handle:
            raise RuntimeError(f"Failed to open VectorStore at '{index_path}'")
        self._handle = ctypes.c_void_p(handle)
        self._lib = lib
        self._closed = False

    # ── search ────────────────────────────────────────────────────────────────

    def search_disk(
        self, query: Sequence[float] | np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Disk-based approximate k-NN search.

        Returns
        -------
        ids : int64 ndarray, shape (n,)
        distances : float32 ndarray, shape (n,)
        """
        return self._search(query, k, disk=True)

    def search_mem(
        self, query: Sequence[float] | np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """In-memory HNSW search (no disk I/O)."""
        return self._search(query, k, disk=False)

    def _search(self, query, k: int, disk: bool):
        self._check_open()
        q = np.asarray(query, dtype=np.float32)
        dim = q.shape[-1] if q.ndim > 1 else q.size
        q = q.ravel()
        out_ids = np.zeros(k, dtype=np.uint64)
        out_dists = np.zeros(k, dtype=np.float32)

        fn = self._lib.amio_store_search_disk if disk else self._lib.amio_store_search_mem
        n = fn(
            self._handle,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(dim),
            ctypes.c_int(k),
            out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out_dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if n < 0:
            raise RuntimeError("Search failed (dimension mismatch or index not loaded?)")
        return out_ids[:n], out_dists[:n]

    # ── insert ────────────────────────────────────────────────────────────────

    def insert(self, vector_id: int, vec: Sequence[float] | np.ndarray) -> None:
        """Insert a vector. WAL and compaction are disabled from Python by default."""
        self._check_open()
        v = np.asarray(vec, dtype=np.float32).ravel()
        ok = self._lib.amio_store_insert(
            self._handle,
            ctypes.c_uint64(vector_id),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(v.size),
        )
        if not ok:
            raise RuntimeError(f"Insert failed for id={vector_id}")

    # ── observability ─────────────────────────────────────────────────────────

    def metrics(self) -> Metrics:
        self._check_open()
        raw = _AmioMetrics()
        self._lib.amio_store_metrics(self._handle, ctypes.byref(raw))
        return Metrics(
            cache_hits=raw.cache_hits,
            cache_misses=raw.cache_misses,
            disk_reads=raw.disk_reads,
            prefetch_submitted=raw.prefetch_submitted,
            theta_phase_switches=raw.theta_phase_switches,
            useful_prefetches=raw.useful_prefetches,
            wasted_prefetches=raw.wasted_prefetches,
            static_pins=raw.static_pins,
            uring_active=bool(raw.uring_active),
        )

    def reset_metrics(self) -> None:
        self._check_open()
        self._lib.amio_store_reset_metrics(self._handle)

    @property
    def partition_profile(self) -> str:
        self._check_open()
        raw = self._lib.amio_store_partition_profile(self._handle)
        return raw.decode() if raw else "UNKNOWN"

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        if not self._closed:
            self._lib.amio_store_close(self._handle)
            self._closed = True

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("VectorStore is already closed")

    def __repr__(self) -> str:
        return (
            f"VectorStore(closed={self._closed}, "
            f"profile={self.partition_profile if not self._closed else '?'})"
        )


# ── standalone file utilities ─────────────────────────────────────────────────

_KIND_NAMES = {0: "unknown", 1: "fvecs", 2: "bvecs", 3: "ivecs"}


def detect_vector_kind(path: str) -> str:
    """Return 'fvecs', 'bvecs', 'ivecs', or 'unknown'."""
    lib = _get_lib()
    k = lib.amio_detect_vector_kind(path.encode())
    return _KIND_NAMES.get(k, "unknown")


def probe_vector_file(path: str) -> tuple[int, int]:
    """Return (dim, num_vectors) by reading the file header."""
    lib = _get_lib()
    dim = ctypes.c_uint32(0)
    nvec = ctypes.c_uint64(0)
    ok = lib.amio_probe_vector_file(
        path.encode(), ctypes.byref(dim), ctypes.byref(nvec)
    )
    if not ok:
        raise ValueError(f"Cannot probe vector file: {path}")
    return dim.value, nvec.value
