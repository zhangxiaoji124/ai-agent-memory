"""
amio – Python bindings for the agent-memory-io-cpp vector store.

Quick start
-----------
from amio import VectorStore, read_fvecs, read_bvecs

# Read a dataset
vecs = read_fvecs("data/sift/sift_query.fvecs")   # (N, 128) float32

# Search
with VectorStore("data/sift_base.index", ef_search=192, cache_mb=512) as vs:
    print("Partition profile:", vs.partition_profile)
    ids, dists = vs.search_disk(vecs[0], k=10)
    m = vs.metrics()
    print(f"cache_hit_rate={m.cache_hit_rate:.2%}  "
          f"prefetch_utilization={m.prefetch_utilization:.2%}")
"""

from .fvecs_io import (
    iter_bvecs,
    iter_fvecs,
    probe,
    read_bvecs,
    read_fvecs,
    read_ivecs,
    write_bvecs,
    write_fvecs,
    write_ivecs,
)
from .vector_store import (
    Metrics,
    VectorStore,
    detect_vector_kind,
    probe_vector_file,
)

__all__ = [
    "VectorStore",
    "Metrics",
    "detect_vector_kind",
    "probe_vector_file",
    "read_fvecs",
    "read_bvecs",
    "read_ivecs",
    "write_fvecs",
    "write_bvecs",
    "write_ivecs",
    "iter_fvecs",
    "iter_bvecs",
    "probe",
]
