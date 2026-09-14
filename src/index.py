"""FAISS index construction, query/recall evaluation, and maintenance benchmarks.

All indexes use inner product on L2-normalized vectors, which is cosine
similarity. Flat is exact and serves as ground truth for the recall proxy.
"""
from __future__ import annotations

import time

import faiss
import numpy as np

from config import IndexConfig


def build_flat(vectors: np.ndarray):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_ivf(vectors: np.ndarray, cfg: IndexConfig = IndexConfig()):
    d = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, cfg.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = cfg.ivf_nprobe
    return index


def build_hnsw(vectors: np.ndarray, cfg: IndexConfig = IndexConfig()):
    index = faiss.IndexHNSWFlat(vectors.shape[1], cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = cfg.hnsw_ef_search
    index.add(vectors)
    return index


BUILDERS = {"flat": build_flat, "ivf": build_ivf, "hnsw": build_hnsw}


def evaluate_index(index, queries: np.ndarray, ground_truth: np.ndarray,
                   k: int = 10, n_repeats: int = 5) -> dict:
    """Mean per-query latency and recall proxy against exact search.

    The recall proxy is set overlap with Flat's top-k, not ground-truth
    relevance — it measures how much the approximation costs, not retrieval
    quality in an absolute sense.
    """
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _, indices = index.search(queries, k)
        times.append(time.perf_counter() - t0)

    overlaps = [
        len(set(indices[i].tolist()) & set(ground_truth[i].tolist())) / k
        for i in range(len(queries))
    ]
    return {
        "avg_query_latency_ms": float(np.mean(times) / len(queries) * 1000),
        "std_query_latency_ms": float(np.std(times) / len(queries) * 1000),
        "recall@k_vs_flat": float(np.mean(overlaps)),
    }


def sample_queries(vectors: np.ndarray, n_queries: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vectors), size=min(n_queries, len(vectors)), replace=False)
    return vectors[idx]


def time_incremental_inserts(builder, base_vectors: np.ndarray, n_batches: int = 10,
                             batch_size: int = 10, seed: int = 0):
    """Wall time to append one batch of vectors to an already-built index.

    Offline archives are append-heavy and query-light, so this cost can matter
    more than query latency. HNSW is roughly 40x Flat here.
    """
    rng = np.random.default_rng(seed)
    d = base_vectors.shape[1]
    index = builder(base_vectors)
    per_batch_ms = []
    for _ in range(n_batches):
        new = rng.normal(size=(batch_size, d)).astype("float32")
        new /= np.linalg.norm(new, axis=1, keepdims=True)
        t0 = time.perf_counter()
        index.add(new)
        per_batch_ms.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(per_batch_ms)), float(np.std(per_batch_ms))


def time_full_rebuild(builder, base_vectors: np.ndarray, n_repeats: int = 5) -> float:
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        builder(base_vectors)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def maintenance_report(vectors: np.ndarray) -> dict:
    return {
        name: {
            "incremental_mean_ms_per_batch": (stats := time_incremental_inserts(fn, vectors))[0],
            "incremental_std_ms_per_batch": stats[1],
            "full_rebuild_ms": time_full_rebuild(fn, vectors),
        }
        for name, fn in BUILDERS.items()
    }
