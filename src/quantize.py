"""Weight-precision ablation for SmolVLM2-256M: FP16 vs INT8 vs NF4.

The regime matters for interpreting these numbers: batch size one, one model
resident at a time, single request in flight. That is what an on-device
deployment looks like, and it is the regime in which INT8's mixed-precision
decomposition overhead has no throughput to hide behind. INT8 is a strict
regression against NF4 here — less memory saved *and* far slower.

These numbers do not transfer to a batched server deployment, and the paper does
not claim they do.
"""
from __future__ import annotations

import time

from config import load_vlm
from embed import clean_output, describe_image

PRECISIONS = ("fp16", "int8", "nf4")
DEFAULT_PROMPT = "Describe this image in one sentence."


def model_memory_mb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


def benchmark_precision(precision: str, images, prompt: str = DEFAULT_PROMPT) -> dict:
    """Load one precision variant, time it over `images`, then report."""
    model, processor = load_vlm(precision)

    describe_image(images[0], model, processor, prompt)  # warm-up, not timed

    t0 = time.perf_counter()
    outputs = [describe_image(img, model, processor, prompt) for img in images]
    elapsed = time.perf_counter() - t0

    return {
        "memory_mb": model_memory_mb(model),
        "avg_latency_s": elapsed / len(images),
        "sample_outputs": [clean_output(o) for o in outputs[:3]],
    }


def run_ablation(images, precisions=PRECISIONS) -> dict:
    """Full sweep. Load one variant at a time — three VLMs will not co-reside on a T4."""
    raw = {p: benchmark_precision(p, images) for p in precisions}

    base_mem = raw["fp16"]["memory_mb"]
    base_lat = raw["fp16"]["avg_latency_s"]
    return {
        "memory_mb": {p: r["memory_mb"] for p, r in raw.items()},
        "latency_s_per_image": {p: r["avg_latency_s"] for p, r in raw.items()},
        "memory_savings_vs_fp16_pct": {
            p: (base_mem - r["memory_mb"]) / base_mem * 100 for p, r in raw.items()
        },
        "latency_ratio_vs_fp16": {
            p: r["avg_latency_s"] / base_lat for p, r in raw.items()
        },
        "sample_outputs": {p: r["sample_outputs"] for p, r in raw.items()},
    }
