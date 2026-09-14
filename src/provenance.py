"""Retrieval-gated query with a structured provenance record.

The record separates two things that are easy to conflate after the fact:

  * why a candidate reached the model  -> rank and similarity score from the gate
  * what the model concluded about it  -> per-candidate VLM output

Because the gate already makes an explicit ranking decision before the VLM runs,
emitting this costs nothing measurable against a ~1.4 s VLM call. Everything is
written to local storage; nothing leaves the device.

Reconstructed to match the schema in `results/day9_provenance_log.json`.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from embed import clean_output, describe_image, embed_query

VERIFY_PROMPT = "Does this image show: {query}? Answer yes or no."


def gated_query(query: str, images_pool, index, clip_model, clip_processor,
                vlm_model, vlm_processor, k: int = 5, index_name: str = "flat",
                vlm_name: str = "SmolVLM2-256M") -> tuple[list, dict]:
    """Run one gated query. Returns (candidate/answer pairs, provenance record)."""
    t0 = time.perf_counter()

    q_embed = embed_query(query, clip_model, clip_processor)
    scores, indices = index.search(q_embed, k)

    candidates = [
        {"image_idx": int(i), "similarity_score": float(s)}
        for s, i in zip(scores[0], indices[0])
    ]

    prompt = VERIFY_PROMPT.format(query=query)
    vlm_outputs = [
        clean_output(
            describe_image(images_pool[c["image_idx"]], vlm_model, vlm_processor, prompt)
        )
        for c in candidates
    ]

    record = {
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "index_config": {"name": index_name, "k": k},
        "retrieved_candidates": candidates,
        "vlm_model": vlm_name,
        "vlm_output": vlm_outputs,
        "pipeline_latency_s": time.perf_counter() - t0,
    }
    return list(zip(candidates, vlm_outputs)), record


def append_to_log(record: dict, path: str | Path) -> None:
    """Append one record to a JSON array log, creating it if absent."""
    path = Path(path)
    log = json.loads(path.read_text()) if path.exists() else []
    log.append(record)
    path.write_text(json.dumps(log, indent=2))


def affirmation_rate(log: list[dict]) -> float:
    """Fraction of (query, candidate) pairs the VLM answered affirmatively.

    This is the crude answer proxy reported in the paper. It is NOT accuracy: it
    is not conditioned on whether retrieval was correct. Computing the joint
    metric — affirmative answers that coincide with a correct retrieval, per
    domain — is flagged as future work in Appendix C and is not implemented here.
    """
    answers = [a for record in log for a in record["vlm_output"]]
    if not answers:
        return 0.0
    return sum(a.strip().lower().startswith("yes") for a in answers) / len(answers)
