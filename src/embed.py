"""CLIP embedding and SmolVLM2 inference helpers.

Note on the CLIP calls below: `get_image_features` / `get_text_features` are
bypassed deliberately. In the pinned `transformers` version they return a
`BaseModelOutputWithPooling` rather than the projected embedding, so the
projection is applied explicitly here. Silently using the pooled output instead
of the projected one degrades retrieval without raising anything.
"""
from __future__ import annotations

import numpy as np
import torch

from config import DEVICE, MAX_NEW_TOKENS


@torch.no_grad()
def embed_images(images, clip_model, clip_processor, batch_size: int = 64) -> np.ndarray:
    """PIL images -> L2-normalized [N, 512] float32 array."""
    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = clip_processor(images=batch, return_tensors="pt").to(DEVICE)
        vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        feats = clip_model.visual_projection(vision_out.pooler_output)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0).astype("float32")


@torch.no_grad()
def embed_texts(texts, clip_model, clip_processor, batch_size: int = 64) -> np.ndarray:
    """Text strings -> L2-normalized [N, 512] float32 array.

    Truncation is required: CLIP's text encoder has a hard 77-token limit and
    VLM-generated pseudo-captions routinely exceed it.
    """
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = clip_processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(DEVICE)
        text_out = clip_model.text_model(**inputs)
        feats = clip_model.text_projection(text_out.pooler_output)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0).astype("float32")


def embed_query(query: str, clip_model, clip_processor) -> np.ndarray:
    """Single query string -> [1, 512] float32 array, ready for `index.search`."""
    return embed_texts([query], clip_model, clip_processor)


@torch.no_grad()
def describe_image(image, vlm_model, vlm_processor, prompt: str) -> str:
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    text_prompt = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = vlm_processor(text=text_prompt, images=[image], return_tensors="pt").to(
        vlm_model.device
    )
    generated = vlm_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    return vlm_processor.batch_decode(generated, skip_special_tokens=True)[0]


def clean_output(raw: str) -> str:
    """Strip the echoed chat template, keep only the assistant turn."""
    if "Assistant:" in raw:
        return raw.split("Assistant:")[-1].strip()
    return raw.strip()
