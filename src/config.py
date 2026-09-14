"""Shared configuration and frozen-model loading.

Every component in this pipeline is frozen except the linear adapter in
`adapter.py`. CLIP and SmolVLM2 are loaded once and never updated.
"""
from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass

import torch

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*processor_kwargs.*")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_MODEL = "openai/clip-vit-base-patch32"
VLM_MODEL = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

EMBED_DIM = 512
MAX_NEW_TOKENS = 64

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")

# Flickr30k test split, first N_DATASET_A images (general photographs).
N_DATASET_A = 2000

# BDD100K scenario-classification val/city street (driving scenes).
# Note the space in the directory name.
N_DATASET_B = 500
DATASET_B_PATH = os.environ.get(
    "DATASET_B_PATH",
    "/kaggle/input/datasets/marquis03/bdd100k-scenario-classification/val/city street",
)


@dataclass
class IndexConfig:
    """Index hyperparameters as swept in the paper."""

    ivf_nlist: int = 64
    ivf_nprobe: int = 10
    hnsw_m: int = 32
    hnsw_ef_search: int = 64


def load_clip():
    """Load frozen CLIP ViT-B/32."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return model, processor


def load_vlm(quantization: str | None = None):
    """Load frozen SmolVLM2-256M.

    quantization: None/"fp16" | "int8" | "nf4"
    """
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    processor = AutoProcessor.from_pretrained(VLM_MODEL)

    if quantization in (None, "fp16"):
        model = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL, torch_dtype=torch.float16
        ).to(DEVICE)
    elif quantization == "int8":
        model = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
    elif quantization == "nf4":
        model = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map="auto",
        )
    else:
        raise ValueError(f"unknown quantization: {quantization!r}")

    return model.eval(), processor
