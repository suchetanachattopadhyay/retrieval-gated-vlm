"""Lightweight linear adapter over frozen CLIP image embeddings.

512x512 + bias = 0.26M parameters, about 0.17% of CLIP's parameter count.
Identity-initialised, so an untrained adapter is a no-op and any measured gain
is attributable to training rather than to reprojection noise. CLIP itself is
never updated.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE, EMBED_DIM


class LinearAdapter(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        out = self.proj(x)
        return out / out.norm(dim=-1, keepdim=True)


def contrastive_loss(image_embeds, text_embeds, temperature: float = 0.07):
    """Symmetric InfoNCE over an in-batch image-caption pairing."""
    logits = image_embeds @ text_embeds.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train_adapter(image_embeds: torch.Tensor, text_embeds: torch.Tensor,
                  epochs: int = 8, batch_size: int = 128, lr: float = 1e-4,
                  seed: int = 0, verbose: bool = True) -> LinearAdapter:
    torch.manual_seed(seed)
    adapter = LinearAdapter(image_embeds.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    n = image_embeds.shape[0]

    adapter.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total, batches = 0.0, 0
        for i in range(0, n, batch_size):
            sel = perm[i : i + batch_size]
            loss = contrastive_loss(
                adapter(image_embeds[sel].to(DEVICE)), text_embeds[sel].to(DEVICE)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}  avg loss {total / batches:.4f}")

    return adapter.eval()


@torch.no_grad()
def recall_at_k(image_embeds: torch.Tensor, text_embeds: torch.Tensor,
                adapter: LinearAdapter | None = None, k_values=(1, 5, 10)) -> dict:
    """Text-to-image recall over the given pool.

    Caution: as used in the paper this is in-sample — the adapter is evaluated on
    the same pairs it was trained on. Held-out numbers would be lower. See the
    Limitations section of the paper.
    """
    img = image_embeds.to(DEVICE)
    txt = text_embeds.to(DEVICE)
    if adapter is not None:
        img = adapter(img)

    sims = txt @ img.T
    ranks = np.array([
        (torch.argsort(sims[i], descending=True) == i).nonzero(as_tuple=True)[0].item()
        for i in range(sims.shape[0])
    ])
    return {f"R@{k}": float((ranks < k).mean()) for k in k_values}
