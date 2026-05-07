"""Load StackGAN-v2 weights and generate 256x256 images."""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import G_NET


# Empirically a stddev of 0.5 produces clearly bird-shaped outputs from the
# CUB pretrained generator when no real embedding is available.
SYNTHETIC_EMBEDDING_SCALE = 0.5


class StackGANInference:
    def __init__(self, weights_path, embeddings_path=None,
                 captions_json_path=None, device="cpu"):
        self.device = torch.device(device)
        self.weights_path = Path(weights_path)
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None

        self._load_embeddings()
        self._load_model()
        self._load_captions(captions_json_path)

    @property
    def using_synthetic_embeddings(self):
        return self.embeddings is None

    def _load_embeddings(self):
        if self.embeddings_path is None or not self.embeddings_path.exists():
            self.embeddings = None
            self.num_images = 10000
            self.captions_per_image = 1
            return
        with open(self.embeddings_path, "rb") as f:
            embeddings = pickle.load(f, encoding="latin1")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.embeddings = embeddings
        self.num_images = embeddings.shape[0]
        self.captions_per_image = embeddings.shape[1]

    def _load_model(self):
        netG = G_NET()
        sd = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # Strip "module." prefix from DataParallel-saved checkpoints.
        cleaned = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }
        netG.load_state_dict(cleaned, strict=False)
        netG.eval().to(self.device)
        self.netG = netG

    def _load_captions(self, path):
        if path is None:
            self.captions = []
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.captions = json.load(f)
        except FileNotFoundError:
            self.captions = []

    def caption_labels(self):
        return [c["label"] for c in self.captions]

    def lookup_caption(self, label):
        for c in self.captions:
            if c["label"] == label:
                return c
        raise KeyError(label)

    def _embedding_for(self, image_idx, caption_idx):
        if self.embeddings is not None:
            emb = self.embeddings[image_idx, caption_idx, :]
            return torch.from_numpy(emb).float().unsqueeze(0).to(self.device)

        # Fallback: deterministic synthetic embedding so the demo still works
        # without the Kaggle pickle. StackGAN's bird-prior is strong enough
        # that random embeddings still produce recognizable birds.
        g = torch.Generator()
        g.manual_seed(int(image_idx) * 100003 + int(caption_idx) * 1009 + 17)
        emb = torch.randn(1, 1024, generator=g) * SYNTHETIC_EMBEDDING_SCALE
        return emb.to(self.device)

    @torch.no_grad()
    def generate(self, image_idx, caption_idx=0, seed=None):
        emb = self._embedding_for(image_idx, caption_idx)

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))
        z = torch.randn(1, 100, generator=g).to(self.device)

        fake_imgs, _, _ = self.netG(z, emb)
        img = fake_imgs[-1][0]  # (3, 256, 256), values in [-1, 1]
        arr = ((img + 1) / 2 * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)

    def generate_by_label(self, label, seed=None):
        c = self.lookup_caption(label)
        return self.generate(c["image_idx"], c.get("caption_idx", 0), seed=seed)
