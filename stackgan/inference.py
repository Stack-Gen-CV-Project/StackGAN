import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import G_NET


class StackGANInference:
    def __init__(self, weights_path, embeddings_path=None,
                 captions_json_path=None, device="cpu"):
        self.device = torch.device(device)
        self._load_embeddings(embeddings_path)
        self._load_model(weights_path)
        self._load_captions(captions_json_path)

    @property
    def using_synthetic_embeddings(self):
        return self.embeddings is None

    def _load_embeddings(self, path):
        self.embeddings = None
        if path and Path(path).exists():
            with open(path, "rb") as f:
                emb = np.asarray(pickle.load(f, encoding="latin1"), dtype=np.float32)
            self.embeddings = emb

    def _load_model(self, path):
        netG = G_NET()
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        cleaned = {(k.replace("module.", "", 1) if k.startswith("module.") else k): v
                   for k, v in sd.items()}
        netG.load_state_dict(cleaned, strict=False)
        self.netG = netG.eval().to(self.device)

    def _load_captions(self, path):
        self.captions = []
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.captions = json.load(f)
            except FileNotFoundError:
                pass

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

        g = torch.Generator()
        g.manual_seed(int(image_idx) * 100003 + int(caption_idx) * 1009 + 17)
        return torch.randn(1, 1024, generator=g).to(self.device) * 0.5

    @torch.no_grad()
    def generate(self, image_idx, caption_idx=0, seed=None):
        emb = self._embedding_for(image_idx, caption_idx)
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))
        z = torch.randn(1, 100, generator=g).to(self.device)

        fake_imgs, _, _ = self.netG(z, emb)
        img = fake_imgs[-1][0]
        arr = ((img + 1) / 2 * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)

    def generate_by_label(self, label, seed=None):
        c = self.lookup_caption(label)
        return self.generate(c["image_idx"], c.get("caption_idx", 0), seed=seed)
