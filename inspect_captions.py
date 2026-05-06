"""Helper: render one StackGAN-v2 generation per dropdown caption to disk.

After running `download_weights.py`, run this once to preview what each entry
in `stackgan/dropdown_captions.json` actually generates. Useful for hand-
curating the dropdown list to favor indices that produce the cleanest birds.

Outputs to `outputs/preview/<label>.png` and prints the timing of each.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from stackgan import StackGANInference


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "netG_210000.pth"
DEFAULT_EMBEDDINGS = REPO_ROOT / "stackgan" / "embeddings" / "char-CNN-RNN-embeddings.pickle"
DEFAULT_CAPTIONS = REPO_ROOT / "stackgan" / "dropdown_captions.json"
DEFAULT_OUT = REPO_ROOT / "outputs" / "preview"


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    p.add_argument("--captions", default=str(DEFAULT_CAPTIONS))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--scan", type=int, default=0,
        help="if >0, ignore captions JSON and render every Nth pickle index up to this count"
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sg = StackGANInference(
        weights_path=args.weights,
        embeddings_path=args.embeddings,
        captions_json_path=args.captions,
        device=args.device,
    )

    if args.scan > 0:
        items = [
            {"label": f"scan_idx_{i}", "image_idx": i, "caption_idx": 0}
            for i in range(0, args.scan, max(1, sg.num_images // args.scan))
        ]
    else:
        with open(args.captions, "r", encoding="utf-8") as f:
            items = json.load(f)

    for item in items:
        label = item["label"]
        idx = int(item["image_idx"])
        cap_idx = int(item.get("caption_idx", 0))
        t0 = time.perf_counter()
        img = sg.generate(idx, cap_idx, seed=args.seed)
        dt = time.perf_counter() - t0
        path = out_dir / f"{safe_filename(label)}.png"
        img.save(path)
        print(f"  [{dt:>5.2f}s] {path}")

    print(f"Wrote {len(items)} preview images to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
