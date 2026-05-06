"""Architecture-only smoke test — does NOT require downloaded weights.

Builds G_NET with the same config as the CUB pretrained run and runs one
forward pass on random noise + a random text embedding, checking that the
3 fake_imgs returned have the expected shapes (64, 128, 256).

Useful as the very first thing to run after `pip install -r requirements.txt`
to confirm the port loads cleanly under your installed PyTorch.
"""

from __future__ import annotations

import sys

import torch

from stackgan.model import G_NET


def main() -> int:
    print(f"torch {torch.__version__}")
    netG = G_NET()
    netG.eval()
    n_params = sum(p.numel() for p in netG.parameters())
    print(f"G_NET parameters: {n_params/1e6:.2f} M")

    z = torch.randn(2, 100)
    text_emb = torch.randn(2, 1024)

    with torch.no_grad():
        fake_imgs, mu, logvar = netG(z, text_emb)

    expected = [(2, 3, 64, 64), (2, 3, 128, 128), (2, 3, 256, 256)]
    for img, exp in zip(fake_imgs, expected):
        actual = tuple(img.shape)
        ok = actual == exp
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] expected {exp}, got {actual}")
        if not ok:
            return 1

    print("Forward pass OK. Architecture matches StackGAN-v2 CUB config.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
