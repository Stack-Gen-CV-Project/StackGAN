"""End-to-end smoke test of the Gradio app's StackGAN path WITHOUT launching
a server. Builds the Blocks instance (which loads StackGAN weights into RAM)
and directly invokes the dropdown-driven generation function. Saves the
result image to outputs/app_smoke.png. Skips SD 2.1 to avoid a 5 GB download
on the dev box.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr  # noqa: F401  imported via app

import app as app_module
from stackgan import StackGANInference


REPO_ROOT = Path(__file__).resolve().parent


def main() -> int:
    args = argparse.Namespace(
        device="cpu",
        weights=str(REPO_ROOT / "weights" / "netG_210000.pth"),
        embeddings=str(REPO_ROOT / "stackgan" / "embeddings" / "char-CNN-RNN-embeddings.pickle"),
        captions=str(REPO_ROOT / "stackgan" / "dropdown_captions.json"),
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        no_stackgan=False,
    )

    # 1. Direct exercise of StackGANInference (bypasses the Gradio plumbing).
    sg = StackGANInference(
        weights_path=args.weights,
        embeddings_path=args.embeddings if Path(args.embeddings).exists() else None,
        captions_json_path=args.captions,
        device="cpu",
    )
    print(f"  using_synthetic_embeddings = {sg.using_synthetic_embeddings}")
    print(f"  num caption labels         = {len(sg.caption_labels())}")

    label = sg.caption_labels()[0]
    img = sg.generate_by_label(label, seed=42)
    out = REPO_ROOT / "outputs" / "app_smoke.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"  generated: {out} ({img.size[0]}x{img.size[1]})")

    # 2. Build the Gradio Blocks object — this will fail if there's an API
    # incompatibility with the installed gradio version. Don't launch().
    demo = app_module.build_app(args)
    print(f"  Gradio Blocks built: title={demo.title!r}")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
