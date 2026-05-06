"""Gradio demo: StackGAN-v2 (CUB-pretrained) vs Stable Diffusion 2.1 base.

Side-by-side text-to-image comparison.

Run locally (CPU dev smoke):     python app.py --device cpu --share=false
Run on Colab T4 GPU (default):   python app.py
Run on Kaggle (better timeouts): python app.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gradio as gr

from sd21_pipeline import DEFAULT_SD_MODEL_ID, SD21Pipeline
from stackgan import StackGANInference


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "netG_210000.pth"
DEFAULT_EMBEDDINGS = REPO_ROOT / "stackgan" / "embeddings" / "char-CNN-RNN-embeddings.pickle"
DEFAULT_CAPTIONS = REPO_ROOT / "stackgan" / "dropdown_captions.json"


DISCLAIMER = (
    "**StackGAN-v2 was trained only on the CUB-200-2011 bird dataset (2017).** "
    "It cannot draw anything that isn't a bird, and it expects pre-computed "
    "char-CNN-RNN text embeddings — that's why its caption is a dropdown, not "
    "free text. The Stable Diffusion side (2022) takes free text and was "
    "trained on LAION-5B. The visible quality and generality gap is the point "
    "of the demo."
)

SYNTHETIC_NOTE = (
    "_Note: char-CNN-RNN embeddings pickle not found — StackGAN is using "
    "deterministic synthetic embeddings derived from each dropdown index. "
    "StackGAN's bird-domain prior is strong enough that this still produces "
    "recognizable birds, but they aren't bound to specific CUB captions. "
    "Run `python download_weights.py` with a Kaggle token to enable the real "
    "CUB embeddings._"
)

EXAMPLE_FREE_TEXT = [
    "a photograph of a cardinal perched on a snowy branch",
    "a small yellow songbird singing on a green leaf, photo",
    "a great horned owl on a tree at night, cinematic photo",
    "a corgi running through a sunflower field, dslr photo",
    "an astronaut riding a horse on Mars, digital art",
    "a cup of coffee on a wooden table, soft morning light",
]


def _bool_arg(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    p.add_argument("--captions", default=str(DEFAULT_CAPTIONS))
    p.add_argument("--share", type=_bool_arg, default=True,
                   help="enable a Gradio share link (default true; set false for private LAN)")
    p.add_argument("--server-name", default="0.0.0.0")
    p.add_argument("--server-port", type=int, default=7860)
    p.add_argument("--no-stackgan", action="store_true",
                   help="run with only the SD panel (skips StackGAN load if weights missing)")
    p.add_argument("--sd-model-id", default=DEFAULT_SD_MODEL_ID,
                   help=("HuggingFace model id for the diffusion side. Default is the "
                         "ungated `stable-diffusion-v1-5/stable-diffusion-v1-5`. "
                         "Override to e.g. `stabilityai/stable-diffusion-2-1-base` "
                         "if you've accepted its license and exported HF_TOKEN."))
    return p.parse_args()


def build_app(args: argparse.Namespace) -> gr.Blocks:
    sd_model_id = getattr(args, "sd_model_id", DEFAULT_SD_MODEL_ID)
    print(f"[init] diffusion model id = {sd_model_id}")
    sd_pipe = SD21Pipeline(model_id=sd_model_id, device=args.device)

    stackgan: StackGANInference | None = None
    if not args.no_stackgan:
        weights_ok = Path(args.weights).exists()
        embeddings_ok = Path(args.embeddings).exists()
        if weights_ok:
            print(f"[init] loading StackGAN-v2 from {args.weights}"
                  + ("" if embeddings_ok else " (synthetic-embedding fallback)"))
            stackgan = StackGANInference(
                weights_path=args.weights,
                embeddings_path=args.embeddings if embeddings_ok else None,
                captions_json_path=args.captions,
                device=args.device,
            )
        else:
            print(
                "[init] StackGAN weights not found at "
                f"{args.weights}. Run `python download_weights.py` first. "
                "Continuing with SD 2.1 only."
            )

    caption_choices = stackgan.caption_labels() if stackgan is not None else []
    default_caption = caption_choices[0] if caption_choices else None

    def run_stackgan(label: str | None, seed: int):
        if stackgan is None:
            return None, "StackGAN not loaded — run download_weights.py."
        if not label:
            return None, "Pick a caption from the dropdown."
        t0 = time.perf_counter()
        img = stackgan.generate_by_label(label, seed=int(seed))
        dt = time.perf_counter() - t0
        return img, f"StackGAN-v2 · {dt:.2f} s · 256×256"

    def run_sd(prompt: str, seed: int):
        prompt = (prompt or "").strip()
        if not prompt:
            return None, "Enter a free-text caption."
        t0 = time.perf_counter()
        try:
            img = sd_pipe.generate(prompt, seed=int(seed))
        except Exception as exc:
            return None, f"**Stable Diffusion error:** {exc}"
        dt = time.perf_counter() - t0
        return img, f"Stable Diffusion ({sd_model_id}) · {dt:.2f} s · 512×512"

    def run_both(label: str | None, prompt: str, seed: int):
        sg_img, sg_info = run_stackgan(label, seed)
        sd_img, sd_info = run_sd(prompt, seed)
        return sg_img, sg_info, sd_img, sd_info

    sd_short = sd_model_id.split("/")[-1]
    with gr.Blocks(title=f"StackGAN-v2 vs {sd_short}") as demo:
        synthetic_banner = ""
        if stackgan is not None and stackgan.using_synthetic_embeddings:
            synthetic_banner = "\n\n" + SYNTHETIC_NOTE
        gr.Markdown(
            f"# Text-to-Image: StackGAN-v2 (CUB) vs {sd_short}\n\n"
            f"{DISCLAIMER}{synthetic_banner}"
        )

        with gr.Row():
            with gr.Column(scale=1):
                cap_dropdown = gr.Dropdown(
                    choices=caption_choices,
                    value=default_caption,
                    label="StackGAN caption (CUB test set)",
                    interactive=stackgan is not None,
                )
                free_text = gr.Textbox(
                    label="Stable Diffusion caption (free text)",
                    placeholder="a small yellow songbird singing on a green leaf, photo",
                    value=EXAMPLE_FREE_TEXT[0],
                    lines=2,
                )
                seed_in = gr.Number(label="Seed", value=42, precision=0)
                go = gr.Button("Generate side-by-side", variant="primary")
                gr.Examples(
                    label="Free-text examples (also try in the StackGAN dropdown)",
                    examples=[[t] for t in EXAMPLE_FREE_TEXT],
                    inputs=[free_text],
                )

            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### StackGAN-v2 (2017, CUB-only)")
                        sg_image = gr.Image(label="StackGAN-v2 output", type="pil")
                        sg_info = gr.Markdown("Pick a caption and press Generate.")
                    with gr.Column():
                        gr.Markdown(f"### Stable Diffusion ({sd_short})\n2022 latent diffusion · LAION-5B")
                        sd_image = gr.Image(label="SD output", type="pil")
                        sd_info = gr.Markdown("Type a prompt and press Generate.")

        go.click(
            fn=run_both,
            inputs=[cap_dropdown, free_text, seed_in],
            outputs=[sg_image, sg_info, sd_image, sd_info],
        )

    return demo


def main() -> None:
    args = parse_args()
    app = build_app(args)
    # Single-slot queue prevents concurrent T4 OOM if multiple graders click at once.
    app.queue(max_size=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
