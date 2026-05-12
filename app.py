import argparse
import time
from pathlib import Path

import gradio as gr

from sd21_pipeline import DEFAULT_SD_MODEL_ID, SDPipeline
from stackgan import StackGANInference


HERE = Path(__file__).parent
WEIGHTS = HERE / "weights" / "netG_210000.pth"
EMBEDDINGS = HERE / "stackgan" / "embeddings" / "char-CNN-RNN-embeddings.pickle"
CAPTIONS_JSON = HERE / "stackgan" / "dropdown_captions.json"

EXAMPLE_PROMPTS = [
    "a small yellow songbird singing on a green leaf, photo",
    "a great horned owl on a tree at night, cinematic photo",
    "a photograph of a cardinal perched on a snowy branch",
    "a corgi running through a sunflower field, dslr photo",
    "an astronaut riding a horse on Mars, digital art",
    "a cup of coffee on a wooden table, soft morning light",
    "a parrot with colorful feathers sitting on a branch, detailed photo",
    "a peacock spreading its tail feathers, golden hour light",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--sd-model-id", default=DEFAULT_SD_MODEL_ID,
                   help="HuggingFace model id (default: SDXL Turbo)")
    p.add_argument("--share", default="false",
                   help="set 'true' for a public Gradio share URL")
    return p.parse_args()


def load_stackgan(device):
    if not WEIGHTS.exists():
        print(f"StackGAN weights not found at {WEIGHTS}. Run: python download_weights.py")
        return None
    emb = EMBEDDINGS if EMBEDDINGS.exists() else None
    if emb is None:
        print("CUB embeddings not found - using synthetic fallback.")
    return StackGANInference(
        weights_path=str(WEIGHTS),
        embeddings_path=str(emb) if emb else None,
        captions_json_path=str(CAPTIONS_JSON),
        device=device,
    )


def main():
    args = parse_args()
    stackgan = load_stackgan(args.device)
    sd_pipe = SDPipeline(model_id=args.sd_model_id, device=args.device)

    caption_choices = stackgan.caption_labels() if stackgan else []
    default_caption = caption_choices[0] if caption_choices else None

    def run_stackgan(label, seed):
        if stackgan is None:
            return None, "StackGAN not loaded."
        if not label:
            return None, "Pick a caption from the dropdown."
        t0 = time.time()
        img = stackgan.generate_by_label(label, seed=int(seed))
        return img, f"StackGAN-v2 — {time.time() - t0:.1f}s — 256×256"

    def run_sd(prompt, seed):
        prompt = (prompt or "").strip()
        if not prompt:
            return None, "Type a prompt."
        t0 = time.time()
        try:
            img = sd_pipe.generate(prompt, seed=int(seed))
        except Exception as e:
            return None, f"SDXL Turbo error: {e}"
        return img, f"SDXL Turbo — {time.time() - t0:.1f}s — 512×512"

    def run_both(label, prompt, seed):
        sg_img, sg_info = run_stackgan(label, seed)
        sd_img, sd_info = run_sd(prompt, seed)
        return sg_img, sg_info, sd_img, sd_info

    with gr.Blocks(title="StackGAN-v2 vs SDXL Turbo") as demo:
        gr.Markdown(
            "# StackGAN-v2 vs SDXL Turbo\n\n"
            "StackGAN-v2 (2017) only draws birds from the CUB dataset — "
            "pick a pre-computed embedding from the dropdown. "
            "SDXL Turbo (2023) accepts any free text and generates in 4 steps. "
            "Same caption, side by side."
        )
        with gr.Row():
            with gr.Column(scale=1):
                cap_dropdown = gr.Dropdown(
                    choices=caption_choices, value=default_caption,
                    label="StackGAN caption (CUB)",
                    interactive=stackgan is not None,
                )
                free_text = gr.Textbox(
                    label="SDXL Turbo prompt",
                    value=EXAMPLE_PROMPTS[0], lines=2,
                )
                seed_in = gr.Number(label="Seed", value=42, precision=0)
                go = gr.Button("Generate", variant="primary")
                gr.Examples([[p] for p in EXAMPLE_PROMPTS], inputs=[free_text])

            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### StackGAN-v2 (2017, CUB only)")
                        sg_image = gr.Image(label="output", type="pil")
                        sg_info = gr.Markdown("Pick a caption and press Generate.")
                    with gr.Column():
                        gr.Markdown("### SDXL Turbo (2023, any text)")
                        sd_image = gr.Image(label="output", type="pil")
                        sd_info = gr.Markdown("Type a prompt and press Generate.")

        go.click(run_both, inputs=[cap_dropdown, free_text, seed_in],
                 outputs=[sg_image, sg_info, sd_image, sd_info])

    demo.launch(server_name="0.0.0.0", share=args.share.lower() in ("1", "true", "yes"))


if __name__ == "__main__":
    main()
