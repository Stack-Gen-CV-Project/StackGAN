"""Stable Diffusion wrapper for the Gradio demo.

Defaults to the **ungated** `stable-diffusion-v1-5/stable-diffusion-v1-5`
community mirror so the demo runs without any HuggingFace token. As of late
2024 the original Stability AI checkpoints (`stabilityai/stable-diffusion-2-1`,
`...-2-1-base`, `runwayml/stable-diffusion-v1-5`) became gated/removed —
attempting to load them without auth returns 401.

Override via:
  - the `--sd-model-id` CLI flag on app.py, or
  - the `SD_MODEL_ID` environment variable, or
  - constructing SDPipeline(model_id=...) directly.

If the chosen model is gated, set the `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`)
env var to a token from <https://huggingface.co/settings/tokens> and accept
the model's license on its HF page.

Configured for the T4 16 GB Colab path: fp16 weights, DPMSolverMultistep
scheduler, 25 inference steps, safety checker disabled (course demo).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from PIL import Image


# Ungated community mirror — works without a HuggingFace token.
DEFAULT_SD_MODEL_ID = os.environ.get(
    "SD_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5"
)

# Backwards-compat alias used elsewhere in the project.
SD_MODEL_ID = DEFAULT_SD_MODEL_ID


def _hf_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


class SD21Pipeline:
    """Lazy-loading wrapper around diffusers.StableDiffusionPipeline.

    Class name kept for import-compat; works for any SD checkpoint
    (1.5 / 2.1 / 2.1-base / SDXL fine-tunes that share the SD pipeline API).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_SD_MODEL_ID,
        device: str | torch.device = "cuda",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self._pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        token = _hf_token()
        try:
            self._pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                cache_dir=self.cache_dir,
                token=token,
            )
        except Exception as exc:
            msg = str(exc)
            if "401" in msg or "gated" in msg.lower() or "Repository Not Found" in msg:
                raise RuntimeError(
                    f"Could not load '{self.model_id}'. It is likely gated.\n"
                    "Either:\n"
                    "  (a) set HF_TOKEN to a token from "
                    "https://huggingface.co/settings/tokens AND accept the model's "
                    "license on its HuggingFace page, or\n"
                    "  (b) use the default ungated model "
                    f"({DEFAULT_SD_MODEL_ID!r}) by removing --sd-model-id and "
                    "unsetting SD_MODEL_ID."
                ) from exc
            raise

        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config
        )
        self._pipe = self._pipe.to(self.device)
        # Memory savings on T4: don't precompute full attention maps.
        try:
            self._pipe.enable_attention_slicing()
        except Exception:
            pass

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        height: int = 512,
        width: int = 512,
    ) -> Image.Image:
        self._ensure_loaded()
        gen = (
            torch.Generator(device=self.device).manual_seed(int(seed))
            if seed is not None
            else None
        )
        with torch.inference_mode():
            out = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
                height=height,
                width=width,
            )
        return out.images[0]


# Backwards-compat alias matching the new naming.
SDPipeline = SD21Pipeline
