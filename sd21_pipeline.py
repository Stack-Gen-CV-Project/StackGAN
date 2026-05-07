"""Stable Diffusion wrapper.

Default model is the ungated community mirror of SD v1.5. Override with
--sd-model-id or the SD_MODEL_ID env var. Use HF_TOKEN for gated models
(SD 2.1, SDXL-Turbo, etc.).
"""

import os

import torch
from PIL import Image


DEFAULT_SD_MODEL_ID = os.environ.get(
    "SD_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5"
)


def _hf_token():
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


class SD21Pipeline:
    """Lazy-loading SD wrapper. Class name kept for import-compat;
    works with any SD-compatible checkpoint."""

    def __init__(self, model_id=DEFAULT_SD_MODEL_ID, device="cuda", cache_dir=None):
        self.model_id = model_id
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                cache_dir=self.cache_dir,
                token=_hf_token(),
            )
        except Exception as e:
            msg = str(e)
            if "401" in msg or "gated" in msg.lower() or "Repository Not Found" in msg:
                raise RuntimeError(
                    f"Could not load '{self.model_id}'. The model is probably gated. "
                    "Either remove --sd-model-id (the default works without a token) "
                    "or set HF_TOKEN to a token from "
                    "https://huggingface.co/settings/tokens "
                    "and accept the model's license on its HuggingFace page."
                ) from e
            raise

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        self._pipe = pipe

    def generate(self, prompt, seed=None, num_inference_steps=25,
                 guidance_scale=7.5, height=512, width=512):
        self._ensure_loaded()
        gen = (
            torch.Generator(device=self.device).manual_seed(int(seed))
            if seed is not None else None
        )
        with torch.inference_mode():
            out = self._pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen, height=height, width=width,
            )
        return out.images[0]
