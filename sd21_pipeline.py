"""Stable Diffusion 2.1 base wrapper for the Gradio demo.

Uses the ungated `stabilityai/stable-diffusion-2-1-base` checkpoint (512x512
output) so the demo runs without a HuggingFace auth token. Configured for the
T4 16 GB Colab path: fp16 weights, DPMSolverMultistep scheduler, 25 inference
steps, safety checker disabled (course demo, internal use).
"""

from __future__ import annotations

from typing import Optional

import torch
from PIL import Image


SD_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"


class SD21Pipeline:
    """Lazy-loading wrapper around diffusers.StableDiffusionPipeline."""

    def __init__(
        self,
        model_id: str = SD_MODEL_ID,
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
        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=self.cache_dir,
        )
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
        gen = torch.Generator(device=self.device).manual_seed(int(seed)) if seed is not None else None
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
