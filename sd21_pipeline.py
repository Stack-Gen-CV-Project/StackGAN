import os

import torch


DEFAULT_SD_MODEL_ID = "stabilityai/sdxl-turbo"


class SDPipeline:
    def __init__(self, model_id=DEFAULT_SD_MODEL_ID, device="cuda"):
        self.model_id = model_id
        self.device = torch.device(device)
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        from diffusers import AutoPipelineForText2Image

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        token = os.environ.get("HF_TOKEN")

        pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            token=token,
            variant="fp16" if dtype == torch.float16 else None,
        )
        pipe = pipe.to(self.device)
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        self._pipe = pipe

    def generate(self, prompt, seed=None, height=512, width=512):
        self._ensure_loaded()
        gen = (
            torch.Generator(device=self.device).manual_seed(int(seed))
            if seed is not None else None
        )
        is_turbo = "turbo" in self.model_id.lower()
        with torch.inference_mode():
            out = self._pipe(
                prompt=prompt,
                num_inference_steps=4 if is_turbo else 25,
                guidance_scale=0.0 if is_turbo else 7.5,
                generator=gen,
                height=height,
                width=width,
            )
        return out.images[0]
