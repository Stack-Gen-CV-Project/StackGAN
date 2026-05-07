# StackGAN-v2 vs Stable Diffusion

Small text-to-image demo for the Computer Vision course.
Runs StackGAN-v2 (2017, trained on CUB bird photos) and Stable Diffusion v1.5
(2022) side by side on the same caption so you can see the difference between
an old class-conditional GAN and a modern diffusion model.

| | |
| :-: | :-: |
| ![](samples/stackgan_synthetic_bird_1.png) | ![](samples/stackgan_synthetic_bird_2.png) |

Both models are pre-trained — there is no training code in this repo.

## What you need

- Python 3.10+
- ~5 GB free disk space (most of it is the Stable Diffusion checkpoint)
- GPU is nice (Colab T4 works); CPU also works but is slow

## Install

```
pip install -r requirements.txt
python download_weights.py
```

`download_weights.py` grabs:

- StackGAN-v2 generator from Google Drive (~80 MB)
- The CUB caption embeddings from Kaggle (~120 MB) — needs a Kaggle API token,
  see below. If you skip it the demo falls back to synthetic embeddings that
  still produce real-looking bird images.

Stable Diffusion v1.5 (~4 GB) is downloaded automatically by `diffusers` the
first time you click Generate.

## Run

```
python app.py
```

Then open <http://localhost:7860>.

For Google Colab, open `notebooks/colab_demo.ipynb` and run the cells.

## Kaggle token (optional)

The original Google Drive link for the CUB embeddings file is dead, so we
mirror it via Kaggle.

1. Make a Kaggle account (free).
2. <https://www.kaggle.com/settings> → "Create New API Token". This downloads
   `kaggle.json`.
3. Move it to:
   - Linux/Mac: `~/.kaggle/kaggle.json` (then `chmod 600`)
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - Colab: the notebook prompts you to upload it.
4. Re-run `python download_weights.py`.

## Use SD 2.1 (or another model) instead of v1.5

SD v1.5 is the default because it's the only major Stable Diffusion checkpoint
that doesn't need a HuggingFace token. To use SD 2.1:

1. Make a HuggingFace account.
2. Visit <https://huggingface.co/stabilityai/stable-diffusion-2-1-base> and
   click "Agree and access repository".
3. <https://huggingface.co/settings/tokens> → "New token" (Read).
4. Run with the token:
   ```
   # Linux / macOS
   HF_TOKEN=hf_xxx python app.py --sd-model-id stabilityai/stable-diffusion-2-1-base

   # Windows PowerShell
   $env:HF_TOKEN="hf_xxx"
   python app.py --sd-model-id stabilityai/stable-diffusion-2-1-base
   ```

You can pass any HuggingFace model id that uses the SD pipeline API
(`Lykon/dreamshaper-7`, `dreamlike-art/dreamlike-photoreal-2.0`, etc.).

## Files

```
app.py                              main Gradio app
sd21_pipeline.py                    Stable Diffusion wrapper
download_weights.py                 download script
stackgan/
  model.py                          StackGAN-v2 generator architecture
  inference.py                      load weights + generate
  dropdown_captions.json            20 dropdown entries for the demo
  embeddings/                       (created by download_weights.py)
weights/                            (created by download_weights.py)
notebooks/colab_demo.ipynb          Colab launcher
samples/                            example outputs
requirements.txt
```

## Credits

- StackGAN-v2 — <https://github.com/hanzhanggit/StackGAN-v2>
  (Zhang et al. 2018)
- char-CNN-RNN text encoder — <https://github.com/reedscot/icml2016>
  (Reed et al. 2016)
- CUB embeddings mirror — <https://www.kaggle.com/datasets/somthirthabhowmk2001/text-to-image-cub-200-2011>
- Stable Diffusion v1.5 — <https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5>
  (Rombach et al. 2022)
- CUB-200-2011 — <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>
