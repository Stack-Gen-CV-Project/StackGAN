# StackGAN-v2 vs Stable Diffusion 2.1 — Side-by-Side Text-to-Image Demo

| StackGAN-v2 sample 1 | StackGAN-v2 sample 2 |
|---|---|
| ![](samples/stackgan_synthetic_bird_1.png) | ![](samples/stackgan_synthetic_bird_2.png) |

A Gradio app that takes a caption and generates the same scene with **two
text-to-image models from very different eras**, displayed side by side:

| Model | Year | Trained on | Input | Output |
|---|---|---|---|---|
| **StackGAN-v2** (StackGAN++) | 2017 | CUB-200-2011 (200 bird species) | Pre-computed char-CNN-RNN embedding (dropdown of CUB test captions) | 256 × 256 |
| **Stable Diffusion 2.1 base** | 2022 | LAION-5B (~5 B image-text pairs) | Free-text prompt | 512 × 512 |

The visible quality and generality gap is the demo's pedagogical point — a
2017 class-conditional GAN restricted to 200 bird species cannot generalize,
while a 2022 latent diffusion model handles arbitrary captions.

## Quick start (Colab T4 GPU — recommended)

1. Open `notebooks/colab_demo.ipynb` in Google Colab.
2. Set the runtime to **GPU (T4)**: `Runtime → Change runtime type → T4 GPU`.
3. **Kaggle API token** — needed once to fetch the CUB char-CNN-RNN
   embeddings pickle. The original 2017 Google Drive link is permanently dead;
   we now pull from a Kaggle mirror.
   1. Go to <https://www.kaggle.com/settings> → "Create New API Token". A
      `kaggle.json` will download.
   2. In the Colab notebook, the second cell prompts you to upload it.
4. Run all cells. The last cell launches Gradio with a public share link.

The first run downloads the StackGAN checkpoint (~80 MB), the embeddings
pickle (~120 MB), and SD 2.1 weights (~5 GB) into the Colab session storage.
Total cold-start time: ~4 minutes.

## Quick start (Windows / local CPU dev smoke)

For code-only testing on the Windows dev box (no GPU required, but
generations take 1–2 minutes per image — *not* the demo runtime):

```powershell
# 1. Create venv with the existing miniconda3 base Python (3.10+ recommended)
C:\ProgramData\Miniconda3\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install deps
pip install -r requirements.txt

# 3. Download StackGAN checkpoints (~270 MB, Google Drive)
python download_weights.py

# 4. Architecture-only smoke test (no weights needed)
python smoke_test.py

# 5. Run the demo on CPU
python app.py --device cpu --share false
```

Open <http://localhost:7860> in a browser.

> **Conda alternative.** The `environment.yml` works with `conda env create` if
> the default-channel ToS has been accepted on the machine. Conda 25+ requires
> `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`
> first, otherwise use the venv path above.

## File layout

```
.
├── app.py                          # Gradio entry point
├── sd21_pipeline.py                # Stable Diffusion 2.1 wrapper
├── download_weights.py             # gdown for StackGAN weights + embeddings
├── stackgan/
│   ├── model.py                    # StackGAN-v2 G_NET (ported, modernized)
│   ├── inference.py                # High-level idx → PIL.Image API
│   ├── dropdown_captions.json      # 20 curated CUB caption labels
│   └── embeddings/                 # downloaded by download_weights.py
│       └── char-CNN-RNN-embeddings.pickle
├── weights/                        # downloaded by download_weights.py
│   └── netG_210000.pth
├── notebooks/
│   └── colab_demo.ipynb            # one-cell Colab launcher
├── requirements.txt
├── environment.yml                 # conda env for Windows local dev
└── README.md
```

## Architecture notes — why StackGAN-v2 instead of v1

The course brief mentions **StackGAN** with two stages (64×64 → 256×256). We
use **StackGAN-v2 / StackGAN++** ([Zhang et al. 2018][v2]) for one practical
reason: **the v1 PyTorch port (`hanzhanggit/StackGAN-Pytorch`) ships only
COCO pretrained weights, not CUB**. The CUB pretrained weights for v1 only
exist for the original TensorFlow 0.12 implementation, which won't run on a
modern Colab Python 3.10 / PyTorch 2.x stack without a multi-day port.

StackGAN-v2 is a strict upgrade of v1, by the same authors, with publicly
available PyTorch CUB weights. It still uses:

- **char-CNN-RNN** pretrained text embeddings (1024-dim)
- **Conditioning Augmentation (CA)** for embedding stochasticity
- The same hierarchical 64×64 → 256×256 upsampling structure

The demo experience is identical — caption → 256×256 bird image — and the
"visible gap vs modern diffusion" point holds equally.

[v2]: https://arxiv.org/abs/1710.10916

## Known limitations

- **StackGAN cannot do free text.** It needs a 1024-dim char-CNN-RNN embedding
  that was pre-computed on the CUB test captions. The dropdown picks one of 20
  pre-computed embeddings. Free-text captions for StackGAN would require
  porting the char-CNN-RNN encoder from Lua/Torch7 — out of scope.
- **Kaggle account required for first-time setup.** The original Google Drive
  ID for `char-CNN-RNN-embeddings.pickle` is permanently dead (deprecated
  2014-era ID format). We mirror via the Kaggle dataset
  [text-to-image-cub-200-2011][kaggle-mirror], which needs a free Kaggle API
  token (one-time setup, see Quick start step 3).
- **Single-user demo.** `gr.Queue(max_size=1)` prevents concurrent T4 OOM.
  One grader at a time.
- **Colab sessions idle out after 90 minutes.** Use **Kaggle** (9-hour
  sessions) for actual TA grading demos if Colab is too aggressive.
- **No quantitative metrics.** This is a qualitative side-by-side demo per
  course scope. CLIP-Score is the cheapest metric to add later if needed.

## Configuration

`app.py` accepts:

| Flag | Default | Notes |
|---|---|---|
| `--device` | `cuda` | use `cpu` for Windows local dev |
| `--weights` | `weights/netG_210000.pth` | StackGAN-v2 generator |
| `--embeddings` | `stackgan/embeddings/char-CNN-RNN-embeddings.pickle` | CUB test embeddings |
| `--captions` | `stackgan/dropdown_captions.json` | dropdown labels & indices |
| `--share` | `true` | Gradio public share URL on launch |
| `--server-port` | `7860` | local port |
| `--no-stackgan` | off | run SD 2.1 only (e.g. before checkpoints download) |

## Credits

- StackGAN-v2 architecture & weights: [hanzhanggit/StackGAN-v2](https://github.com/hanzhanggit/StackGAN-v2)
- char-CNN-RNN text encoder (used to pre-compute embeddings): [reedscot/icml2016](https://github.com/reedscot/icml2016)
- Embeddings pickle mirror: [text-to-image-cub-200-2011][kaggle-mirror] (Kaggle)
- Stable Diffusion 2.1: [Stability AI](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
- CUB-200-2011 dataset: [Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

[kaggle-mirror]: https://www.kaggle.com/datasets/somthirthabhowmk2001/text-to-image-cub-200-2011
