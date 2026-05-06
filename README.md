<div align="center">

# StackGAN-v2 vs Stable Diffusion 2.1
### A side-by-side text-to-image demo bridging 5 years of generative modeling

[![Python](https://img.shields.io/badge/python-3.10%20|%203.13-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/gradio-%E2%89%A55.0-FF7C00?logo=gradio)](https://gradio.app/)
[![HuggingFace](https://img.shields.io/badge/diffusers-SD_2.1-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stack-Gen-CV-Project/StackGAN/blob/main/notebooks/colab_demo.ipynb)

| StackGAN-v2 (2017, CUB) | StackGAN-v2 (2017, CUB) |
| :---: | :---: |
| ![](samples/stackgan_synthetic_bird_1.png) | ![](samples/stackgan_synthetic_bird_2.png) |

</div>

A Gradio web app that takes a caption and runs **two text-to-image models from very
different eras** side by side, on the same input. The visible quality and generality
gap between a 2017 class-conditional GAN restricted to 200 bird species and a 2022
latent diffusion model trained on 5 billion image-text pairs is the demo's
pedagogical point.

| Model | Year | Trained on | Input | Output | Params |
| --- | :---: | --- | --- | :---: | :---: |
| **StackGAN-v2** (StackGAN++) | 2017 | CUB-200-2011 (200 bird species) | Pre-computed `char-CNN-RNN` embedding (CUB-test dropdown) | 256 × 256 | ~21 M |
| **Stable Diffusion 2.1 base** | 2022 | LAION-5B (~5 B image-text pairs) | Free-text prompt | 512 × 512 | ~1.3 B |

---

## Table of contents

1. [Quick start (Colab — recommended)](#quick-start-colab--recommended)
2. [Quick start (local CPU dev)](#quick-start-local-cpu-dev)
3. [How it works](#how-it-works)
4. [Project structure](#project-structure)
5. [Smoke tests](#smoke-tests)
6. [Why StackGAN-v2 instead of v1](#why-stackgan-v2-instead-of-v1)
7. [Synthetic-embedding fallback](#synthetic-embedding-fallback)
8. [Configuration](#configuration)
9. [Known limitations](#known-limitations)
10. [Troubleshooting](#troubleshooting)
11. [Credits & citations](#credits--citations)

---

## Quick start (Colab — recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stack-Gen-CV-Project/StackGAN/blob/main/notebooks/colab_demo.ipynb)

1. Click the **Open in Colab** badge above (or open
   [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb) directly).
2. **Runtime → Change runtime type → T4 GPU**.
3. **Get a Kaggle API token** (one time, ~30 s):
   1. Go to <https://www.kaggle.com/settings> → *Create New API Token*.
      A `kaggle.json` will download.
   2. The Colab notebook's third cell prompts you to upload it.
   > Why Kaggle? The original Google Drive link for the CUB
   > `char-CNN-RNN-embeddings.pickle` (a 2014-era ID `0B3y_msr...`)
   > is **permanently dead**. We mirror via Kaggle.
4. **Run all cells.** The last cell prints a public Gradio share link.

**Cold-start budget on a fresh T4:**

| Step | Size | Time |
| --- | :---: | :---: |
| `pip install -r requirements.txt` | ~2 GB cached | ~60 s |
| StackGAN-v2 generator (Google Drive) | 79 MB | ~25 s |
| CUB embeddings pickle (Kaggle) | ~120 MB | ~30 s |
| Stable Diffusion 2.1 base (HF Hub) | ~5 GB | ~90 s |
| Gradio launch + share URL | — | ~10 s |
| **Total** | | **~3.5 min** |

Per-image inference timing on T4: **StackGAN ~0.4 s** · **SD 2.1 ~6 s** (25 DPM-Solver steps).

---

## Quick start (local CPU dev)

For code-only smoke testing on a Windows or Linux dev machine. No GPU required, but
inference is 50–100× slower than the T4 demo runtime — this path is for verifying
the pipeline imports and runs, not for actually demoing.

### Windows (PowerShell)

```powershell
# 1. Create venv (any Python 3.10+ works; below uses the bundled miniconda3 base).
C:\ProgramData\Miniconda3\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install deps
pip install -r requirements.txt

# 3. Architecture-only smoke test (no weights download needed)
python smoke_test.py

# 4. Download StackGAN generator (~80 MB) and CUB embeddings (~120 MB)
#    Skip the embeddings if you don't have a Kaggle token yet — the demo
#    will fall back to synthetic-embedding mode (still produces birds).
python download_weights.py
# or:                              (no Kaggle account)
python download_weights.py --skip-embeddings

# 5. End-to-end smoke test (no SD 2.1 download — exercises StackGAN only)
python app_smoke_test.py

# 6. Launch the full demo on CPU (slow — ~2 min per image)
python app.py --device cpu --share false
```

### Linux / macOS

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py
python download_weights.py
python app.py --device cpu --share false
```

Open <http://localhost:7860> when the Gradio banner prints.

> **Conda alternative.** `environment.yml` works with `conda env create -f environment.yml`,
> but Conda 25+ requires you to first accept the default-channel ToS:
> `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`.
> The venv path above sidesteps this.

---

## How it works

```mermaid
flowchart LR
  user["User caption"]
  user -->|free text| sd["Stable Diffusion 2.1 base
fp16 · DPMSolverMultistep · 25 steps"]
  user -->|"dropdown
(CUB test idx)"| pickle[("char-CNN-RNN
embeddings pickle
(2933, 10, 1024)")]
  pickle -->|"emb[idx]"| ca["CA_NET
1024 → 128 (μ, log σ²)
reparametrize"]
  z["z ~ N(0, I) (100-d)"] --> init
  ca --> init["INIT_STAGE_G
228 → 1024 × 4 × 4
4× upBlock"]
  init --> img1["64 × 64 image"]
  init --> next2["NEXT_STAGE_G
+CA, 2× ResBlock, upsample"]
  ca --> next2
  next2 --> img2["128 × 128"]
  next2 --> next3["NEXT_STAGE_G"]
  ca --> next3
  next3 --> img3["256 × 256 (used)"]
  sd --> sdimg["512 × 512"]
  img3 --> ui[["Gradio side-by-side"]]
  sdimg --> ui
```

Both pipelines stay resident on the T4 in fp16. Total VRAM peak is ~7 GB
(SD 2.1 ~5 GB + StackGAN G_NET ~200 MB + VAE decode peak ~1 GB) — comfortable on
the free 16 GB tier without `enable_model_cpu_offload`.

### StackGAN-v2 forward pass details

Architecture inlined from `hanzhanggit/StackGAN-v2/code/model.py` and modernized
for PyTorch 2.x in [`stackgan/model.py`](stackgan/model.py):

- `Variable(...)` wrapping → plain tensors with `torch.no_grad()` at inference.
- `F.sigmoid` → `torch.sigmoid` (the former is deprecated).
- `cfg.CUDA` device check inside `CA_NET.reparametrize` → tensor-derived
  (`torch.randn_like(std)`).
- All cfg values inlined as module-level constants matching `cfg/eval_birds.yml`:
  `Z_DIM=100`, `EMBEDDING_DIM=128`, `GF_DIM=64`, `R_NUM=2`, `B_CONDITION=True`,
  `BRANCH_NUM=3`, `TEXT.DIMENSION=1024`.

The state-dict key naming is preserved exactly, so the official
`netG_210000.pth` checkpoint loads with **0 missing / 0 unexpected keys** —
verified at startup.

### Stable Diffusion 2.1 wrapper

[`sd21_pipeline.py`](sd21_pipeline.py) wraps `diffusers.StableDiffusionPipeline` for
the **ungated** `stabilityai/stable-diffusion-2-1-base` checkpoint (so no HF auth
token is needed):

- `torch_dtype=torch.float16` on CUDA, `float32` on CPU.
- `DPMSolverMultistepScheduler` (25 steps default — high quality, ~6 s on T4).
- `safety_checker=None` for course demo use.
- `enable_attention_slicing()` to reduce peak VRAM.

---

## Project structure

```
.
├── app.py                              # Gradio Blocks entry point
├── sd21_pipeline.py                    # Stable Diffusion 2.1 wrapper (lazy-loading)
├── download_weights.py                 # Google Drive + Kaggle downloader (handles dead IDs)
├── smoke_test.py                       # Architecture-only forward-pass test
├── app_smoke_test.py                   # End-to-end test without launching a server
├── inspect_captions.py                 # Render preview images per dropdown index
├── stackgan/
│   ├── __init__.py                     # exports StackGANInference
│   ├── model.py                        # StackGAN-v2 G_NET (ported, modernized)
│   ├── inference.py                    # idx → PIL.Image API + synthetic-emb fallback
│   ├── dropdown_captions.json          # 20 curated dropdown entries
│   └── embeddings/                     # populated by download_weights.py
│       └── char-CNN-RNN-embeddings.pickle
├── weights/                            # populated by download_weights.py
│   └── netG_210000.pth                 # StackGAN-v2 CUB pretrained generator
├── notebooks/
│   └── colab_demo.ipynb                # T4 launcher with Kaggle credential UX
├── samples/                            # committed example outputs
│   ├── stackgan_synthetic_bird_1.png
│   └── stackgan_synthetic_bird_2.png
├── requirements.txt                    # pip dependencies
├── environment.yml                     # conda-forge env spec (alt to venv)
├── project_sprint_plan.html            # original 7-sprint scoping doc (obsolete)
└── README.md
```

---

## Smoke tests

The repo ships three test scripts you can run before downloading anything heavy:

| Script | What it verifies | Needs weights? | Needs SD 2.1? |
| --- | --- | :---: | :---: |
| `python smoke_test.py` | G_NET architecture builds, forward pass shapes are `(64, 128, 256)` | ❌ | ❌ |
| `python app_smoke_test.py` | Real checkpoint loads with 0 mismatched keys, dropdown → bird PIL, Gradio Blocks builds | ✅ | ❌ |
| `python download_weights.py` | gdown extraction handles the zip wrapper, Kaggle auth works | — | — |

Both `smoke_test.py` and `app_smoke_test.py` run in <30 s on CPU and are the
fastest sanity checks before committing changes.

---

## Why StackGAN-v2 instead of v1

The course brief item #7 mentions **StackGAN** with a two-stage 64×64 → 256×256
pipeline. We use **StackGAN-v2 / StackGAN++** ([Zhang et al. 2018][v2]) for one
practical reason: **the v1 PyTorch port (`hanzhanggit/StackGAN-Pytorch`) ships
only COCO pretrained weights, not CUB**. The CUB pretrained weights for v1 only
exist for the *original TensorFlow 0.12* implementation, which won't run on a
modern Colab Python 3.10 / PyTorch 2.x stack without a multi-day port.

StackGAN-v2 is a strict upgrade of v1 by the same authors, with publicly available
PyTorch CUB weights. It still uses:

- The same **char-CNN-RNN** pretrained text embeddings (1024-dim).
- The same **Conditioning Augmentation (CA)** for embedding stochasticity.
- The same hierarchical **64×64 → 128×128 → 256×256** upsampling structure
  (StackGAN++ joints the stages into a tree of generators producing all three
  resolutions in one forward pass).

The user-facing demo experience is identical — caption → 256×256 bird image —
and the "visible gap vs modern diffusion" point holds equally.

[v2]: https://arxiv.org/abs/1710.10916

---

## Synthetic-embedding fallback

The demo has two text-conditioning modes:

1. **Real CUB embeddings** (preferred): pulls
   `char-CNN-RNN-embeddings.pickle` from a Kaggle mirror, then indexes
   `embeddings[image_idx, caption_idx, :]` for each dropdown entry.
2. **Synthetic deterministic embeddings** (fallback when the pickle isn't
   present): each `image_idx` seeds a standard-normal 1024-dim vector scaled
   by 0.5. **Empirically this still produces clearly recognizable birds**
   because StackGAN's training distribution is so narrow — the generator
   has effectively learned a strong "bird manifold" prior, and arbitrary
   embeddings get smoothed onto it by the CA_NET.

The two `samples/` images at the top of this README were generated in
synthetic mode — proof that the demo gives meaningful output even before a
grader sets up Kaggle credentials.

```python
# stackgan/inference.py — _embedding_for():
if self.embeddings is None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(image_idx * 100003 + caption_idx * 1009 + 17)
    return (torch.randn(1, 1024, generator=gen) * 0.5).to(self.device)
```

---

## Configuration

`app.py` accepts the following CLI flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--device` | `cuda` | `cpu` for local dev, `cuda` on Colab |
| `--weights` | `weights/netG_210000.pth` | StackGAN-v2 generator path |
| `--embeddings` | `stackgan/embeddings/char-CNN-RNN-embeddings.pickle` | optional; falls back to synthetic if missing |
| `--captions` | `stackgan/dropdown_captions.json` | dropdown labels & pickle indices |
| `--share` | `true` | Gradio public share URL on launch |
| `--server-name` | `0.0.0.0` | local bind address |
| `--server-port` | `7860` | local port |
| `--no-stackgan` | (off) | run SD 2.1 only (e.g. before the StackGAN download) |

`download_weights.py`:

| Flag | Notes |
| --- | --- |
| `--force` | re-download even if local files exist |
| `--skip-embeddings` | skip Kaggle download entirely (use synthetic fallback) |
| `--root <path>` | project root (defaults to script directory) |

---

## Known limitations

- **StackGAN cannot do free text.** It needs a 1024-dim char-CNN-RNN embedding
  pre-computed on CUB test captions. The dropdown picks one of 20 curated
  pre-computed embeddings. True free-text input on the GAN side would require
  porting Reed et al.'s char-CNN-RNN encoder from Lua/Torch7 — out of scope.
- **The dropdown labels are illustrative, not paired ground truth** unless
  you run the optional curation step. Each entry maps a label to a CUB test
  pickle index. The displayed label is a human-readable hint; the actual
  underlying caption text isn't read from CUB unless you populate it.
- **Single-user demo.** `gr.Queue(max_size=1)` prevents concurrent T4 OOM —
  one grader at a time.
- **Colab sessions idle out after 90 minutes.** Use **Kaggle Notebooks**
  (9-hour sessions) for actual TA grading demos if Colab is too aggressive.
- **No quantitative metrics.** This is a qualitative side-by-side demo per
  course scope. CLIP-Score is the cheapest add-on if metrics are later
  required (no reference images needed).

---

## Troubleshooting

A list of every gotcha encountered during development, with the fix.

### `RuntimeError: Expected hasRecord("version") to be true` when loading the .pth

The Google Drive download for `1s5Yf3nFiXx0lltMFOiJWB6s1LP24RcwH` returns a
**zip archive** containing `birds_3stages/netG_210000.pth`, not a raw .pth.
`download_weights.py` auto-detects via the `PK\x03\x04` magic bytes and
extracts. If you downloaded manually, run:

```bash
unzip netG_210000.pth -d _tmp/ && mv _tmp/birds_3stages/netG_210000.pth weights/
```

### `gdown.exceptions.FileURLRetrievalError: Cannot retrieve the public link of the file`

The original Google Drive ID for `char-CNN-RNN-embeddings.pickle`
(`0B3y_msrWZaXLT1BZdVdycDY5TEE`) uses Google's pre-2014 ID format and is
permanently dead. There is no fix from the gdown side — use the Kaggle mirror
that `download_weights.py` falls back to:

```bash
# Set up Kaggle once (visit https://www.kaggle.com/settings → New API Token)
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
python download_weights.py
```

### `ModuleNotFoundError: No module named 'audioop'` on Python 3.13

Python 3.13 dropped `audioop` (PEP 594). pydub (a gradio dep) hasn't been
updated. Fix: `pip install audioop-lts` (already pinned in
`requirements.txt` for `python_version >= "3.13"`).

### `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'`

Indicates `gradio<5` was installed against `huggingface_hub>=1.0`, which
removed `HfFolder`. Upgrade gradio: `pip install -U "gradio>=5.0"`.

### `CondaToSNonInteractiveError: Terms of Service have not been accepted`

Conda 25+ requires the default channels' ToS to be accepted before
`conda env create` will run. Either accept (one-time):

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

…or use the venv quick-start path above — it doesn't touch conda channels.

### Gradio share URL dies after ~90 min on Colab

Free Colab sessions idle out. Move the demo to **Kaggle Notebooks**, which
have 9-hour sessions and the same GPU class.

### `OSError` / "out of memory" on T4

Set `--device cpu` first to confirm the pipelines themselves work, then on
T4: confirm only one inference is in-flight (the `gr.Queue(max_size=1)`
should already enforce this). If you've added other heavy models, swap SD
2.1 for `enable_model_cpu_offload()` in `sd21_pipeline.py`.

---

## Credits & citations

This project repackages and modernizes existing pretrained models. All credit
for the underlying research goes to the original authors.

- **StackGAN-v2 architecture & CUB pretrained weights**:
  [hanzhanggit/StackGAN-v2](https://github.com/hanzhanggit/StackGAN-v2)
  ([Zhang et al. 2018](https://arxiv.org/abs/1710.10916), PAMI).
- **char-CNN-RNN text encoder** (used to pre-compute CUB embeddings):
  [reedscot/icml2016](https://github.com/reedscot/icml2016)
  ([Reed et al. 2016](https://arxiv.org/abs/1605.05395), ICML).
- **CUB embeddings pickle mirror**:
  [text-to-image-cub-200-2011][kaggle-mirror] on Kaggle.
- **Stable Diffusion 2.1 base**:
  [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
  ([Rombach et al. 2022](https://arxiv.org/abs/2112.10752), CVPR).
- **CUB-200-2011 dataset**:
  [Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
  (Wah et al. 2011).

[kaggle-mirror]: https://www.kaggle.com/datasets/somthirthabhowmk2001/text-to-image-cub-200-2011

### BibTeX

```bibtex
@article{zhang2018stackgan,
  title   = {{StackGAN++}: Realistic Image Synthesis with Stacked Generative Adversarial Networks},
  author  = {Zhang, Han and Xu, Tao and Li, Hongsheng and Zhang, Shaoting and Wang, Xiaogang and Huang, Xiaolei and Metaxas, Dimitris N.},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2018}
}

@inproceedings{reed2016learning,
  title     = {Learning Deep Representations of Fine-Grained Visual Descriptions},
  author    = {Reed, Scott and Akata, Zeynep and Lee, Honglak and Schiele, Bernt},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}

@inproceedings{rombach2022high,
  title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```
