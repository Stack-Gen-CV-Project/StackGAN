# Presentation content: StackGAN-v2 vs SDXL Turbo

This file is the slide-by-slide content for the Computer Vision course presentation.
Hand it to Claude and ask it to generate a PowerPoint deck from it.


---

## Slide 1 — Title

**Title:** StackGAN-v2 vs SDXL Turbo: A Side-by-Side Look at Six Years of Text-to-Image Progress

**Subtitle:** Computer Vision course project — Topic #7

**Footer:** Name / Course / Date


---

## Slide 2 — What is this project?

- A working demo that takes a caption and generates an image with two different models at the same time.
- Left side: StackGAN-v2 (2017), the model the course topic is about.
- Right side: SDXL Turbo (2023), a modern diffusion model.
- Both run from pre-trained weights. No training was done — the deliverable is the comparison and the analysis.

**One-line goal:** make the quality gap between a 2017 class-conditional GAN and a 2023 latent diffusion model visible in a single click.


---

## Slide 3 — Why this comparison?

- StackGAN-v2 is one of the early models that proved text-to-image was even possible at high resolution (256×256).
- It is constrained: trained on CUB-200-2011 (birds only), so it can only generate birds, and only from the 200 species in the dataset.
- SDXL Turbo is one of the fastest open-weight diffusion models today: 4 inference steps, runs on a free Colab T4 in around 2 seconds.
- Showing them side by side answers the question the course is really about: what changed between 2017 and 2023?


---

## Slide 4 — The two models at a glance

| | StackGAN-v2 (2017) | SDXL Turbo (2023) |
|---|---|---|
| Domain | Birds only (CUB-200) | Anything in LAION-5B |
| Input | Pre-computed text embedding | Free text prompt |
| Architecture | 3-stage class-conditional GAN | Latent diffusion + adversarial distillation |
| Output resolution | 256×256 | 512×512 |
| Inference steps | 1 forward pass | 4 denoising steps |
| Speed on T4 | ~0.5 s | ~2 s |
| Weight size | ~80 MB | ~6.9 GB |


---

## Slide 5 — StackGAN-v2 architecture

```
Text embedding (1024-d, from char-CNN-RNN)
          │
       CA_NET  →  mu, logvar  →  c_code (128-d)
          │                          │
       noise z (100-d)               │
          │                          │
      ┌───┴───┐                      │
      │ Stage 1│  ← c_code + z      │
      │ 4→64 px│                     │
      └───┬───┘                      │
      ┌───┴───┐                      │
      │ Stage 2│  ← c_code + h1     │
      │64→128px│                     │
      └───┬───┘                      │
      ┌───┴───┐                      │
      │ Stage 3│  ← c_code + h2     │
      │128→256 │                     │
      └───┬───┘
          │
     256×256 image
```

- Each stage: joint conv → residual blocks → upsample → conv3x3 + tanh.
- The trick: progressive growing inside one forward pass, with the text conditioning re-injected at every stage.


---

## Slide 6 — SDXL Turbo at a glance

- Built on top of SDXL: U-Net denoiser + VAE decoder + two text encoders (CLIP-L and OpenCLIP-G).
- Distilled with adversarial diffusion distillation (ADD) — collapses the usual 25–50 denoising steps down to 4.
- Free-form text: no dataset constraint, no dropdown.
- Released ungated on Hugging Face → no HF token needed for the demo.


---

## Slide 7 — System overview

```
              ┌──────────────────────────┐
              │      Gradio web UI       │
              │  (left pane / right pane)│
              └────────┬─────────┬───────┘
                       │         │
            ┌──────────┘         └──────────┐
            ▼                                ▼
  StackGAN inference module      SDXL Turbo wrapper
  - load netG_210000.pth          - diffusers AutoPipeline
  - lookup CUB embedding          - fp16, VAE tiling
  - 3-stage generator forward     - 4 inference steps
            │                                │
            ▼                                ▼
        256×256 bird                   512×512 image
```

- Single Python process. Both models live on the same GPU.
- The UI takes one caption from the user, runs both pipelines, returns both images.


---

## Slide 8 — Files in the repo

```
app.py                       # Gradio app, side-by-side layout
sd21_pipeline.py             # SDXL Turbo wrapper (fp16, VAE tiling)
download_weights.py          # Pulls StackGAN weights + CUB embeddings
stackgan/
    model.py                 # 3-stage generator (CA_NET + G_NET)
    inference.py             # Loads weights, generates images
    dropdown_captions.json   # 3 curated CUB entries
notebooks/colab_demo.ipynb   # One-click Colab launcher
requirements.txt
```

- Roughly 350 lines of Python total, plus the notebook.
- Tested locally on Windows + conda3 and remotely on Colab T4.


---

## Slide 9 — Challenges: roadmap

The rest of the slides walk through the obstacles we hit. Six main ones:

1. The model we picked first had no usable pre-trained weights.
2. The dataset embeddings link was permanently dead.
3. The weight download was secretly a zip file.
4. The original code didn't run on modern PyTorch.
5. The checkpoint was saved from a multi-GPU run.
6. The GPU we had access to couldn't fit the model we originally wanted.

Plus a few smaller ones along the way.


---

## Slide 10 — Challenge 1: StackGAN v1 had no CUB weights

- Course topic asked for StackGAN.
- First plan: use the original StackGAN v1 PyTorch port.
- Problem: the only public pre-trained weights for the v1 PyTorch port are for the COCO dataset. We needed CUB to match the original paper's bird demo.
- The author never released a CUB checkpoint for the v1 port — only for the Lua/Torch7 version, which is effectively unrunnable on modern hardware.

**Fix:** Pivoted to StackGAN-v2 (also called StackGAN++) from the same lab. v2 ships an official PyTorch CUB checkpoint and matches the course brief just as well. Day one work was scrapped.

**Lesson:** check the weight availability before committing to a paper, not after.


---

## Slide 11 — Challenge 2: The text encoder pickle was gone

- StackGAN needs a 1024-d text embedding as input. These come from a char-CNN-RNN encoder trained by Reed et al. (2016).
- The official `char-CNN-RNN-embeddings.pickle` was hosted on Google Drive with ID `0B3y_msrWZaXLT1BZdVdycDY5TEE`.
- That ID is in the old pre-2015 Drive format. Google deprecated those IDs years ago — the link now returns a 404 / "file not found" page.
- Re-training the encoder from scratch would have cost days of work (it's Lua/Torch7 code) and added no value to the demo.

**Fix:** found a community mirror of the same pickle on Kaggle: `somthirthabhowmk2001/text-to-image-cub-200-2011`. The downloader now pulls from there using the Kaggle API.

**Side effect:** Kaggle downloads require a `kaggle.json` API token, which created a UX problem (Challenge 6).


---

## Slide 12 — Challenge 3: The weights file was secretly a zip

- The StackGAN-v2 CUB weights live on Drive at ID `1s5Yf3nFiXx0lltMFOiJWB6s1LP24RcwH`.
- We named the local file `netG_210000.pth` and called `torch.load` on it. It crashed.
- Inspection: the first four bytes were `PK\x03\x04`. That is the ZIP magic number, not a PyTorch checkpoint. The Drive download was a zip archive with `birds_3stages/netG_210000.pth` inside it.

**Fix:** `download_weights.py` now sniffs the first four bytes after download. If it sees the ZIP magic, it unzips into place automatically and removes the archive. Transparent to the user.

```python
with open(WEIGHTS_PATH, "rb") as f:
    is_zip = f.read(4) == b"PK\x03\x04"
if is_zip:
    # extract netG_210000.pth out of the archive
    ...
```


---

## Slide 13 — Challenge 4: The reference code didn't run on PyTorch 2.x

- The official StackGAN-v2 code was written against PyTorch 0.3-ish.
- It used `torch.autograd.Variable`, `F.sigmoid` as a function on Variables, manual `.data` access, and `torch.randn(*size).cuda()`-style allocations.
- On PyTorch 2.x most of that is deprecated or broken: `Variable` is gone, `F.sigmoid` warns, and `randn_like` is now the idiomatic way to match device/dtype.

**Fix:** ported the generator file (`stackgan/model.py`) by hand:
- Removed all `Variable` wrapping.
- Swapped `F.sigmoid` for `torch.sigmoid` inside the GLU block.
- Replaced manual noise allocation with `torch.randn_like(std)` for the CA_NET reparametrize step.
- Used `weights_only=False` on `torch.load` because the checkpoint stores numpy arrays, which the new safe-loader rejects by default.

Result: same numerical behavior as the original, runs cleanly on torch 2.1.


---

## Slide 14 — Challenge 5: DataParallel prefix in the checkpoint

- `torch.load(...)` worked, but `netG.load_state_dict(...)` complained about every single key.
- Cause: the original training was done with `nn.DataParallel`, which wraps the module and prepends `module.` to every parameter name in the state dict.
- Our model isn't wrapped, so the keys didn't match.

**Fix:** strip the prefix on load:

```python
cleaned = {
    (k.replace("module.", "", 1) if k.startswith("module.") else k): v
    for k, v in sd.items()
}
netG.load_state_dict(cleaned, strict=False)
```

Tiny three-line fix, but failing silently here would have given us a randomly initialized generator and birds that look like static.


---

## Slide 15 — Challenge 6: T4 VRAM forced a model swap

- The Colab free tier gives a Tesla T4 with 16 GB of VRAM and a session that idles out after 90 minutes.
- First plan for the modern model: Flux.1-schnell. Beautiful samples, but the weights + activations don't reliably fit in 16 GB. We saw OOM crashes during the first generation.
- Second plan: Stable Diffusion 2.1-base. Fits comfortably, but at 25 inference steps it took ~8 seconds per image on T4. Too slow for an interactive demo.

**Fix:** switched to **SDXL Turbo**.
- Same VAE family as SDXL but distilled down to **4 steps** with classifier-free guidance disabled (`guidance_scale=0.0`).
- Fits on T4 with fp16 + VAE tiling enabled.
- ~6× faster than SD 2.1 in practice on the same hardware.
- Released ungated, so no Hugging Face token gymnastics for the grader.

```python
pipe.enable_vae_tiling()   # the line that made T4 viable
```


---

## Slide 16 — Challenge 7: Synthetic embedding fallback

- For graders or TAs who don't want to set up a Kaggle account just to run our demo, we needed a way to skip the embedding download.
- The naive option — disable StackGAN entirely without embeddings — would have killed half the demo.
- Question: what happens if you feed the generator random noise where the text embedding is supposed to go?
- We tried it. With a `torch.randn(1, 1024) * 0.5` "embedding", StackGAN still produces recognizable birds — terns, songbirds, seabirds.
- This works because CUB has only 200 species, the generator was trained heavily on bird textures, and the bird-domain prior dominates the output.

**Fix:** the inference module detects the missing pickle and falls back to a deterministic random embedding seeded from the image index. The user sees real-looking birds either way; the dropdown labels just no longer correspond to specific captions.

This was an unexpected finding and arguably one of the more interesting results in the project.


---

## Slide 17 — Challenge 8: Kaggle auth in Colab

- Real CUB embeddings need a `kaggle.json` token.
- Locally that goes at `~/.kaggle/kaggle.json`. On Colab there is no persistent home directory between sessions.

**Fix:** the Colab notebook has a dedicated cell that prompts the user to upload `kaggle.json` through `google.colab.files.upload`, writes it to `~/.kaggle/`, chmods it to `600`, then re-runs the downloader. If the user clicks Cancel, the notebook prints a message and falls through to the synthetic-embedding path. No hard failure either way.


---

## Slide 18 — Smaller obstacles worth mentioning

- **HF gated models.** `stabilityai/stable-diffusion-2-1` is gated (requires a token + accepting the license). We deliberately picked `stable-diffusion-2-1-base` (and later SDXL Turbo) because they are ungated. Saved every grader from making an HF account.
- **Sample images for the README.** The repo needs hero images, but committing 6 GB of weights to GitHub is not an option. We committed two small `samples/stackgan_synthetic_bird_*.png` outputs instead.
- **CRLF vs LF line endings.** Working on Windows + git's `autocrlf` produced noisy diffs the first time we committed. Solved by leaving autocrlf alone and ignoring the warnings.


---

## Slide 19 — Demo

Live demo (or screenshot of the Gradio UI):

- Left pane: dropdown of 3 CUB bird embeddings, "Generate" button → 256×256 StackGAN image.
- Right pane: free-text prompt box, same Generate button → 512×512 SDXL Turbo image.
- Seed input on the side for reproducibility.
- A few example prompts pre-loaded for the grader: "a corgi running through a sunflower field, dslr photo", "an astronaut riding a horse on Mars, digital art", etc.

Suggested live prompt to type in the demo:
> "a small yellow songbird singing on a green leaf, photo"

Both models try to draw it. StackGAN produces a bird (because it can only do birds). SDXL Turbo produces an actual photo-style image of the described scene. That single side-by-side is the whole talk in one image.


---

## Slide 20 — Results and observations

- **StackGAN-v2 is impressive for 2017** — the bird textures are convincing at a glance, especially feather detail. Faces and beaks are where it falls apart.
- **SDXL Turbo handles everything**, but it doesn't always beat StackGAN on raw bird-feather realism if you constrain the prompt. The constrained model has a real advantage in its narrow domain.
- **Speed:** StackGAN-v2 is faster per image (single forward pass vs 4 denoising steps), but the loading time for SDXL Turbo dominates on cold start.
- **The synthetic-embedding finding** (Slide 16) is, in some ways, the most interesting result: StackGAN's class-conditional prior is so strong that the "text" half of "text-to-image" is partially redundant for this narrow dataset.


---

## Slide 21 — Lessons learned

- **Check weight availability before picking a paper.** Most of the StackGAN v1 → v2 pain could have been avoided with a 10-minute check.
- **Old research links rot.** Plan for the official asset to be gone and a community mirror to be your real source.
- **Make every step optional or fall-back-able.** The synthetic embedding path, the `--skip-embeddings` flag, the cancel-the-upload path in Colab — these are what make a course demo actually survive contact with a grader who has different credentials than you.
- **Pick the smallest model that demonstrates your point.** SDXL Turbo, not Flux, not full SDXL. The demo is about contrast, not about chasing the SOTA.
- **Hardware constraints are design constraints.** Almost every model choice in this project was made against the 16 GB T4 budget.


---

## Slide 22 — Limitations and future work

- StackGAN-v2 can still only do birds. We are not solving that; it is the point of the comparison.
- We did not compute quantitative metrics (FID / IS) because the course brief asked for a demo, not a benchmark. Adding FID on a CUB test split would be a natural next step.
- The free-text Gradio side and the CUB dropdown are coupled to one seed input, which is a UX compromise. A nicer version would have separate seeds per pane.
- No batching: the two models run sequentially. On a bigger GPU they could run in parallel.


---

## Slide 23 — Credits and references

- StackGAN-v2 — Zhang et al., 2018. Repo: github.com/hanzhanggit/StackGAN-v2
- char-CNN-RNN encoder — Reed et al., ICML 2016. Repo: github.com/reedscot/icml2016
- SDXL Turbo — Sauer et al., 2023. Model card: huggingface.co/stabilityai/sdxl-turbo
- CUB-200-2011 — Wah et al., Caltech. vision.caltech.edu/visipedia/CUB-200-2011.html
- diffusers, transformers, accelerate — Hugging Face
- Gradio — gradio.app


---

## Slide 24 — Thank you / Q&A

- Repo: github.com/Stack-Gen-CV-Project/StackGAN
- Demo link (Colab): the badge in the README opens the notebook directly on T4.
- Questions?


---

## Instructions for Claude (when generating the PowerPoint)

- Use a clean, minimal slide template. Default theme, dark blue or neutral grey, no clip art.
- Title font: 32–36 pt. Body font: 18–22 pt. Don't shrink the body text smaller than 16 pt.
- For the architecture diagram slides (5, 7, 8), render the ASCII block as a Mermaid diagram or a clean SmartArt flow — do **not** keep the monospace ASCII in the final deck.
- For the comparison table (slide 4) and the "Before / After" tradeoffs, use real PowerPoint tables, not images.
- The challenge slides (10–18) should share a consistent layout: heading at top, one short paragraph for the problem, a "Fix:" block underneath. Keep the code snippets short and in a monospace font.
- Do not add icons or emojis on the slides.
- Don't put speaker notes on the slide face itself. If the tool supports it, put the explanatory paragraphs in the speaker-notes pane and keep the slide body to 4–6 short bullets max.
