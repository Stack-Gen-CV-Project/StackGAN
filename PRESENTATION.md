# Presentation content: StackGAN-v2 vs SDXL Turbo

16 slides. Stays simple throughout. The last two slides before Q&A are the technical deep dive — they're there to show we actually understand how the models are built and trained, not just call them as black boxes.

---

## Slide 1 — Title

**Title:** StackGAN-v2 vs SDXL Turbo

**Subtitle:** Comparing two text-to-image AI models, six years apart

**Footer:** Team names / Course name / Date

---

## Slide 2 — What we built

- A small website.
- You type a sentence.
- It shows you two images for that sentence, from two different AI models.
- That's it.

---

## Slide 3 — The two models

- **Left:** StackGAN-v2 — released in 2017.
- **Right:** SDXL Turbo — released in 2023.
- Both are already trained. We didn't train anything ourselves.

---

## Slide 4 — Why this comparison

- Six years between the two models.
- Six years is a long time in AI.
- Putting them next to each other is the easiest way to see how much has changed.

---

## Slide 5 — StackGAN-v2 (the old one)

- From 2017.
- Only knows birds — was trained on 200 species.
- Output: 256 × 256 pixels.

---

## Slide 6 — SDXL Turbo (the new one)

- From 2023.
- Can draw anything you describe.
- About 2 seconds per image on a small GPU.
- Output: 512 × 512 pixels.

---

## Slide 7 — The app (live demo)

- Left pane: pick a bird from a dropdown — StackGAN draws it.
- Right pane: type any sentence — SDXL Turbo draws it.
- One button. Two images. Side by side.

(Use a screenshot of the live app.)

---

## Slide 8 — How we built it

- Python.
- Gradio for the web interface.
- Hugging Face Diffusers for SDXL Turbo.
- About 350 lines of code.

---

## Slide 9 — Challenge 1: dead links

- The original StackGAN files lived on Google Drive.
- The links are very old — they don't work anymore.
- We found a community copy on Kaggle and downloaded from there.

---

## Slide 10 — Challenge 2: small GPU

- Free Colab gives a Tesla T4 — 16 GB of memory.
- The newest, biggest models don't fit.
- We chose SDXL Turbo because it's small enough AND fast.

---

## Slide 11 — Challenge 3: old code

- The original StackGAN code is from 2017.
- It uses commands that don't exist in modern PyTorch anymore.
- We updated a few lines to make it run.

---

## Slide 12 — A surprising finding

- StackGAN was trained intensely on birds.
- We gave it pure random numbers instead of a real description.
- It still drew birds.
- The model has "memorized" what birds look like.

---

## Slide 13 — What we learned

- Old AI research has many broken links — plan for that.
- Hardware limits matter — pick a model that fits.
- Side-by-side comparison is the best way to feel the difference.

---

## Slide 14 — Deep dive: how StackGAN-v2 is built and trained

**Architecture:**
- Inputs: a 1024-dim text embedding + a 100-dim random noise vector.
- A small layer called **CA_NET** compresses the text embedding into a 128-dim conditioning code (variational layer: outputs a mean + log-variance and samples).
- **Three generator stages**: each one upsamples — 4 → 64 → 128 → 256 pixels. The text conditioning is re-injected at every stage.
- Uses **Gated Linear Units (GLU)** instead of ReLU — half the channels gate the other half.

**Training:**
- Trained as a **GAN**: Generator vs. Discriminator.
- Dataset: **CUB-200-2011** — about 12,000 photos of 200 bird species, with 10 captions per image.
- **210,000 training iterations** — that's why the checkpoint is named `netG_210000.pth`.
- Loss = adversarial + KL divergence (from CA_NET) + color consistency between stages.
- Adam optimizer, learning rate 2e-4, batch size 24, two GPUs, about a week.

---

## Slide 15 — Deep dive: how SDXL Turbo is built and trained

**Architecture:**
- It's a **latent diffusion model** — all the work happens in a small compressed latent space, not on pixels.
- **VAE** encodes a 512×512 RGB image into a 64×64×4 latent map (and decodes back at the end).
- **U-Net denoiser** (~2.6 billion parameters) — predicts noise from a noisy latent + text + timestep.
- **Two text encoders** combined: CLIP-L (~123M params) + OpenCLIP-G (~694M params), outputs concatenated.
- Generation: random latent → U-Net predicts noise → subtract → repeat **4 times** → VAE decode.

**Training:**
- Distilled from regular SDXL using **Adversarial Diffusion Distillation (ADD)**.
- **Teacher–student** setup: slow 25-step SDXL is the teacher, fast 4-step SDXL is the student.
- Two losses: **score distillation** (match the teacher's gradient) + **adversarial** (a discriminator forces realism).
- Trained on Stability AI's internal multi-billion-image dataset, fp16 precision.
- Result: SDXL-quality output in 4 steps instead of 25.

---

## Slide 16 — Thank you + Q&A

- Code: github.com/Stack-Gen-CV-Project/StackGAN
- Thanks for listening.
- Any questions?
