"""Generate a project summary PDF with troubleshooting guide."""

from fpdf import FPDF


class ProjectPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, "StackGAN-v2 vs SDXL Turbo  |  Computer Vision Project", align="C")
        self.ln(10)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 80, 160)
        self.set_x(10)
        self.cell(190, 9, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 80, 160)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.set_x(10)
        self.cell(190, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.set_x(10)
        self.multi_cell(190, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.set_x(10)
        self.multi_cell(190, 5.5, "  - " + text)

    def code_block(self, text):
        self.set_fill_color(240, 240, 245)
        self.set_font("Courier", "", 9)
        self.set_text_color(30, 30, 30)
        self.set_x(10)
        self.multi_cell(190, 5, text, fill=True)
        self.ln(3)

    def problem_block(self, problem, cause, solution):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(180, 40, 40)
        self.set_x(10)
        self.cell(190, 6, f"PROBLEM: {problem}", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(80, 80, 80)
        self.set_x(10)
        self.multi_cell(190, 5, f"Cause: {cause}")
        self.set_text_color(30, 120, 30)
        self.set_font("Helvetica", "B", 9)
        self.set_x(10)
        self.multi_cell(190, 5, f"Solution: {solution}")
        self.ln(3)


pdf = ProjectPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# ===================== TITLE PAGE =====================
pdf.ln(20)
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(30, 80, 160)
pdf.cell(0, 15, "StackGAN-v2 vs SDXL Turbo", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(4)
pdf.set_font("Helvetica", "", 14)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, "Text-to-Image Generation Comparison", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "Computer Vision Course Project", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 7, "Author: Ahmed Yusri", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "University: MUC - Computer Engineering", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "GitHub: github.com/Stack-Gen-CV-Project/StackGAN", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(15)

pdf.set_draw_color(30, 80, 160)
pdf.line(60, pdf.get_y(), 150, pdf.get_y())
pdf.ln(10)

pdf.set_font("Helvetica", "I", 11)
pdf.set_text_color(100, 100, 100)
pdf.multi_cell(0, 6, (
    "This document provides a complete summary of the StackGAN-v2 vs SDXL Turbo "
    "project, including architecture details, component descriptions, and a "
    "comprehensive troubleshooting guide for common issues."
), align="C")

# ===================== PAGE 2: PROJECT OVERVIEW =====================
pdf.add_page()
pdf.section_title("1. Project Overview")

pdf.body_text(
    "This project compares two text-to-image generation approaches side by side:\n\n"
    "  - StackGAN-v2 (2017): A GAN-based model trained on the CUB-200-2011 bird "
    "dataset. It generates 256x256 bird images from pre-computed text embeddings. "
    "It only knows how to draw birds.\n\n"
    "  - SDXL Turbo (2023): A diffusion-based model trained on a massive dataset. "
    "It generates 512x512 images from any free-text prompt in just 4 inference "
    "steps. It can draw anything.\n\n"
    "Both models are pre-trained. There is no training code in this repo. The "
    "demo runs via a Gradio web interface on http://localhost:7860 or a public "
    "Gradio share URL on Google Colab."
)

pdf.section_title("2. Technology Stack")

pdf.bullet("Python 3.10+")
pdf.bullet("PyTorch 2.x (deep learning framework)")
pdf.bullet("HuggingFace Diffusers (SDXL Turbo loading)")
pdf.bullet("Gradio 5.x (web UI framework)")
pdf.bullet("Google Colab T4 GPU (primary runtime)")
pdf.bullet("gdown (Google Drive downloads)")
pdf.bullet("Kaggle API (CUB embeddings, optional)")
pdf.ln(3)

pdf.section_title("3. Project Structure")
pdf.code_block(
    "StackGAN/\n"
    "  app.py                       Main Gradio web app\n"
    "  sd21_pipeline.py             SDXL Turbo wrapper\n"
    "  download_weights.py          Weight downloader\n"
    "  requirements.txt             Dependencies\n"
    "  stackgan/\n"
    "    model.py                   Generator architecture\n"
    "    inference.py               Load + generate\n"
    "    dropdown_captions.json     3 CUB bird captions\n"
    "  notebooks/\n"
    "    colab_demo.ipynb           Colab launcher\n"
    "  weights/                     (downloaded) netG_210000.pth\n"
    "  samples/                     Example output images"
)

# ===================== PAGE 3: STACKGAN ARCHITECTURE =====================
pdf.add_page()
pdf.section_title("4. StackGAN-v2 Architecture")

pdf.body_text(
    "StackGAN-v2 is a 3-stage progressively growing generator. Each stage "
    "doubles the image resolution while injecting the text conditioning. "
    "The architecture was ported from the original StackGAN-v2 repo and "
    "updated for PyTorch 2.x."
)

pdf.sub_title("4.1 Global Configuration")
pdf.code_block(
    "Z_DIM = 100           Noise vector dimension\n"
    "EMBEDDING_DIM = 128   Conditioning code size\n"
    "GF_DIM = 64           Base channel width\n"
    "R_NUM = 2             Residual blocks per stage\n"
    "TEXT_DIM = 1024       char-CNN-RNN embedding size"
)

pdf.sub_title("4.2 CA_NET (Conditional Augmenting Network)")
pdf.body_text(
    "Converts a 1024-dim text embedding into a 128-dim conditioning code "
    "using a VAE-style encoder. It outputs mu and logvar, then samples via "
    "the reparameterization trick: c = mu + std * epsilon, where epsilon is "
    "random noise. This allows generating different images from the same "
    "caption and enables smooth interpolation in the latent space."
)

pdf.sub_title("4.3 Stage 1: INIT_STAGE_G (4x4 -> 64x64)")
pdf.body_text(
    "Concatenates the noise vector z (100-dim) with the conditioning code "
    "(128-dim), maps to a 4x4 feature map via a fully connected layer, "
    "then applies 4 upsampling blocks (nearest neighbor + conv + BN + GLU) "
    "to reach 64x64 resolution. Channel count decreases: 1024 -> 512 -> "
    "256 -> 128 -> 64."
)

pdf.sub_title("4.4 Stage 2 & 3: NEXT_STAGE_G (64->128->256)")
pdf.body_text(
    "Each refinement stage takes the previous features, tiles the "
    "conditioning code to match spatial dimensions, concatenates them, "
    "applies a joint convolution, 2 residual blocks for refinement, "
    "and one upsampling block. Stage 2 produces 128x128, Stage 3 "
    "produces 256x256."
)

pdf.sub_title("4.5 Key Components")
pdf.bullet("GLU (Gated Linear Unit): Splits channels, gates with sigmoid. More expressive than ReLU.")
pdf.bullet("ResBlock: Residual connection with 2 convolutions. Prevents vanishing gradients.")
pdf.bullet("upBlock: Nearest-neighbor upsample + conv + BN + GLU. Doubles spatial resolution.")
pdf.bullet("GET_IMAGE_G: Final conv3x3 + Tanh to produce [-1,1] RGB image.")
pdf.ln(3)

# ===================== PAGE 4: SDXL TURBO =====================
pdf.add_page()
pdf.section_title("5. SDXL Turbo Pipeline")

pdf.body_text(
    "SDXL Turbo (stabilityai/sdxl-turbo) is a text-to-image diffusion model "
    "released by Stability AI in November 2023. It is based on the SDXL "
    "architecture but distilled via adversarial training to produce good "
    "results in just 1-4 denoising steps."
)

pdf.sub_title("5.1 Why SDXL Turbo over SD v1.5?")
pdf.bullet("6x faster: 4 inference steps vs 25 steps")
pdf.bullet("Better quality: SDXL architecture with larger UNet")
pdf.bullet("No CFG needed: guidance_scale=0.0 (trained without it)")
pdf.bullet("Ungated: No HuggingFace token required")
pdf.bullet("Fits T4: ~6.5GB in fp16, T4 has 16GB VRAM")
pdf.ln(3)

pdf.sub_title("5.2 Pipeline Details")
pdf.body_text(
    "Uses AutoPipelineForText2Image from HuggingFace Diffusers, which "
    "auto-detects the correct pipeline class. Loaded lazily on first "
    "generate() call to avoid slowing app startup. Uses fp16 on GPU "
    "for memory efficiency and enable_vae_tiling() to reduce peak "
    "VRAM usage."
)

pdf.sub_title("5.3 Inference Parameters")
pdf.code_block(
    "num_inference_steps = 4     (Turbo-specific, was 25 for SD 1.5)\n"
    "guidance_scale = 0.0        (no CFG, Turbo was trained without it)\n"
    "height = 512, width = 512   (native resolution)\n"
    "dtype = float16             (on GPU) / float32 (on CPU)\n"
    "variant = 'fp16'            (download fp16 weights directly)"
)

# ===================== PAGE 5: INFERENCE & APP =====================
pdf.add_page()
pdf.section_title("6. Inference Pipeline (stackgan/inference.py)")

pdf.body_text(
    "The StackGANInference class handles loading weights, loading embeddings, "
    "and generating images. Key design decisions:"
)

pdf.sub_title("6.1 Weight Loading")
pdf.bullet("Loads netG_210000.pth (trained for 210,000 iterations)")
pdf.bullet("Strips 'module.' prefix from DataParallel checkpoints")
pdf.bullet("Uses strict=False for partial weight loading")
pdf.bullet("Sets eval mode (disables dropout/batchnorm training behavior)")
pdf.ln(2)

pdf.sub_title("6.2 Embedding System")
pdf.body_text(
    "Two modes: real embeddings from CUB dataset (a pickle file with shape "
    "[num_images, captions_per_image, 1024]) or synthetic fallback "
    "(deterministic random vectors). The synthetic fallback works because "
    "StackGAN has a strong bird prior -- almost any reasonable 1024-dim "
    "vector produces a bird-shaped output."
)

pdf.sub_title("6.3 Image Generation")
pdf.body_text(
    "1. Get text embedding (real or synthetic)\n"
    "2. Sample noise z ~ N(0,1) with given seed\n"
    "3. Run generator: netG(z, emb) -> 3 images at 64/128/256\n"
    "4. Take the last image (256x256, values in [-1,1])\n"
    "5. Convert: ((img + 1) / 2 * 255).clamp(0, 255).byte()\n"
    "6. Return as PIL Image"
)

pdf.section_title("7. Gradio Web App (app.py)")

pdf.body_text(
    "The main entry point creates a side-by-side comparison UI. The left "
    "panel has controls (dropdown for StackGAN, textbox for SDXL Turbo, "
    "seed input, generate button, examples). The right panel shows both "
    "generated images with timing info. The 'Generate' button triggers "
    "both models simultaneously."
)

pdf.sub_title("7.1 Command-Line Arguments")
pdf.code_block(
    "--device        cuda or cpu (default: cuda)\n"
    "--sd-model-id   HuggingFace model ID (default: stabilityai/sdxl-turbo)\n"
    "--share         true for public Gradio URL (for Colab)"
)

# ===================== PAGE 6: TROUBLESHOOTING =====================
pdf.add_page()
pdf.section_title("8. Troubleshooting - Common Problems & Solutions")

pdf.problem_block(
    "CUDA Out of Memory on T4",
    "SDXL Turbo requires ~6.5GB VRAM. With StackGAN also loaded, "
    "total usage may exceed T4's 16GB during peak.",
    "1) Restart runtime and run only SDXL (skip StackGAN).\n"
    "2) Use --device cpu for StackGAN side (slower but no VRAM).\n"
    "3) Close other Colab notebooks using the same GPU.\n"
    "4) The code already uses fp16 and VAE tiling to minimize usage."
)

pdf.problem_block(
    "StackGAN weights not found at weights/netG_210000.pth",
    "The download_weights.py script was not run, or the Google Drive "
    "download failed (rate limits, network issues).",
    "1) Run: python download_weights.py --skip-embeddings\n"
    "2) If gdown fails, download manually from Google Drive.\n"
    "3) Place the .pth file in weights/netG_210000.pth"
)

pdf.problem_block(
    "Kaggle authentication failed",
    "The kaggle.json token is missing or in the wrong location.",
    "1) Go to https://www.kaggle.com/settings\n"
    "2) Click 'Create New API Token' (downloads kaggle.json)\n"
    "3) Place it at: ~/.kaggle/kaggle.json (or %%USERPROFILE%%\\.kaggle on Windows)\n"
    "4) Re-run: python download_weights.py\n"
    "5) Or skip: python download_weights.py --skip-embeddings"
)

pdf.problem_block(
    "SDXL Turbo download is very slow or fails",
    "The model is ~6.5GB and HuggingFace servers may be slow.",
    "1) On Colab: the model is cached after first download.\n"
    "2) Locally: download once, it stays in HF cache.\n"
    "3) If gated model error: set HF_TOKEN env var.\n"
    "4) Use a smaller model: --sd-model-id stable-diffusion-v1-5/stable-diffusion-v1-5"
)

pdf.problem_block(
    "ModuleNotFoundError: No module named 'xxx'",
    "A required package is not installed.",
    "1) Run: pip install -r requirements.txt\n"
    "2) On Colab: restart runtime after installing.\n"
    "3) If 'kaggle' is missing: pip install kaggle\n"
    "4) If 'gdown' is missing: pip install gdown"
)

# ===================== PAGE 7: MORE PROBLEMS =====================
pdf.add_page()

pdf.problem_block(
    "Colab disconnects during generation",
    "Colab free tier has idle timeouts (~90 min) and may disconnect "
    "if the browser tab is inactive.",
    "1) Keep the Colab tab active and visible.\n"
    "2) Use Colab Pro for longer runtimes.\n"
    "3) The Gradio share URL survives disconnection for a while.\n"
    "4) Models are re-downloaded on reconnect (cached in HF cache)."
)

pdf.problem_block(
    "Generation is very slow on CPU",
    "SDXL Turbo on CPU takes 30-60 seconds per image vs 2 seconds on T4.",
    "1) Always use --device cuda on Colab (T4 GPU).\n"
    "2) Locally: ensure CUDA is available (torch.cuda.is_available()).\n"
    "3) StackGAN is fast even on CPU (~1-2 seconds).\n"
    "4) For CPU-only: use --sd-model-id stable-diffusion-v1-5/stable-diffusion-v1-5"
)

pdf.problem_block(
    "PIL/numpy deprecation warnings",
    "Newer versions of Pillow and numpy may show warnings about "
    "deprecated APIs.",
    "These are warnings, not errors. The code works fine.\n"
    "To suppress: pip install 'Pillow>=10.0' 'numpy>=1.24'"
)

pdf.problem_block(
    "Git push fails: 'unable to access github.com'",
    "No network connection or GitHub credentials not configured.",
    "1) Check internet connection.\n"
    "2) Set up credentials:\n"
    "   git config --global credential.helper store\n"
    "   git remote set-url origin https://USER:TOKEN@github.com/...\n"
    "   git push origin main\n"
    "3) Create token at: https://github.com/settings/tokens"
)

pdf.problem_block(
    "'module.' prefix error when loading weights",
    "The checkpoint was saved with nn.DataParallel which adds 'module.' "
    "prefix to all parameter names.",
    "Already handled in the code. inference.py strips the prefix:\n"
    "  cleaned = {k.replace('module.', '', 1): v for k, v in sd.items()}\n"
    "If you see this error, update inference.py."
)

pdf.problem_block(
    "Gradio share URL not working",
    "The --share flag was not set, or Gradio tunneling failed.",
    "1) Launch with: python app.py --share true\n"
    "2) On Colab, this is already set in the notebook.\n"
    "3) If tunneling fails, use the local URL instead.\n"
    "4) Check firewall/proxy settings."
)

# ===================== PAGE 8: MORE PROBLEMS =====================
pdf.add_page()

pdf.problem_block(
    "StackGAN produces noise or garbage images",
    "The weights file may be corrupted, or the embeddings don't match.",
    "1) Re-download weights: python download_weights.py\n"
    "2) Check file size: should be ~80 MB.\n"
    "3) Try different seed values (change the seed number).\n"
    "4) If using synthetic embeddings, try all 3 dropdown options."
)

pdf.problem_block(
    "SDXL Turbo: 'Repository not found' or 401 error",
    "The model is gated and requires a HuggingFace token.",
    "1) SDXL Turbo (default) is ungated -- no token needed.\n"
    "2) If using --sd-model-id with a gated model:\n"
    "   a) Accept the license on the model's HF page.\n"
    "   b) Create token at https://huggingface.co/settings/tokens\n"
    "   c) Set: export HF_TOKEN=hf_xxxxx\n"
    "   d) Re-run: python app.py --sd-model-id <model-id>"
)

pdf.problem_block(
    "Python version compatibility issues",
    "Some packages may not support older Python versions.",
    "1) Requires Python 3.10+\n"
    "2) Colab has Python 3.10 by default (compatible).\n"
    "3) Locally: check with python --version\n"
    "4) audioop-lts package only needed for Python 3.13+"
)

pdf.problem_block(
    "Colab notebook cells fail after restart",
    "Colab restarts the runtime when installing packages. "
    "You must re-run cells from the beginning.",
    "1) After restart: Runtime -> Run all (or run cells in order).\n"
    "2) The pip install cell must run before any import cells.\n"
    "3) Weights are in the cloned repo, not lost on restart.\n"
    "4) SDXL Turbo is cached in HF cache (survives restarts)."
)

pdf.problem_block(
    "Low quality / blurry StackGAN output",
    "StackGAN-v2 was trained on CUB-200 (small dataset, 200 bird species). "
    "256x256 was state-of-the-art in 2017.",
    "This is expected behavior, not a bug. StackGAN-v2 is a 2017 model:\n"
    "- Limited to 256x256 resolution\n"
    "- Only trained on birds (CUB-200)\n"
    "- Uses char-CNN-RNN text encoder (simpler than modern CLIP)\n"
    "For better quality, use the SDXL Turbo side of the demo."
)

# ===================== PAGE 9: RUNNING THE PROJECT =====================
pdf.add_page()
pdf.section_title("9. How to Run the Project")

pdf.sub_title("9.1 On Google Colab (Recommended)")
pdf.code_block(
    "1. Open the Colab notebook link from the README\n"
    "2. Set Runtime -> Change runtime type -> T4 GPU\n"
    "3. Run all cells (4 cells total)\n"
    "4. Click the public Gradio URL that appears\n"
    "5. Select a StackGAN caption and type an SDXL prompt\n"
    "6. Click Generate -> see both images side by side"
)

pdf.sub_title("9.2 On Local Machine")
pdf.code_block(
    "1. Clone: git clone https://github.com/Stack-Gen-CV-Project/StackGAN.git\n"
    "2. Install: pip install -r requirements.txt\n"
    "3. Download: python download_weights.py --skip-embeddings\n"
    "4. Run: python app.py\n"
    "5. Open: http://localhost:7860"
)

pdf.sub_title("9.3 Key Commands")
pdf.code_block(
    "python download_weights.py              Download all weights\n"
    "python download_weights.py --skip-embeddings  Skip Kaggle part\n"
    "python app.py                           Launch (default: CUDA)\n"
    "python app.py --device cpu              Launch on CPU\n"
    "python app.py --share true              Public Gradio URL\n"
    "python app.py --sd-model-id <id>        Use different SD model"
)

pdf.section_title("10. References")

pdf.bullet("StackGAN-v2: Zhang et al. 2018 - github.com/hanzhanggit/StackGAN-v2")
pdf.bullet("char-CNN-RNN: Reed et al. 2016 - github.com/reedscot/icml2016")
pdf.bullet("SDXL Turbo: Sauer et al. 2023 - huggingface.co/stabilityai/sdxl-turbo")
pdf.bullet("CUB-200-2011: vision.caltech.edu/visipedia/CUB-200-2011.html")
pdf.bullet("Diffusers: huggingface.co/docs/diffusers")
pdf.bullet("Gradio: gradio.app")


# Save
output_path = "D:/MUC/Computer Vision/Project/StackGAN/StackGAN_Project_Summary.pdf"
pdf.output(output_path)
print(f"PDF saved to: {output_path}")
