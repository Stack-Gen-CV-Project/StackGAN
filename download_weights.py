"""Download StackGAN-v2 weights and CUB embeddings.

  python download_weights.py
  python download_weights.py --skip-embeddings   # if you don't have a Kaggle token
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import gdown


# StackGAN-v2 birds generator (Google Drive link is a zip containing
# birds_3stages/netG_210000.pth)
GDRIVE_WEIGHTS_ID = "1s5Yf3nFiXx0lltMFOiJWB6s1LP24RcwH"

# Kaggle dataset that mirrors the CUB char-CNN-RNN embeddings pickle.
# The original Google Drive ID for it is dead.
KAGGLE_DATASET = "somthirthabhowmk2001/text-to-image-cub-200-2011"


HERE = Path(__file__).parent
WEIGHTS_PATH = HERE / "weights" / "netG_210000.pth"
EMBEDDINGS_PICKLE = HERE / "stackgan" / "embeddings" / "char-CNN-RNN-embeddings.pickle"
KAGGLE_CACHE = HERE / ".cache" / "kaggle"


def _is_zip(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def download_stackgan_weights(force=False):
    if WEIGHTS_PATH.exists() and not force:
        print(f"  [skip] {WEIGHTS_PATH}")
        return
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={GDRIVE_WEIGHTS_ID}"
    print(f"  [get ] {url}")
    gdown.download(url, str(WEIGHTS_PATH), quiet=False)

    if _is_zip(WEIGHTS_PATH):
        archive = WEIGHTS_PATH.with_suffix(".pth.zip")
        WEIGHTS_PATH.rename(archive)
        with zipfile.ZipFile(archive) as zf:
            inner = next(n for n in zf.namelist() if n.endswith("netG_210000.pth"))
            with zf.open(inner) as src, open(WEIGHTS_PATH, "wb") as dst:
                dst.write(src.read())
        print(f"  [unzip] {inner} -> {WEIGHTS_PATH}")
        archive.unlink()


def download_embeddings(force=False):
    if EMBEDDINGS_PICKLE.exists() and not force:
        print(f"  [skip] {EMBEDDINGS_PICKLE}")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("  kaggle package not installed - run `pip install kaggle`.")
        return

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"  Kaggle auth failed: {e}")
        print("  Place a kaggle.json from https://www.kaggle.com/settings at "
              "~/.kaggle/kaggle.json (chmod 600 on Linux/Mac).")
        return

    KAGGLE_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"  [kaggle] {KAGGLE_DATASET}")
    api.dataset_download_files(KAGGLE_DATASET, path=str(KAGGLE_CACHE),
                                quiet=False, unzip=True)

    # Find the pickle (prefer test split) and copy it into place.
    test_match = next(KAGGLE_CACHE.rglob("test/char-CNN-RNN-embeddings.pickle"), None)
    found = test_match or next(KAGGLE_CACHE.rglob("char-CNN-RNN-embeddings.pickle"), None)
    if found is None:
        print("  No char-CNN-RNN-embeddings.pickle inside the Kaggle dataset.")
        return
    EMBEDDINGS_PICKLE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found, EMBEDDINGS_PICKLE)
    print(f"  [copy] {found} -> {EMBEDDINGS_PICKLE}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="don't try to download from Kaggle")
    args = p.parse_args()

    print("Downloading StackGAN-v2 generator...")
    download_stackgan_weights(force=args.force)

    if not args.skip_embeddings:
        print("Downloading CUB embeddings (Kaggle)...")
        download_embeddings(force=args.force)

    print()
    print("Done. Status:")
    for path in (WEIGHTS_PATH, EMBEDDINGS_PICKLE):
        size = path.stat().st_size if path.exists() else 0
        tag = "OK     " if size else "MISSING"
        print(f"  [{tag}] {path}  ({size/1e6:.1f} MB)" if size
              else f"  [{tag}] {path}")

    return 0 if WEIGHTS_PATH.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
