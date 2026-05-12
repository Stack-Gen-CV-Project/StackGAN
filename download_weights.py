import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import gdown


GDRIVE_WEIGHTS_ID = "1s5Yf3nFiXx0lltMFOiJWB6s1LP24RcwH"
KAGGLE_DATASET = "somthirthabhowmk2001/text-to-image-cub-200-2011"

HERE = Path(__file__).parent
WEIGHTS_PATH = HERE / "weights" / "netG_210000.pth"
EMBEDDINGS_PICKLE = HERE / "stackgan" / "embeddings" / "char-CNN-RNN-embeddings.pickle"
KAGGLE_CACHE = HERE / ".cache" / "kaggle"


def download_stackgan_weights():
    if WEIGHTS_PATH.exists():
        print(f"  [skip] {WEIGHTS_PATH}")
        return
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={GDRIVE_WEIGHTS_ID}"
    print(f"  [get ] {url}")
    gdown.download(url, str(WEIGHTS_PATH), quiet=False)

    with open(WEIGHTS_PATH, "rb") as f:
        is_zip = f.read(4) == b"PK\x03\x04"
    if is_zip:
        archive = WEIGHTS_PATH.with_suffix(".pth.zip")
        WEIGHTS_PATH.rename(archive)
        with zipfile.ZipFile(archive) as zf:
            inner = next(n for n in zf.namelist() if n.endswith("netG_210000.pth"))
            with zf.open(inner) as src, open(WEIGHTS_PATH, "wb") as dst:
                dst.write(src.read())
        archive.unlink()
        print(f"  [unzip] -> {WEIGHTS_PATH}")


def download_embeddings():
    if EMBEDDINGS_PICKLE.exists():
        print(f"  [skip] {EMBEDDINGS_PICKLE}")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("  kaggle package not installed - run: pip install kaggle")
        return

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"  Kaggle auth failed: {e}")
        print("  Place kaggle.json from https://www.kaggle.com/settings at")
        print("  ~/.kaggle/kaggle.json (chmod 600 on Linux/Mac).")
        print("  Skipping embeddings — the app will use synthetic fallback.")
        return

    KAGGLE_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"  [kaggle] {KAGGLE_DATASET}")
    api.dataset_download_files(KAGGLE_DATASET, path=str(KAGGLE_CACHE),
                                quiet=False, unzip=True)

    found = (
        next(KAGGLE_CACHE.rglob("test/char-CNN-RNN-embeddings.pickle"), None)
        or next(KAGGLE_CACHE.rglob("char-CNN-RNN-embeddings.pickle"), None)
    )
    if found is None:
        print("  No char-CNN-RNN-embeddings.pickle found in dataset.")
        return
    EMBEDDINGS_PICKLE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found, EMBEDDINGS_PICKLE)
    print(f"  [copy] -> {EMBEDDINGS_PICKLE}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-embeddings", action="store_true")
    args = p.parse_args()

    print("Downloading StackGAN-v2 generator...")
    download_stackgan_weights()

    if not args.skip_embeddings:
        print("Downloading CUB embeddings (Kaggle)...")
        download_embeddings()

    print()
    print("Done. Status:")
    for path in (WEIGHTS_PATH, EMBEDDINGS_PICKLE):
        if path.exists():
            print(f"  [OK]     {path}  ({path.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  [MISSING] {path}")

    return 0 if WEIGHTS_PATH.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
