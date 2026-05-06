"""Download StackGAN-v2 pretrained CUB weights and embedding pickle.

Two artifacts:
  1. Generator weights (`netG_210000.pth`)  — Google Drive (StackGAN-v2 README)
  2. char-CNN-RNN test embeddings pickle    — Kaggle mirror (the original
     2017-era Google Drive ID is permanently dead)

The Kaggle source needs a one-time API token (`~/.kaggle/kaggle.json`).
On Colab use:
    from google.colab import files
    files.upload()  # upload kaggle.json
    !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

Run:
    python download_weights.py

Files written:
    weights/netG_210000.pth
    stackgan/embeddings/char-CNN-RNN-embeddings.pickle
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import gdown


# Source: hanzhanggit/StackGAN-v2 README — "StackGAN-v2 for bird"
STACKGAN_V2_BIRD_WEIGHTS_ID = "1s5Yf3nFiXx0lltMFOiJWB6s1LP24RcwH"

# Kaggle dataset hosting char-CNN-RNN embeddings (mirror of the dead Drive ID)
KAGGLE_DATASET = "somthirthabhowmk2001/text-to-image-cub-200-2011"

# What the embeddings pickle is called inside the dataset (we glob for it
# regardless of subdirectory — different mirrors layout it differently)
EMBEDDINGS_GLOB = "char-CNN-RNN-embeddings.pickle"
TEST_EMBEDDINGS_GLOB = "test/char-CNN-RNN-embeddings.pickle"


def _is_zip(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def _extract_pth_from_zip(archive: Path, target_name: str, out: Path) -> bool:
    """Extract <archive>'s file ending with target_name to out. Returns True on success."""
    with zipfile.ZipFile(archive) as zf:
        candidates = [n for n in zf.namelist() if n.endswith(target_name)]
        if not candidates:
            return False
        with zf.open(candidates[0]) as src, open(out, "wb") as dst:
            dst.write(src.read())
        print(f"  [unzip ] {candidates[0]} -> {out}")
        return True


def gdrive_download(file_id: str, out: Path, force: bool) -> None:
    """Download from Google Drive. If the result is a zip, transparently extract
    the file matching out.name (case-sensitive).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force and not _is_zip(out):
        print(f"  [skip] already present: {out}")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  [gdrive] {url} -> {out}")
    gdown.download(url, str(out), quiet=False)

    if _is_zip(out):
        archive = out.with_suffix(out.suffix + ".zip")
        out.rename(archive)
        if not _extract_pth_from_zip(archive, out.name, out):
            raise SystemExit(
                f"  Downloaded {archive} is a zip but contains no '{out.name}'; "
                "inspect manually."
            )
        archive.unlink()


def kaggle_download(dataset: str, out_dir: Path, force: bool) -> Path:
    """Download a Kaggle dataset into out_dir/<slug>/ and return the directory."""
    slug = dataset.split("/")[-1]
    target = out_dir / slug
    if target.exists() and any(target.iterdir()) and not force:
        print(f"  [skip] kaggle dataset already present: {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "kaggle package not installed. Run `pip install kaggle` first."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise SystemExit(
            "Kaggle authentication failed. Place a kaggle.json API token at "
            "~/.kaggle/kaggle.json (chmod 600 on Linux/macOS). "
            "Get one from https://www.kaggle.com/settings → 'Create New API Token'."
        ) from exc

    print(f"  [kaggle] {dataset} -> {target}")
    api.dataset_download_files(dataset, path=str(target), quiet=False, unzip=True)
    return target


def find_pickle(root: Path) -> Path | None:
    """Find char-CNN-RNN-embeddings.pickle inside root, preferring test/."""
    test_match = next(root.rglob(TEST_EMBEDDINGS_GLOB), None)
    if test_match is not None:
        return test_match
    return next(root.rglob(EMBEDDINGS_GLOB), None)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="re-download even if files exist")
    p.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent,
        help="project root (default: directory of this script)"
    )
    p.add_argument(
        "--skip-embeddings", action="store_true",
        help="skip Kaggle download (e.g. when a kaggle.json isn't set up yet)"
    )
    args = p.parse_args(argv)

    weights_path = args.root / "weights" / "netG_210000.pth"
    embeddings_dir = args.root / "stackgan" / "embeddings"
    embeddings_pickle = embeddings_dir / "char-CNN-RNN-embeddings.pickle"
    cache_dir = args.root / ".cache" / "kaggle"

    print("Downloading StackGAN-v2 CUB pretrained generator (Google Drive)...")
    gdrive_download(STACKGAN_V2_BIRD_WEIGHTS_ID, weights_path, args.force)

    if not args.skip_embeddings:
        print("Downloading CUB char-CNN-RNN test embeddings (Kaggle mirror)...")
        try:
            kaggle_dir = kaggle_download(KAGGLE_DATASET, cache_dir, args.force)
        except SystemExit as exc:
            print(f"  [warn] {exc}")
            print("  Continuing without embeddings — the demo will run SD 2.1 only.")
            kaggle_dir = None

        if kaggle_dir is not None:
            found = find_pickle(kaggle_dir)
            if found is None:
                print(
                    f"  [warn] no '{EMBEDDINGS_GLOB}' inside {kaggle_dir} — "
                    "verify the dataset's file layout."
                )
            else:
                embeddings_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(found, embeddings_pickle)
                print(f"  [copy ] {found} -> {embeddings_pickle}")

    print()
    print("Verification:")
    for path in (weights_path, embeddings_pickle):
        size = path.stat().st_size if path.exists() else 0
        status = "OK" if size > 0 else "MISSING"
        size_str = f"{size/1e6:.1f} MB" if size else "-"
        print(f"  [{status:>7}] {path}  ({size_str})")

    weights_ok = weights_path.exists()
    return 0 if weights_ok else 1


if __name__ == "__main__":
    sys.exit(main())
