"""
Zenodo Download Script for Forest Inspection Dataset
Downloads individual sequences with resume support and progress tracking.

Usage:
    python scripts/download_zenodo.py                    # Download all sequences
    python scripts/download_zenodo.py --seq 1 3 5        # Download specific sequences
    python scripts/download_zenodo.py --seq 1 --dry-run  # Check URLs only
    python scripts/download_zenodo.py --output data/forest_sunny
"""

import argparse
import os
import sys
import hashlib
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD_ID = "15511426"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"

SEQUENCES = {
    "readme": "README.md",
    **{f"seq{i}": f"seq{i}.zip" for i in range(1, 10)},
}


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download a file with resume support and progress bar."""
    headers = {}
    mode = "wb"
    initial_size = 0

    # Resume support
    if dest.exists():
        initial_size = dest.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)

        # If server returns 416 (Range Not Satisfiable), file is complete
        if response.status_code == 416:
            print(f"  ✓ Already downloaded: {dest.name}")
            return True

        if response.status_code not in (200, 206):
            print(f"  ✗ HTTP {response.status_code} for {dest.name}")
            return False

        total_size = int(response.headers.get("content-length", 0)) + initial_size

        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, mode) as f:
            with tqdm(
                total=total_size,
                initial=initial_size,
                unit="B",
                unit_scale=True,
                desc=f"  {dest.name}",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error downloading {dest.name}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract a zip file and remove the archive."""
    try:
        print(f"  Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        print(f"  ✓ Extracted to {extract_to}")
        return True
    except zipfile.BadZipFile:
        print(f"  ✗ Corrupted zip: {zip_path.name}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Forest Inspection Dataset from Zenodo"
    )
    parser.add_argument(
        "--seq",
        nargs="+",
        type=int,
        choices=range(1, 10),
        help="Specific sequences to download (1-9). Default: all",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/forest_sunny",
        help="Output directory (default: data/forest_sunny)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check URLs, don't download",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract zip files after downloading",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep zip files after extraction",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    # Determine which sequences to download
    if args.seq:
        targets = {f"seq{i}": f"seq{i}.zip" for i in args.seq}
    else:
        targets = SEQUENCES.copy()

    print(f"╔══════════════════════════════════════════╗")
    print(f"║  Forest Inspection Dataset Downloader    ║")
    print(f"║  Zenodo Record: {ZENODO_RECORD_ID}            ║")
    print(f"╚══════════════════════════════════════════╝")
    print(f"\nTarget directory: {output_dir.resolve()}")
    print(f"Sequences: {', '.join(targets.keys())}\n")

    if args.dry_run:
        print("--- DRY RUN (checking URLs only) ---\n")
        for name, filename in targets.items():
            url = f"{ZENODO_BASE_URL}/{filename}?download=1"
            try:
                resp = requests.head(url, timeout=10, allow_redirects=True)
                size = int(resp.headers.get("content-length", 0))
                size_mb = size / (1024 * 1024)
                status = "✓" if resp.status_code == 200 else "✗"
                print(f"  {status} {name}: {size_mb:.1f} MB — {url}")
            except requests.exceptions.RequestException as e:
                print(f"  ✗ {name}: {e}")
        return

    # Download
    success_count = 0
    for name, filename in targets.items():
        url = f"{ZENODO_BASE_URL}/{filename}?download=1"
        dest = output_dir / filename

        print(f"\n[{name}] Downloading...")
        if download_file(url, dest):
            success_count += 1

            # Extract if it's a zip
            if not args.no_extract and filename.endswith(".zip"):
                if extract_zip(dest, output_dir):
                    if not args.keep_zip:
                        dest.unlink()
                        print(f"  Removed {filename}")

    print(f"\n{'='*40}")
    print(f"Done! {success_count}/{len(targets)} files downloaded successfully.")
    print(f"Dataset location: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
