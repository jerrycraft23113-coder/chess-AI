"""
Download PGN files from all links in pgn_links.txt
Supports .zip (auto-extract) and .pgn files.
Parallel downloads with progress tracking.
"""

import os
import urllib.request
import urllib.parse
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import argparse
import time

# Repository root (parent of scripts/)
_REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Directories ──────────────────────────────────────────────
DATA_DIR = _REPO_ROOT / "data"
ZIP_DIR = DATA_DIR / "zip"
PGN_DIR = DATA_DIR / "pgn"

# Progress counters (thread-safe)
_lock = Lock()
_done = 0
_failed = 0
_skipped = 0


def _ensure_dirs():
    """Create output directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    ZIP_DIR.mkdir(exist_ok=True)
    PGN_DIR.mkdir(exist_ok=True)


def load_links(links_file: str = "config/pgn_links.txt") -> list:
    """Load all URLs from the links file.

    Args:
        links_file: Path relative to repo root, or absolute path.

    Returns:
        List of URL strings (empty lines and comments ignored).
    """
    path = Path(links_file)
    if not path.is_absolute():
        path = _REPO_ROOT / links_file
    if not path.exists():
        print(f"Error: {path} not found!")
        return []

    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def download_one(url: str, index: int, total: int, timeout: int = 60) -> bool:
    """Download a single URL (.zip or .pgn).

    Args:
        url: Full URL to download.
        index: 1-based index for progress display.
        total: Total number of URLs.
        timeout: HTTP timeout in seconds.

    Returns:
        True if successful, False otherwise.
    """
    global _done, _failed, _skipped

    filename = url.rsplit("/", 1)[-1]
    is_zip = filename.lower().endswith(".zip")

    # Determine output path
    if is_zip:
        out_path = ZIP_DIR / filename
        # Check if already extracted
        pgn_name = filename.replace(".zip", ".pgn")
        if (PGN_DIR / pgn_name).exists():
            with _lock:
                _skipped += 1
                print(f"[{index}/{total}] SKIP {filename} (already extracted)")
            return True
    else:
        out_path = PGN_DIR / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            with _lock:
                _skipped += 1
                print(f"[{index}/{total}] SKIP {filename} (already downloaded)")
            return True

    # Download
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()

        if not data:
            with _lock:
                _failed += 1
                print(f"[{index}/{total}] FAIL {filename} (empty response)")
            return False

        out_path.write_bytes(data)
        size_kb = len(data) / 1024

        # Extract zip
        if is_zip:
            try:
                with zipfile.ZipFile(out_path, "r") as zf:
                    zf.extractall(PGN_DIR)
            except zipfile.BadZipFile:
                with _lock:
                    _failed += 1
                    print(f"[{index}/{total}] FAIL {filename} (bad zip)")
                out_path.unlink(missing_ok=True)
                return False

        with _lock:
            _done += 1
            print(f"[{index}/{total}] OK   {filename} ({size_kb:.0f} KB)")
        return True

    except Exception as e:
        with _lock:
            _failed += 1
            print(f"[{index}/{total}] FAIL {filename} ({type(e).__name__}: {e})")
        return False


def download_all(urls: list, max_workers: int = 8):
    """Download all URLs in parallel.

    Args:
        urls: List of URLs to download.
        max_workers: Max concurrent downloads.
    """
    global _done, _failed, _skipped
    _done = 0
    _failed = 0
    _skipped = 0

    _ensure_dirs()
    total = len(urls)
    t0 = time.time()

    print(f"\nDownloading {total} files (max {max_workers} parallel)...")
    print("=" * 60)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_one, url, i + 1, total): url
            for i, url in enumerate(urls)
        }
        for future in as_completed(futures):
            future.result()  # propagate exceptions if any

    elapsed = time.time() - t0

    # Summary
    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"  Downloaded: {_done}")
    print(f"  Skipped:    {_skipped}")
    print(f"  Failed:     {_failed}")
    print(f"  Total:      {total}")

    # Count PGN files
    pgn_files = list(PGN_DIR.glob("*.pgn"))
    total_size_mb = sum(f.stat().st_size for f in pgn_files) / (1024 * 1024)
    print(f"\nPGN files: {len(pgn_files)} ({total_size_mb:.1f} MB)")
    print(f"Location:  {PGN_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Download all PGN files from config/pgn_links.txt (repo root paths)"
    )
    parser.add_argument("--links", type=str, default="config/pgn_links.txt",
                        help="Links file (relative to repo root, default: config/pgn_links.txt)")
    parser.add_argument("--parallel", type=int, default=8,
                        help="Max parallel downloads (default: 8)")
    parser.add_argument("--test", action="store_true",
                        help="Download only the first 5 links (test mode)")
    args = parser.parse_args()

    print("Chess PGN Data Downloader")
    print("=" * 60)

    urls = load_links(args.links)
    if not urls:
        print("No links found. Exiting.")
        return

    print(f"Found {len(urls)} links in {args.links}")

    if args.test:
        urls = urls[:5]
        print(f"Test mode: downloading first {len(urls)} only")

    download_all(urls, max_workers=args.parallel)


if __name__ == "__main__":
    main()
