"""Download the open PhysioNet/CinC Challenge 2019 training data (40,336 patients).

Source: https://physionet.org/content/challenge-2019/1.0.0/ (open access).
Please cite Reyna et al., Critical Care Medicine 2020, and PhysioNet.

Each patient is one small file, so this fetches them in parallel and skips files
already on disk; if it stops, run it again and it picks up where it left off.

    python -m sepsis.download              # both hospitals -> data/raw/
    python -m sepsis.download --limit 500  # quick try: 500 patients per hospital
"""
from __future__ import annotations

import argparse
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .schema import SITES

BASE = "https://physionet.org/files/challenge-2019/1.0.0/training/{folder}/"


def list_files(folder: str) -> list[str]:
    with urllib.request.urlopen(BASE.format(folder=folder), timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="(p\d{6}\.psv)"', html)))


def fetch(url: str, dest: Path, tries: int = 3) -> bool:
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                dest.write_bytes(r.read())
            return True
        except Exception:  # network hiccup: retry
            continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--limit", type=int, default=0, help="patients per hospital (0 = all)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    for site, folder in SITES.items():
        d = Path(a.out) / folder
        d.mkdir(parents=True, exist_ok=True)
        names = list_files(folder)
        if a.limit:
            names = names[: a.limit]
        todo = [n for n in names if not (d / n).exists()]
        print(f"Hospital {site}: {len(names)} files, {len(todo)} to download")
        failed = 0
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(fetch, BASE.format(folder=folder) + n, d / n): n for n in todo}
            for i, f in enumerate(as_completed(futs), 1):
                failed += not f.result()
                if i % 2000 == 0:
                    print(f"  {i}/{len(todo)}")
        print(f"Hospital {site}: done ({failed} failed; run again to retry)")


if __name__ == "__main__":
    main()
