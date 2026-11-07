"""Read the raw pipe-separated patient files into one hourly table."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ALL_COLUMNS, SITES


def _read(path: Path, site: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", dtype=np.float32)
    missing = set(ALL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    df["patient_id"] = path.stem
    df["site"] = site
    df["hour"] = np.arange(len(df), dtype=np.int16)
    return df


def load_raw(root: str | Path, limit: int = 0, workers: int = 8) -> pd.DataFrame:
    root = Path(root)
    frames = []
    for site, folder in SITES.items():
        files = sorted((root / folder).glob("p*.psv"))
        if limit:
            files = files[:limit]
        if not files:
            raise FileNotFoundError(f"No .psv files in {root / folder}")
        with ThreadPoolExecutor(workers) as ex:
            frames.extend(ex.map(lambda p: _read(p, site), files))
    return pd.concat(frames, ignore_index=True)
