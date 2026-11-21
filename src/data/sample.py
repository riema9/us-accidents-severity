# src/data/sample.py
"""Random sampling of the full 7.7M-row CSV without loading it all into RAM.

The raw file is read in chunks and a fixed fraction of each chunk is kept, so the
sample is spread across the whole 2016-2023 time range (unlike nrows=50000, which
only returns the oldest rows). The result is cached as parquet for fast reruns.
"""

import pandas as pd

from src.config import RAW_CSV_PATH, USECOLS, RANDOM_SEED, INTERIM_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _count_data_rows(path) -> int:
    """Fast line count (minus header). Only used to pick the sampling fraction."""
    n = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 24)
            if not buf:
                break
            n += buf.count(b"\n")
    return max(n - 1, 1)


def _parse_times(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Start_Time/End_Time. Some raw values carry fractional seconds, which
    makes pandas' format inference turn them into NaT; keep the first 19 characters
    ('YYYY-MM-DD HH:MM:SS') so every row parses the same way."""
    for col in ("Start_Time", "End_Time"):
        if col in df.columns:
            s = df[col].astype("string").str.slice(0, 19)
            df[col] = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return df


def create_sample(n_rows: int, chunksize: int = 500_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_CSV_PATH}. Put the CSV in data/raw/")

    logger.info("Counting rows in raw CSV (one-off, a few seconds)...")
    total = _count_data_rows(RAW_CSV_PATH)
    frac = min(1.0, n_rows / total)
    logger.info(f"Raw rows ~{total:,}; sampling fraction {frac:.4f} for ~{n_rows:,} rows")

    parts = []
    for i, chunk in enumerate(pd.read_csv(RAW_CSV_PATH, usecols=USECOLS, chunksize=chunksize)):
        parts.append(chunk.sample(frac=frac, random_state=seed + i))
        logger.info(f"  chunk {i + 1} done ({sum(len(p) for p in parts):,} rows kept)")

    df = pd.concat(parts, ignore_index=True)
    df = _parse_times(df)
    bad = int(df["Start_Time"].isna().sum())
    if bad:
        logger.info(f"{bad:,} rows have an unparseable Start_Time (clean_data will drop them)")
    return df


def load_sample(n_rows: int = 1_000_000, refresh: bool = False, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return a random sample of the raw data, creating and caching it on first use."""
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = INTERIM_DATA_DIR / f"sample_{n_rows}_seed{seed}.parquet"
    if cache.exists() and not refresh:
        logger.info(f"Loading cached sample from {cache}")
        return pd.read_parquet(cache)

    df = create_sample(n_rows, seed=seed)
    df.to_parquet(cache, index=False)
    logger.info(f"Cached sample ({df.shape}) at {cache}")
    return df
