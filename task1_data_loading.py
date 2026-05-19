import pandas as pd
import numpy as np
import glob
import os

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR = r'C:\Users\HP\Downloads\milan_traffic'
OUTPUT   = r'C:\Users\HP\Downloads\milan_traffic\traffic_optimized.parquet'

# ── HELPER ───────────────────────────────────────────────────────────────────
def mem_mb(df):
    return df.memory_usage(deep=True).sum() / 1024**2

# ── FIND FILES ───────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sms-call-internet-mi-*.txt')))
print(f"Found {len(files)} files\n")

# ── LOAD IN CHUNKS ───────────────────────────────────────────────────────────
CHUNKSIZE = 300_000
chunks = []
total_rows = 0

for i, filepath in enumerate(files):
    for chunk in pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        usecols=[0, 1, 4],        # square_id, time_interval, internet traffic
        names=['square_id', 'time_interval', 'internet'],
        chunksize=CHUNKSIZE
    ):
        # Drop missing internet activity
        chunk = chunk.dropna(subset=['internet'])

        # Optimized dtypes
        chunk['square_id']     = chunk['square_id'].astype(np.int16)
        chunk['time_interval'] = chunk['time_interval'].astype(np.int64)
        chunk['internet']      = chunk['internet'].astype(np.float32)

        chunks.append(chunk)
        total_rows += len(chunk)

    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{len(files)} files...")

print(f"\nAll files loaded. Total rows: {total_rows:,}")

# ── COMBINE ──────────────────────────────────────────────────────────────────
print("Combining chunks...")
df = pd.concat(chunks, ignore_index=True)

# Memory comparison
optimized_mem = mem_mb(df)
unoptimized_mem = (df.shape[0] * (8 + 8 + 8)) / 1024**2  # int64+int64+float64
print(f"\n--- Memory Report ---")
print(f"Unoptimized (estimated): {unoptimized_mem:.1f} MB")
print(f"Optimized (actual):      {optimized_mem:.1f} MB")
print(f"Memory saved:            {unoptimized_mem - optimized_mem:.1f} MB ({(1 - optimized_mem/unoptimized_mem)*100:.0f}% reduction)")

# ── CONVERT TIMESTAMP ────────────────────────────────────────────────────────
print("\nConverting timestamps...")
df['datetime'] = pd.to_datetime(df['time_interval'], unit='ms')
df = df.drop(columns=['time_interval'])
df = df.sort_values(['square_id', 'datetime']).reset_index(drop=True)

# ── QUICK SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n--- Dataset Summary ---")
print(f"Shape:        {df.shape}")
print(f"Date range:   {df['datetime'].min()} → {df['datetime'].max()}")
print(f"Unique areas: {df['square_id'].nunique()}")
print(f"\nSample rows:")
print(df.head())

# ── SAVE AS PARQUET ──────────────────────────────────────────────────────────
print(f"\nSaving to parquet...")
df.to_parquet(OUTPUT, index=False, compression='snappy')
print(f"Done! Saved to: {OUTPUT}")