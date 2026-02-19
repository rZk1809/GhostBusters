"""
Generate manifest.json for the versioned AMLSim dataset.
Records row counts, file sizes, and column schemas for reproducibility.
"""
import os
import json
import hashlib
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "amlsim_v1")

def file_info(path):
    size = os.path.getsize(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()
        rows = sum(1 for _ in f)  # count data rows (excludes header)
    cols = [c.strip() for c in header.split(",")]
    # quick hash of first 4KB for integrity
    with open(path, "rb") as f:
        h = hashlib.md5(f.read(4096)).hexdigest()
    return {"rows": rows, "size_bytes": size, "columns": cols, "md5_head": h}

manifest = {"created": datetime.now().isoformat(), "splits": {}}

for split in ["train", "val", "test"]:
    split_dir = os.path.join(BASE, split)
    manifest["splits"][split] = {}
    for fname in sorted(os.listdir(split_dir)):
        if fname.endswith(".csv"):
            info = file_info(os.path.join(split_dir, fname))
            manifest["splits"][split][fname] = info
            print(f"  {split}/{fname}: {info['rows']:,} rows, {info['size_bytes']:,} bytes, {len(info['columns'])} cols")

out_path = os.path.join(BASE, "manifest.json")
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nManifest written to {out_path}")
