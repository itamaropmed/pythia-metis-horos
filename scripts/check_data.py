"""Smoke check: confirm the Mayo working set loads and matches the stored baseline."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RAW = ROOT / CFG["paths"]["raw_mayo"]
DER = ROOT / CFG["paths"]["derived_mayo"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    print(f"project root : {ROOT}")

    print("\n--- raw ---")
    for p in sorted(RAW.glob("*.parquet")):
        print(f"  {p.name:60s} {p.stat().st_size / 1e6:8.1f} MB")

    print("\n--- derived ---")
    df = pd.read_parquet(DER / "df.parquet")
    print(f"  df.parquet            {df.shape[0]:>6,} rows x {df.shape[1]:>3} cols")
    for arm, fn in CFG["mayo"]["arms"].items():
        X = pd.read_parquet(DER / fn)
        print(f"  arm {arm}: {fn:<14} {X.shape[0]:>6,} rows x {X.shape[1]:>3} cols")
    for arm in "ABC":
        P = np.load(DER / f"P{arm}.npy")
        print(f"  OOF P{arm}.npy         {P.shape[0]} predictors x {P.shape[1]:,} rows")

    print("\n--- target ---")
    y = df[CFG["mayo"]["target"]].to_numpy(float)
    mad = float(np.mean(np.abs(y - np.median(y))))
    print(f"  n            {len(y):,}")
    print(f"  patients     {df[CFG['mayo']['group']].nunique():,}")
    print(f"  median       {np.median(y):8.2f} min")
    print(f"  mean         {y.mean():8.2f} min")
    print(f"  sd           {y.std():8.2f} min")
    print(f"  MAD(Y)       {mad:8.2f} min   <- zero-information baseline")

    print("\n--- stored baseline (arm B) ---")
    print("  B*  calibrated floor  28.85   (>= 28.27 at the 5% quantile)")
    print("  Qcap guard (conditional full bound) 24.18")
    print("  E2  attainable floor  31.12   [30.43, 31.74]")
    print("  L   best OOF MAE      32.87   [32.26, 33.50]")

    print("\n--- frozen manifest ---")
    manifest = json.loads((ROOT / "docs" / "MAYO_DATA_MANIFEST.json").read_text())
    mismatches = []
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            mismatches.append(relative)
    expected_shapes = {key: tuple(value) for key, value in manifest["shapes"].items()}
    observed_shapes = {
        "df": df.shape,
        **{f"X{arm}": pd.read_parquet(DER / CFG["mayo"]["arms"][arm]).shape for arm in "ABC"},
        **{f"P{arm}": np.load(DER / f"P{arm}.npy").shape for arm in "ABC"},
    }
    shape_mismatches = sorted(key for key, shape in observed_shapes.items() if shape != expected_shapes[key])
    if mismatches or shape_mismatches:
        raise SystemExit(f"manifest mismatch: hashes={mismatches}, shapes={shape_mismatches}")
    print(f"  verified {len(manifest['files'])} hashes and {len(expected_shapes)} shapes")


if __name__ == "__main__":
    main()
