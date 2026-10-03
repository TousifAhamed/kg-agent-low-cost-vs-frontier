"""Load and normalize the Volve 2018 production dataset (or emit a Volve-shaped seed).

Returns a tidy monthly panel per well plus a producer/injector split, so the
synthetic simulator can calibrate its CRM connectivity and water diagnostics to
real Volve volumes. Deterministic: no randomness lives here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical Volve wells (used for the no-file fallback seed).
_FALLBACK_PRODUCERS = ["15/9-F-1 C", "15/9-F-11", "15/9-F-12", "15/9-F-14", "15/9-F-15 D"]
_FALLBACK_INJECTORS = ["15/9-F-4", "15/9-F-5"]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _short(name: str) -> str:
    """'15/9-F-12' -> 'F-12'  (stable node id stem)."""
    return name.split("-", 1)[-1] if "-" in name else name


def load_volve(cfg: dict) -> dict:
    """Return {'monthly': DataFrame, 'producers': [...], 'injectors': [...],
    'source': str, 'checksum': str|None}.

    monthly columns: well, short, role, date, oil, gas, water, winj, pressure
    """
    vc = cfg["volve"]
    path = Path(vc["path"])
    if not path.exists():
        return _fallback_seed(cfg)

    df = pd.read_excel(path)
    name = "NPD_WELL_BORE_NAME"
    df["DATEPRD"] = pd.to_datetime(df["DATEPRD"])
    # Role by dominant WELL_TYPE per well (Volve mixes OP/WI rows for a few wells).
    role = (
        df.assign(is_inj=(df["WELL_TYPE"] == vc["injector_type"]).astype(int))
        .groupby(name)["is_inj"].mean()
        .apply(lambda x: "injector" if x >= 0.5 else "producer")
    )
    df["role"] = df[name].map(role)

    agg = {
        "BORE_OIL_VOL": "sum", "BORE_GAS_VOL": "sum", "BORE_WAT_VOL": "sum",
        "BORE_WI_VOL": "sum", "AVG_DOWNHOLE_PRESSURE": "mean",
    }
    monthly = (
        df.set_index("DATEPRD")
        .groupby([name, "role"])
        .resample(vc["resample"]).agg(agg)
        .reset_index()
        .rename(columns={
            name: "well", "BORE_OIL_VOL": "oil", "BORE_GAS_VOL": "gas",
            "BORE_WAT_VOL": "water", "BORE_WI_VOL": "winj",
            "AVG_DOWNHOLE_PRESSURE": "pressure", "DATEPRD": "date",
        })
    )
    monthly["short"] = monthly["well"].map(_short)
    monthly[["oil", "gas", "water", "winj"]] = monthly[["oil", "gas", "water", "winj"]].fillna(0.0)
    monthly["pressure"] = monthly.groupby("well")["pressure"].ffill().bfill().fillna(0.0)

    producers = sorted(monthly.loc[monthly.role == "producer", "well"].unique())
    injectors = sorted(monthly.loc[monthly.role == "injector", "well"].unique())
    return {
        "monthly": monthly, "producers": producers, "injectors": injectors,
        "source": str(path), "checksum": file_sha256(path),
    }


def _fallback_seed(cfg: dict) -> dict:
    """Volve-shaped synthetic panel when the real file is absent. Deterministic."""
    rng = np.random.default_rng(cfg["default_seed"])
    dates = pd.date_range("2008-01-01", "2016-09-01", freq=cfg["volve"]["resample"])
    rows = []
    for w in _FALLBACK_PRODUCERS + _FALLBACK_INJECTORS:
        inj = w in _FALLBACK_INJECTORS
        base = rng.uniform(3e4, 3e5)
        for i, d in enumerate(dates):
            decline = np.exp(-i / 60.0)
            wct = min(0.95, 0.05 + i / len(dates) * rng.uniform(0.6, 0.95))
            oil = 0.0 if inj else base * decline * (1 - wct)
            water = 0.0 if inj else base * decline * wct
            rows.append({
                "well": w, "short": _short(w), "role": "injector" if inj else "producer",
                "date": d, "oil": oil, "gas": oil * rng.uniform(80, 140),
                "water": water, "winj": base * 1.2 if inj else 0.0,
                "pressure": rng.uniform(220, 320),
            })
    monthly = pd.DataFrame(rows)
    return {
        "monthly": monthly, "producers": _FALLBACK_PRODUCERS, "injectors": _FALLBACK_INJECTORS,
        "source": "FALLBACK_SEED (no Volve file)", "checksum": None,
    }
