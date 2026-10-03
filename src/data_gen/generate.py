"""Synthetic Dashboard Simulator v1.0 — deterministic generator.

Calibrates to real Volve monthly volumes and emits the two proprietary-dashboard
stand-ins the project unifies:
  * Well Connectivity Dashboard  -> CRM-style injector->producer connectivity edges
  * Water Management Analytics    -> water-cut / log10(WOR) / Np + breakthrough alerts
  * Intervention log              -> actions with cost-per-barrel outcomes

Every emitted node/edge carries a PROV back-pointer {source_record_id, timestamp,
generator_seed} so the M3 citation guardrail and the hallucination metric have ground truth.

Run:  python -m src.data_gen.generate --seed 42
Out:  data/synthetic/v1/{wells,connectivity,readings,alerts,interventions}.csv + manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .volve_loader import load_volve

OUT_DIR = Path("data/synthetic/v1")
CFG_PATH = Path("configs/generator.yaml")


def _pos(name: str) -> np.ndarray:
    """Deterministic 2-D map position from a stable hash of the well name."""
    h = hashlib.md5(name.encode()).digest()
    return np.array([h[0], h[1]], dtype=float) / 255.0


def build_connectivity(volve: dict, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """CRM-style gains f_ij: distance-decayed, size-weighted, softmax per injector, + noise."""
    cc = cfg["connectivity"]
    prod = volve["producers"]
    inj = volve["injectors"]
    monthly = volve["monthly"]
    liq = monthly[monthly.role == "producer"].groupby("well")[["oil", "water"]].sum().sum(axis=1)
    rows = []
    for i in inj:
        pi = _pos(i)
        raw = {}
        for p in prod:
            d = np.linalg.norm(pi - _pos(p))
            size = float(liq.get(p, 1.0))
            raw[p] = np.exp(-d * 3.0) * np.log1p(size)
        z = np.array(list(raw.values()))
        w = z / z.sum()  # softmax-like normalization: gains per injector sum to 1
        for p, wij in zip(raw, w):
            noisy = float(np.clip(wij + rng.normal(0, cc["noise_sigma"]), 0.0, 1.0))
            if noisy < cc["min_edge_weight"]:
                continue
            rows.append({
                "edge_id": f"C_{i.split('-')[-1]}_{p.split('-')[-1]}".replace(" ", ""),
                "injector": i, "producer": p, "weight": round(noisy, 4),
                "tau_days": cc["tau_days"], "noise_sigma": cc["noise_sigma"],
                "source_record_id": f"volve:conn:{i}->{p}",
            })
    return pd.DataFrame(rows)


def build_readings(volve: dict) -> pd.DataFrame:
    """Per-producer monthly diagnostics: water_cut, log10(WOR), Np (cum oil), pressure."""
    m = volve["monthly"]
    prod = m[m.role == "producer"].copy().sort_values(["well", "date"])
    prod["water_cut"] = prod["water"] / (prod["oil"] + prod["water"]).replace(0, np.nan)
    prod["wor"] = prod["water"] / prod["oil"].replace(0, np.nan)
    prod["log_wor"] = np.log10(prod["wor"].replace(0, np.nan))
    prod["np_cum"] = prod.groupby("well")["oil"].cumsum()

    rows = []
    for _, r in prod.iterrows():
        stem = f"{r['short']}_{r['date'].date()}".replace(" ", "")
        for metric, val, unit in [
            ("water_cut", r["water_cut"], "frac"),
            ("log_wor", r["log_wor"], "log10"),
            ("np_cum", r["np_cum"], "sm3"),
            ("pressure", r["pressure"], "bar"),
        ]:
            if pd.isna(val):
                continue
            rows.append({
                "reading_id": f"R_{metric}_{stem}",
                "well": r["well"], "short": r["short"], "date": r["date"],
                "metric": metric, "value": round(float(val), 4), "unit": unit,
                "sensor_id": f"S_{metric}_{r['short']}",
                "source_record_id": f"volve:prod:{r['well']}:{r['date'].date()}",
            })
    return pd.DataFrame(rows)


def build_alerts(readings: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Rule-based water alerts, each TRIGGERED_BY a specific reading."""
    w = cfg["water"]
    rows = []
    wct = readings[readings.metric == "water_cut"].sort_values(["well", "date"])
    for well, g in wct.groupby("well"):
        broke = False
        for _, r in g.iterrows():
            v = r["value"]
            code, sev = None, None
            if v >= w["severe_wct"]:
                code, sev = "WCT_SEVERE", "high"
            elif not broke and v >= w["breakthrough_wct"]:
                code, sev = "WATER_BREAKTHROUGH", "medium"
                broke = True
            if code:
                rows.append({
                    "alert_id": f"A_{code}_{r['short']}_{r['date'].date()}".replace(" ", ""),
                    "alert_code": code, "severity": sev, "well": well,
                    "raised_at": r["date"], "triggered_by": r["reading_id"],
                    "source_record_id": f"rule:{code}:{r['reading_id']}",
                })
    log_wor = readings[readings.metric == "log_wor"]
    for _, r in log_wor[log_wor.value >= w["high_wor_log"]].iterrows():
        rows.append({
            "alert_id": f"A_HIGH_WOR_{r['short']}_{r['date'].date()}".replace(" ", ""),
            "alert_code": "HIGH_WOR", "severity": "medium", "well": r["well"],
            "raised_at": r["date"], "triggered_by": r["reading_id"],
            "source_record_id": f"rule:HIGH_WOR:{r['reading_id']}",
        })
    return pd.DataFrame(rows)


def build_interventions(alerts: pd.DataFrame, readings: pd.DataFrame, cfg: dict,
                        rng: np.random.Generator) -> pd.DataFrame:
    """One recommended intervention per breakthrough/severe alert, with cost-per-bbl outcome."""
    ic = cfg["intervention"]
    # Recommend on the more serious alerts only.
    serious = alerts[alerts.alert_code.isin(["WATER_BREAKTHROUGH", "WCT_SEVERE"])]
    np_by_well = (readings[readings.metric == "np_cum"].sort_values("date")
                  .groupby("well").last()["reading_id"].to_dict())
    rows = []
    for _, a in serious.iterrows():
        action = rng.choice(ic["actions"])
        cost = float(np.exp(rng.normal(np.log(ic["cost_per_bbl_mean"]), ic["cost_per_bbl_sigma"])))
        uplift = float(max(0.0, rng.normal(ic["expected_uplift_bbl_mean"],
                                           ic["expected_uplift_bbl_mean"] * 0.3)))
        if rng.random() < ic["label_noise"]:      # avoid too-clean outcomes
            uplift *= rng.uniform(0.3, 1.7)
        rows.append({
            "intervention_id": f"I_{a['alert_id']}".replace("A_", ""),
            "action_type": action, "est_cost_per_bbl": round(cost, 2),
            "expected_uplift_bbl": round(uplift, 1), "status": "recommended",
            "recommended_for": a["alert_id"], "well": a["well"],
            "followed_by": np_by_well.get(a["well"], ""),
            "source_record_id": f"sim:intervention:{a['alert_id']}",
        })
    return pd.DataFrame(rows)


def build_wells(volve: dict) -> pd.DataFrame:
    rows = []
    for role, key in [("producer", "producers"), ("injector", "injectors")]:
        for w in volve[key]:
            short = w.split("-", 1)[-1] if "-" in w else w
            rows.append({"well_id": f"W_{short}".replace(" ", ""), "well_name": w,
                         "short": short, "well_type": role,
                         "field": "Volve", "source_record_id": f"volve:well:{w}"})
    return pd.DataFrame(rows)


def _checksum_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def generate(seed: int, cfg: dict) -> dict:
    rng = np.random.default_rng(seed)
    volve = load_volve(cfg)
    wells = build_wells(volve)
    connectivity = build_connectivity(volve, cfg, rng)
    readings = build_readings(volve)
    alerts = build_alerts(readings, cfg)
    interventions = build_interventions(alerts, readings, cfg, rng)

    for df in (wells, connectivity, readings, alerts, interventions):
        df.insert(0, "generator_seed", seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {"wells": wells, "connectivity": connectivity, "readings": readings,
              "alerts": alerts, "interventions": interventions}
    manifest = {"version": cfg["version"], "seed": seed, "volve_source": volve["source"],
                "volve_checksum": volve["checksum"], "counts": {}, "checksums": {}}
    for name, df in tables.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        manifest["counts"][name] = len(df)
        manifest["checksums"][name] = _checksum_df(df)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (OUT_DIR / "PROVENANCE.txt").write_text(
        f"Synthetic Dashboard Simulator v{cfg['version']} | seed={seed}\n"
        f"Volve source: {volve['source']}\nVolve checksum: {volve['checksum']}\n"
        f"{'*** FALLBACK SEED — not calibrated to real Volve ***' if volve['checksum'] is None else 'Calibrated to real Volve 2018 volumes.'}\n"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", type=Path, default=CFG_PATH)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seed = args.seed if args.seed is not None else cfg["default_seed"]
    manifest = generate(seed, cfg)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
