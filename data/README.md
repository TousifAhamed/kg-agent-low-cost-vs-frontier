# Data Card — Knowledge-Graph Agent on the Volve Field (upstream petroleum)

## Primary source: Volve Field Dataset (Equinor, 2018 release)
- **License:** Equinor's *Terms and Conditions for Use of License to Data*
  (https://www.equinor.com/energy/volve-data-sharing), based on CC BY 4.0 with two modifications: the
  licensed material may not be sold, and the license covers all data in the dataset. Attribution required.
- **What we use:** only the **production history** slice (well headers, daily oil/gas/water volumes,
  bottom-hole & wellhead pressure, water-cut). We do **not** need seismic, WITSML drilling, or the full ~40 GB.
- **Not redistributed here.** The raw file is not committed; download it from Equinor.

### Drop-path (only needed to regenerate `synthetic/v1/`)
```
data/raw/volve_production_data.xlsx      <-- the Equinor production-data workbook
```
Source: Equinor portal, https://www.equinor.com/energy/volve-data-sharing (register → Production data,
`Volve production data.xlsx`). Rename it to `volve_production_data.xlsx` and place it at `data/raw/`
(the path is set in `configs/generator.yaml`; the loader reads Excel via `pandas.read_excel`).
Then: `python -m src.data_gen.generate --seed 42`.

The released `synthetic/v1/` tables were generated from the real Equinor workbook
(SHA-256 `514d4e38763e09be7fbad12313429909b9799b1a6ec999bf5f36e0df1b6c9cae`, recorded in
`synthetic/v1/manifest.json` and `synthetic/v1/PROVENANCE.txt`), so the raw file is **not** needed to
reproduce the paper.

### Expected schema (columns the generator reads; extras ignored)
| Column (case-insensitive contains) | Maps to |
|---|---|
| `WELL` / `NPD_WELL_BORE_NAME`      | Well.wellName |
| `DATEPRD` / date                   | MetricReading.timestamp |
| `BORE_OIL_VOL`                     | oil rate (Np accumulation) |
| `BORE_WAT_VOL`                     | water volume (water-cut, WOR) |
| `BORE_GAS_VOL`                     | gas volume |
| `AVG_DOWNHOLE_PRESSURE`            | pressure reading (connectivity/CRM) |
| `FLOW_KIND` / `WELL_TYPE`          | producer vs injector role |

### Fallback (no Volve file)
If `data/raw/volve_production_data.xlsx` is absent, the generator emits a **Volve-shaped synthetic seed**
(documented Volve well names: 15/9-F-1 C, F-4, F-5, F-11, F-12, F-14, F-15 D + injectors) so the
pipeline still runs end-to-end, and says so in `PROVENANCE.txt`. The paper's numbers use the real file.

## Derived tables — `data/synthetic/v1/`
Deterministic (seed 42), versioned. Built from the real production series:
- **Wells and connectivity** — CRM-style injector→producer connectivity weights (τ = 45 days).
- **Readings** — per-producer water-cut, log10(WOR), cumulative oil (Np).
- **Alerts** — fixed rules: first water cut ≥ 0.50 → `WATER_BREAKTHROUGH`; water cut ≥ 0.90 →
  `WCT_SEVERE`; log10(WOR) ≥ 0 → `HIGH_WOR`.
- **Interventions** — one per breakthrough or severe alert; action and cost-per-barrel / uplift are
  **sampled** (simulated), not field data.

### Known limitations
- Volve is a **single field**, so generalization claims must be hedged.
- Alerts are rule-derived and intervention costs are simulated: they are model-generated ground truth,
  not field-verified.

## Benchmark — `data/benchmark/benchmark.jsonl`
50 questions with gold answers computed from the knowledge graph by `src/eval/build_benchmark.py`
(human-readable copy: `docs/questions.md`). Seven items are known to be defective (Q16, Q19, Q21, Q25,
Q39, Q40, Q43); they are kept unchanged so the published numbers reproduce, and are documented in the
paper's Appendix A and in the top-level README.
