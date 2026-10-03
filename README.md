# Can a Knowledge-Graph Agent Close the Gap Between a Low-Cost and a Frontier-Class LLM? An Audited Upstream Petroleum Case Study

**Tousifahamed Allabksha Nadaf** (Wipro) · NeurIPS 2026 Workshop SLM-Agents (1st Workshop on SLMs for
Agentic Systems), Paris · Poster

📄 **Paper:** [`manuscript/main.pdf`](manuscript/main.pdf) · **OpenReview:** https://openreview.net/forum?id=9TWY7YtnfX

Code, data, run logs, and audit verdicts for the paper. Every number in the paper can be recomputed
**offline from the stored runs**, with no API keys (see [Reproduce the paper](#reproduce-the-paper-offline)).

## TL;DR

One ReAct agent over a knowledge graph of the public Volve oil field, with a structured graph-query
tool, a deterministic aggregation tool (`compute_metric`) and a citation-traceability guardrail, is run on
a 50-question upstream-petroleum benchmark with a low-cost backbone (**Gemini 3.1 Flash-Lite**) and a
frontier-class one (**Claude Opus 4.8**), three runs each. The two land within one exact-match item of
each other, which this benchmark cannot separate. Hand-auditing **both** backbones' answers flipped which
one was ahead, and exposed 7 defective benchmark items.

## Results

Full agent, 36 exact-match (EM) items, mean of 3 runs per backbone:

| | Gemini 3.1 Flash-Lite | Claude Opus 4.8 |
|---|---|---|
| Strict EM | 0.741 | 0.852 |
| Normalized EM | 0.898 | 0.889 |
| **Audited EM** (every answer judged by hand) | **0.926** | **0.954** |

The backbones differ on 3 of 36 items on audit, all in Opus's favour (two-sided sign test p = 0.25).

Ablation ladder (Flash-Lite, normalized EM): LLM-only 0.028 · Text-RAG 0.583 · graph-retrieval-only
0.278 · **full agent 0.898** · without `compute_metric` 0.750 · without guardrail 0.861.

## Repository layout

| Path | Contents |
|---|---|
| `src/data_gen/` | Builds the derived tables from the Volve production data (alert rules, CRM connectivity, interventions) |
| `src/kg_build/` | Builds the knowledge graph (in-memory NetworkX; optional Neo4j export) |
| `src/kg/`, `src/agent/` | Graph queries; the ReAct agent, its three tools, the citation guardrail, perturbations |
| `src/baselines/` | S1 LLM-only, S2 Text-RAG (FAISS + MiniLM), S3 graph-retrieval-only |
| `src/eval/` | Benchmark builder, strict/normalized graders, rubric judge, experiment drivers, regrade, **symmetric audit** (`audit_both.py`), appendix tables |
| `src/llm.py` | Provider-agnostic backbone wrapper (Gemini / Claude) |
| `configs/`, `ontology/`, `prompts/` | Generator config, ontology (JSON-LD), ReAct system prompt |
| `data/` | Data card, derived tables (`synthetic/v1/`), the 50-item benchmark (`benchmark/benchmark.jsonl`) |
| `docs/questions.md` | Human-readable list of the 50 questions |
| `kg/` | Graph exports (JSON-LD, Cypher dump) |
| `results/` | **All run logs** (`*.jsonl`, one row per question and run) and the audit verdicts |
| `tables/` | Aggregated result tables |
| `manuscript/` | LaTeX source, figures, generated appendix tables, and the camera-ready PDF |
| `tests/` | Graph sanity queries |

Which files are which run:

| Paper | Files |
|---|---|
| Small backbone (Gemini 3.1 Flash-Lite) | `results/gemini/` — `S1`, `S2_minilm_k10`, `S3_kg_rag`, `S4_r1..3`, `S4_no_compute`, `S4_no_guardrail` |
| Robustness (paper's **RQ2**) | `results/gemini/RQ3_dropout{30,60}`, `RQ3_noise{30,60}` (files keep their pre-renumbering names) |
| Large backbone (Claude Opus 4.8) | `results/baselines/` (S1–S3; S2 run is `S2_text_rag_minilm_k10`) and `results/experiments/S4_r1..3` |
| Symmetric audit (Table 1, Appendix C) | `results/audit/em_audit_both_backbones.csv` — 216 rows, verdict and reason per answer |

## Reproduce the paper (offline)

Python 3.11, CPU only, no API keys. The stored run logs are the inputs.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m src.kg_build.build_graph_nx   # knowledge graph -> kg/networkx-graph.pkl (~4 s, deterministic)
pytest -q                               # graph sanity queries
python -m src.eval.regrade              # strict vs normalized EM for every run -> tables/em_regrade.csv
python -m src.eval.experiments_gemini   # aggregate small-backbone runs (stored strict EM, NDCG, rubric)
python -m src.eval.experiments          # aggregate large-backbone runs
python -m src.eval.audit_both           # symmetric audit: Table 1, paired bootstrap, sign test
python -m src.eval.appendix_tables      # appendix tables -> manuscript/generated/
python manuscript/make_figures.py       # Figures 1 and 2
python -m src.eval.build_benchmark      # optional: regenerate the 50-item benchmark from the graph
cd manuscript && pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```

Every step before the LaTeX build reproduces the committed files byte for byte (checked on the release
commit); the rebuilt PDF differs only in its embedded timestamps. Normalized EM per run is in
`tables/em_regrade.csv`; the audited scores and paired statistics are printed by `audit_both`.

## Re-running the LLM experiments (optional; needs API keys and costs money)

```bash
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
export ANTHROPIC_API_KEY=...

LLM_BACKEND=gemini python -m src.eval.experiments_gemini --run   # all small-backbone conditions
LLM_BACKEND=claude python -m src.eval.run_all --run              # large-backbone baselines S1, S2
LLM_BACKEND=claude python -m src.eval.experiments --run          # large-backbone full-agent runs
```

Each driver skips conditions whose log already exists, so move `results/` aside to start fresh. Note that
`experiments.py --run` also runs ablations and perturbations, which the paper ran on the small backbone
only. Re-runs will **not** reproduce the exact numbers: the backbones were called through floating API
aliases (`gemini-3.1-flash-lite`, `claude-opus-4-8`) at default temperature (large-backbone runs on
2026-07-03, small-backbone runs on 2026-07-17 to 07-22), and no snapshot identifiers were logged.

## Implementation notes

- **Graph backend.** Every reported run queries an in-memory NetworkX graph. `src/kg_build/build_graph.py`
  writes a Cypher export for Neo4j 5.26 (`docker-compose.yml`), and `tests/` runs a Neo4j parity test when
  `NEO4J_URI` is set, but Neo4j was not used in the reported experiments.
- **`cypher_query` does not run Cypher.** Despite its name, the model chooses a structured operation
  (`nodes`, `neighbours`, `latest`, `subgraph`) plus filters; deterministic code executes it.
- **The guardrail checks traceability, not support.** A cited ID is accepted only if it exists in the graph
  and a tool returned it during the run; it does not verify that the node supports the claim.
- **The rubric judge is the backbone being scored** (Flash-Lite judged Flash-Lite, Opus judged Opus), so
  explanation scores are not comparable across backbones.
- **Perturbations** (`src/agent/perturb.py`): edge dropout removes `INJECTS_INTO` edges and `MetricReading`
  nodes; weight noise perturbs connectivity weights only. Alerts, interventions, wells, and the
  `semantic_search` fallback corpus are never perturbed. One draw (seed 42) per level.

## Known benchmark defects

The benchmark is kept exactly as published so the reported numbers reproduce. Seven items are defective
(paper, Appendix A); the audit judges answers against the questions as worded with corrected gold.

| Item | Defect |
|---|---|
| Q16 | Asks *which* pairs exceed weight 0.3, but the gold is the count (2). |
| Q19 | Gold `false` comes from a generator bug: `NetworkXStore.injectors_of` returns no injectors for well names containing a space (`F-1 C`, `F-15 D`). The correct answer is `true`. |
| Q21, Q40 | "Earliest breakthrough" is a tie (F-12 and F-14, both 2010-06-01); the gold names one. |
| Q25 | Asks how many producers *currently* exceed 90 % water cut (1), but the gold counts those that *ever* did (2). |
| Q39, Q43 | Gold evidence is one specific alert, but the question admits others, so a correct answer citing a different valid alert scores 0. |

## Data and license

- **Code:** MIT (see `LICENSE`).
- **Data:** derived from Equinor's Volve field dataset, used under Equinor's *Terms and Conditions for Use
  of License to Data* (https://www.equinor.com/energy/volve-data-sharing): based on CC BY 4.0, with the
  licensed material not to be sold. The raw Equinor file is **not** redistributed; the derived tables in
  `data/synthetic/v1/` were generated from it (checksum in `manifest.json`). Alerts are rule-derived and
  intervention costs are simulated. See [`data/README.md`](data/README.md).

## Citation

```bibtex
@inproceedings{nadaf2026kgagent,
  title     = {Can a Knowledge-Graph Agent Close the Gap Between a Low-Cost and a Frontier-Class {LLM}?
               An Audited Upstream Petroleum Case Study},
  author    = {Nadaf, Tousifahamed Allabksha},
  booktitle = {NeurIPS 2026 Workshop on SLMs for Agentic Systems (SLM-Agents)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=9TWY7YtnfX}
}
```

Contact: ahamedpapers@gmail.com. An AI coding and writing assistant was used during this project, as
disclosed in the paper.
