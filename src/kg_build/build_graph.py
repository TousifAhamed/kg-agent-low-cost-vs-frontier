"""Materialize the KG into Neo4j (primary store) and emit a replayable Cypher dump.

Reads the same synthetic CSVs as the NetworkX fallback, maps them to ontology
instances with PROV back-pointers, and loads them into Neo4j Community (Docker).
Always writes kg/neo4j-dump.cypher; loads into a live Neo4j only if reachable.

Neo4j via Docker:  docker compose up -d   (see docker-compose.yml)
Run:  python -m src.kg_build.build_graph        # writes dump; loads if NEO4J reachable
Out:  kg/neo4j-dump.cypher
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

SYN = Path("data/synthetic/v1")
DUMP = Path("kg/neo4j-dump.cypher")


def _s(v) -> str:
    return str(v).replace("'", "\\'")


def cypher_statements() -> list[str]:
    """Build parameter-free but escaped Cypher for a portable, replayable dump."""
    wells = pd.read_csv(SYN / "wells.csv")
    conn = pd.read_csv(SYN / "connectivity.csv")
    readings = pd.read_csv(SYN / "readings.csv")
    alerts = pd.read_csv(SYN / "alerts.csv")
    interventions = pd.read_csv(SYN / "interventions.csv")

    st: list[str] = ["CREATE CONSTRAINT well_id IF NOT EXISTS FOR (w:Well) REQUIRE w.id IS UNIQUE;",
                     "MERGE (:Field {id:'F_Volve', name:'Volve'});"]
    for _, w in wells.iterrows():
        st.append(
            f"MERGE (w:Well {{id:'{w['well_id']}'}}) SET w.name='{_s(w['well_name'])}', "
            f"w.short='{_s(w['short'])}', w.well_type='{w['well_type']}', "
            f"w.source_record_id='{_s(w['source_record_id'])}', w.generator_seed={w['generator_seed']};")
    for _, r in readings.iterrows():
        st.append(
            f"MERGE (s:Sensor {{id:'{r['sensor_id']}'}}) SET s.metric_type='{r['metric']}', s.unit='{r['unit']}';")
        st.append(
            f"MERGE (m:MetricReading {{id:'{r['reading_id']}'}}) SET m.metric='{r['metric']}', "
            f"m.value={r['value']}, m.unit='{r['unit']}', m.timestamp='{r['date']}', "
            f"m.source_record_id='{_s(r['source_record_id'])}', m.generator_seed={r['generator_seed']};")
        st.append(
            f"MATCH (m:MetricReading {{id:'{r['reading_id']}'}}),(s:Sensor {{id:'{r['sensor_id']}'}}) "
            f"MERGE (m)-[:MEASURED_BY]->(s);")
        st.append(
            f"MATCH (s:Sensor {{id:'{r['sensor_id']}'}}),(w:Well {{id:'W_{_s(r['short'])}'}}) MERGE (s)-[:onWell]->(w);")
    for _, c in conn.iterrows():
        si, sp = c["injector"].split("-", 1)[-1], c["producer"].split("-", 1)[-1]
        st.append(
            f"MATCH (i:Well {{id:'W_{_s(si)}'}}),(p:Well {{id:'W_{_s(sp)}'}}) "
            f"MERGE (i)-[e:INJECTS_INTO]->(p) SET e.weight={c['weight']}, e.tau_days={c['tau_days']}, "
            f"e.noise_sigma={c['noise_sigma']}, e.source_record_id='{_s(c['source_record_id'])}';")
    for _, a in alerts.iterrows():
        st.append(
            f"MERGE (a:Alert {{id:'{a['alert_id']}'}}) SET a.alert_code='{a['alert_code']}', "
            f"a.severity='{a['severity']}', a.raised_at='{a['raised_at']}', "
            f"a.source_record_id='{_s(a['source_record_id'])}', a.generator_seed={a['generator_seed']};")
        st.append(
            f"MATCH (a:Alert {{id:'{a['alert_id']}'}}),(m:MetricReading {{id:'{a['triggered_by']}'}}) "
            f"MERGE (a)-[:TRIGGERED_BY]->(m);")
    for _, iv in interventions.iterrows():
        st.append(
            f"MERGE (v:Intervention {{id:'{iv['intervention_id']}'}}) SET v.action_type='{iv['action_type']}', "
            f"v.est_cost_per_bbl={iv['est_cost_per_bbl']}, v.expected_uplift_bbl={iv['expected_uplift_bbl']}, "
            f"v.status='{iv['status']}', v.source_record_id='{_s(iv['source_record_id'])}';")
        st.append(
            f"MATCH (v:Intervention {{id:'{iv['intervention_id']}'}}),(a:Alert {{id:'{iv['recommended_for']}'}}) "
            f"MERGE (v)-[:RECOMMENDED_FOR]->(a);")
    return st


def load_into_neo4j(statements: list[str]) -> bool:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    try:
        from neo4j import GraphDatabase
        drv = GraphDatabase.driver(uri, auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "cew1password")))
        with drv.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
            for stmt in statements:
                s.run(stmt)
        drv.close()
        return True
    except Exception as e:  # Docker/Neo4j not up -> dump still written; NetworkX fallback covers M2
        print(f"[neo4j] not loaded ({type(e).__name__}: {e}); dump written, NetworkX fallback available.")
        return False


def main() -> None:
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    st = cypher_statements()
    DUMP.write_text("\n".join(st))
    print(f"Wrote {len(st)} Cypher statements -> {DUMP}")
    if load_into_neo4j(st):
        print("Loaded into Neo4j.")


if __name__ == "__main__":
    main()
