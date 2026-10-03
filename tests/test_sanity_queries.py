"""10 sanity queries pinned to the seed-42 generator output.

Runs against the NetworkX store by default (portable, no Docker). If NEO4J_URI is
set and reachable, the cross-backend tests also assert Neo4j returns the same
answers — the roadmap's "NetworkX fallback must pass the same 10 sanity queries".

Regenerate fixtures if the generator/seed changes:
    python -m src.data_gen.generate --seed 42
    python -m src.kg_build.build_graph_nx
"""
import os

import pytest

from src.kg.query import NetworkXStore

store = NetworkXStore()


# 1
def test_well_counts():
    assert store.count_wells_by_type() == {"injector": 2, "producer": 5}

# 2
def test_total_producers():
    assert sum(1 for _ in store._nodes("Well")) == 7

# 3
def test_injectors_of_f12():
    assert store.injectors_of("F-12") == ["F-4", "F-5"]

# 4
def test_strongest_connection():
    s = store.strongest_connection()
    assert (s["injector"], s["producer"]) == ("F-5", "F-11")
    assert s["weight"] == pytest.approx(0.4368, abs=1e-4)

# 5
def test_alert_code_breakdown():
    assert store.count_alerts_by_code() == {
        "HIGH_WOR": 185, "WATER_BREAKTHROUGH": 5, "WCT_SEVERE": 44}

# 6
def test_one_breakthrough_per_producer():
    # exactly 5 producers -> 5 first-crossing breakthrough alerts
    assert store.count_alerts_by_code()["WATER_BREAKTHROUGH"] == 5

# 7
def test_intervention_count():
    assert store.count_interventions() == 49

# 8
def test_cheapest_intervention():
    c = store.cheapest_intervention()
    assert c["est_cost_per_bbl"] == pytest.approx(8.28, abs=1e-2)
    assert c["action_type"] == "water_shutoff"

# 9
def test_latest_np_f12():
    assert store.latest_np("F-12") == pytest.approx(4579609.55, rel=1e-6)

# 10
def test_full_provenance_coverage():
    # every Well/MetricReading/Alert/Intervention carries a PROV back-pointer
    assert store.provenance_coverage() == 1.0


# --- cross-backend parity (only if Neo4j is up) ---
@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="Neo4j not configured")
def test_neo4j_parity():
    from src.kg.query import Neo4jStore
    neo = Neo4jStore()
    assert neo.count_wells_by_type() == store.count_wells_by_type()
    assert neo.injectors_of("F-12") == store.injectors_of("F-12")
    assert neo.provenance_coverage() == 1.0
    neo.close()
