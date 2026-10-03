"""RQ3 robustness: controlled corruption of the KG.

Two independent perturbations, applied to a copy of the seed-42 graph:
  * edge_dropout(p)  — remove a fraction p of INJECTS_INTO edges + MetricReading nodes
                       (models an INCOMPLETE graph: missing connectivity / sensor gaps)
  * weight_noise(s)  — add Gaussian noise (relative sigma s) to CRM connectivity weights
                       (models a NOISY graph: mis-estimated connectivity)

We re-run S4 on each perturbed graph and measure EM/NDCG degradation vs the clean KG.
Deterministic given a seed so the robustness curve is reproducible.
"""
from __future__ import annotations

import random

import networkx as nx

from src.agent.retrieval import KG


def _copy(kg: KG) -> nx.MultiDiGraph:
    return kg.g.copy()


def edge_dropout(kg: KG, p: float, seed: int = 0) -> KG:
    rng = random.Random(seed)
    g = _copy(kg)
    inj = [(u, v, k) for u, v, k, d in g.edges(keys=True, data=True) if d.get("label") == "INJECTS_INTO"]
    for u, v, k in inj:
        if rng.random() < p:
            g.remove_edge(u, v, k)
    reads = [n for n, d in g.nodes(data=True) if d.get("label") == "MetricReading"]
    for n in reads:
        if rng.random() < p:
            g.remove_node(n)
    return KG(graph=g)


def weight_noise(kg: KG, sigma: float, seed: int = 0) -> KG:
    rng = random.Random(seed)
    g = _copy(kg)
    for u, v, k, d in g.edges(keys=True, data=True):
        if d.get("label") == "INJECTS_INTO" and "weight" in d:
            d["weight"] = max(0.0, d["weight"] * (1.0 + rng.gauss(0, sigma)))
    return KG(graph=g)
