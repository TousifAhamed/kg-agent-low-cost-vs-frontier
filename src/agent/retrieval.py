"""Grounded retrieval layer for the KG agent.

Wraps the pickled NetworkX MultiDiGraph with:
  * generic, injection-safe accessors (nodes_by_label / node / neighbours)
  * k-hop subgraph extraction around linked seed nodes
  * a FAISS semantic index over the same serialized dashboard docs S2 used

Everything returns records that INCLUDE the KG node id, so the ReAct loop can
cite real, existing evidence (validated by src/agent/cite.py). No string is ever
interpolated into a query; callers pass structured filters only.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

PKL = Path("kg/networkx-graph.pkl")


class KG:
    def __init__(self, path: Path = PKL, graph: nx.MultiDiGraph | None = None):
        if graph is not None:  # e.g. a perturbed copy (RQ3)
            self.g = graph
            return
        # SAFE: locally generated pickle (src/kg_build/build_graph_nx.py), not untrusted input.
        with open(path, "rb") as fh:
            self.g: nx.MultiDiGraph = pickle.load(fh)

    # --- primitives ---
    def has_node(self, nid: str) -> bool:
        return nid in self.g

    def node(self, nid: str) -> dict | None:
        return {"id": nid, **self.g.nodes[nid]} if nid in self.g else None

    def _match(self, d: dict, filters: dict) -> bool:
        for k, v in (filters or {}).items():
            av = d.get(k)
            if isinstance(v, str) and k == "well":
                if av is None or not str(av).endswith(v):
                    return False
            elif av != v:
                return False
        return True

    def nodes_by_label(self, label: str, filters: dict | None = None) -> list[dict]:
        return [{"id": n, **d} for n, d in self.g.nodes(data=True)
                if d.get("label") == label and self._match(d, filters or {})]

    def neighbours(self, nid: str) -> list[dict]:
        """One-hop out/in edges as {direction, rel, other, weight?}."""
        out = []
        if nid not in self.g:
            return out
        for _, v, k, d in self.g.out_edges(nid, keys=True, data=True):
            out.append({"direction": "out", "rel": d.get("label", k), "other": v,
                        **{a: d[a] for a in ("weight", "tau_days") if a in d}})
        for u, _, k, d in self.g.in_edges(nid, keys=True, data=True):
            out.append({"direction": "in", "rel": d.get("label", k), "other": u,
                        **{a: d[a] for a in ("weight", "tau_days") if a in d}})
        return out

    def khop_subgraph(self, seeds: list[str], hops: int = 1) -> dict:
        """Undirected k-hop expansion; returns {nodes:[...], edges:[...]} with ids."""
        seen = {s for s in seeds if s in self.g}
        frontier = set(seen)
        for _ in range(max(0, hops)):
            nxt = set()
            for n in frontier:
                nxt |= set(self.g.successors(n)) | set(self.g.predecessors(n))
            seen |= nxt
            frontier = nxt
        sub = self.g.subgraph(seen)
        nodes = [{"id": n, **d} for n, d in sub.nodes(data=True)]
        edges = [{"from": u, "to": v, "rel": d.get("label", k),
                  **{a: d[a] for a in ("weight",) if a in d}}
                 for u, v, k, d in sub.edges(keys=True, data=True)]
        return {"nodes": nodes, "edges": edges}


class SemanticIndex:
    """FAISS index over the S2 dashboard corpus (same docs, so S4 vs S2 is apples-to-apples)."""

    def __init__(self, embedder: str = "minilm"):
        import faiss
        from sentence_transformers import SentenceTransformer

        from src.baselines.corpus import build_documents
        from src.baselines.text_rag import EMBEDDERS
        self.docs = build_documents()
        self.model = SentenceTransformer(EMBEDDERS[embedder])
        emb = self.model.encode([d["text"] for d in self.docs], normalize_embeddings=True,
                                show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, query: str, k: int = 5) -> list[dict]:
        qv = self.model.encode([query], normalize_embeddings=True).astype("float32")
        _, idx = self.index.search(qv, k)
        return [{"id": self.docs[i]["id"], "text": self.docs[i]["text"]} for i in idx[0]]
