"""S2 baseline: Text-RAG (dashboard outputs dumped as text).

Embeds the serialized dashboard documents (src/baselines/corpus.py) with a sentence
embedder, retrieves the top-k most similar to each question via FAISS, and feeds them
to the backbone. This is the strong-but-unstructured baseline the KG agent must beat
(RQ2: +10pp EM / +0.10 NDCG target).

Supports the roadmap's config sweep (embedder in {MiniLM, BGE-small}, top_k in {3,5,10});
report the BEST config, not the first.

Run:  python -m src.baselines.text_rag --embedder minilm --topk 5
Out:  results/baselines/S2_text_rag_<embedder>_k<k>.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.baselines.corpus import build_documents
from src.baselines.llm_only import load_bench
from src.eval.metrics import grade
from src.llm import LLM

EMBEDDERS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2",
             "bge": "BAAI/bge-small-en-v1.5"}

SYSTEM = ("You are a petroleum production analyst. Use ONLY the provided dashboard context to answer. "
          "Be concise. List well identifiers or give the exact number as asked; for ranking, list ids in order. "
          "If the context is insufficient, say so. End with a line 'ANSWER: <concise answer>'.")


def build_index(model: SentenceTransformer, docs: list[dict]):
    emb = model.encode([d["text"] for d in docs], normalize_embeddings=True,
                        show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def run(embedder: str, topk: int) -> Path:
    out = Path(f"results/baselines/S2_text_rag_{embedder}_k{topk}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    docs = build_documents()
    model = SentenceTransformer(EMBEDDERS[embedder])
    index = build_index(model, docs)
    llm = LLM()
    rows = []
    for q in load_bench():
        qv = model.encode([q["question"]], normalize_embeddings=True).astype("float32")
        _, idx = index.search(qv, topk)
        ctx = "\n".join(f"[{docs[i]['id']}] {docs[i]['text']}" for i in idx[0])
        pred = llm.ask(SYSTEM, f"Context:\n{ctx}\n\nQuestion: {q['question']}")
        ans = pred.split("ANSWER:")[-1].strip() if "ANSWER:" in pred else pred
        g = grade(q, ans)
        rows.append({"question_id": q["id"], "system": "S2", "category": q["category"],
                     "embedder": embedder, "topk": topk, "answer": ans,
                     "retrieved": [docs[i]["id"] for i in idx[0]], **g})
        print(f"{q['id']} {q['category'][:4]} em={g['exact_match']} ndcg={g['ndcg_at_5']}")
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"S2 [{embedder} k={topk}] done: {len(rows)} answers, {llm.calls} calls -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=list(EMBEDDERS), default="minilm")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    run(args.embedder, args.topk)


if __name__ == "__main__":
    main()
