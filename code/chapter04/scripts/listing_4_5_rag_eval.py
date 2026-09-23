"""Listing 4.5 -- Measuring RAG quality with Precision@k and Hit@k.

Run a small labelled query set (one expected document per query) through
the Listing 4.4 pipeline and report the two metrics every RAG team should
have before they reach for chunking or reranker tuning:

- Precision@k: of the k chunks we retrieved, what fraction came from the
  expected document.
- Hit@k:       did the expected document appear at least once in the top-k?
  (1 or 0 per query, averaged over queries; sometimes called success rate)

The mean of Precision@k across queries is the headline retrieval quality
score; the mean of Hit@k tells you how often retrieval handed the
generator any chance of being right at all.

Hit@1 is Hit@k with k=1 and is reported separately because top-1 is what a
no-rerank production setup actually shows the LLM as the most relevant chunk.

Why not "Recall@k"? Recall divides the relevant documents retrieved by all
the relevant documents that exist. With one labelled document per query the
two coincide, but published RAG benchmarks report recall on multi-document
labels, so the metric is named for what it measures here. ``recall_at_k``
stays as an alias for scripts written against earlier versions.

The script intentionally avoids any LLM call; it measures retrieval in
isolation so a bad answer can be attributed to retrieval vs. generation.
For end-to-end answer quality, use the LLM-as-judge pattern from
``prompt_validator.py`` (Listing 4.3).

Run from code/:

    python -m chapter04.scripts.listing_4_5_rag_eval \
        --docs chapter04/data/policy_docs.jsonl \
        --eval chapter04/data/rag_eval.jsonl \
        --k 3 \
        --output chapter04/runs/rag_eval.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from common.jsonl import read_jsonl_list

from chapter04.rag_minimal import EMBED_BACKENDS, MinimalRAG


def precision_at_k(retrieved: List[Dict[str, object]], expected_doc_id: str) -> float:
    """Fraction of retrieved chunks that came from the expected document."""
    if not retrieved:
        return 0.0
    hits = sum(1 for item in retrieved if item["metadata"]["id"] == expected_doc_id)
    return hits / len(retrieved)


def hit_at_k(retrieved: List[Dict[str, object]], expected_doc_id: str) -> float:
    """1.0 if the expected document appears anywhere in the top-k, else 0.0.

    To score multi-document labels as true recall, pass a set of expected ids
    and divide the number found by len(expected) instead.
    """
    return 1.0 if any(item["metadata"]["id"] == expected_doc_id for item in retrieved) else 0.0


recall_at_k = hit_at_k  # backwards-compatible alias (pre-2026-09 name)


def hit_at_1(retrieved: List[Dict[str, object]], expected_doc_id: str) -> float:
    """Hit@k with k=1: the top-ranked chunk came from the expected document."""
    return hit_at_k(retrieved[:1], expected_doc_id)


def evaluate(rag: MinimalRAG, queries: List[Dict[str, str]], k: int) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for case in queries:
        retrieved = rag.retrieve(case["query"], k=k)
        rows.append(
            {
                "query": case["query"],
                "expected_doc_id": case["expected_doc_id"],
                "top_k_doc_ids": [item["metadata"]["id"] for item in retrieved],
                "precision_at_k": round(precision_at_k(retrieved, case["expected_doc_id"]), 4),
                "hit_at_k": round(hit_at_k(retrieved, case["expected_doc_id"]), 4),
                "hit_at_1": round(hit_at_1(retrieved, case["expected_doc_id"]), 4),
            }
        )

    n = max(len(rows), 1)
    summary = {
        "k": k,
        "queries": len(rows),
        "mean_precision_at_k": round(sum(r["precision_at_k"] for r in rows) / n, 4),
        "mean_hit_at_k": round(sum(r["hit_at_k"] for r in rows) / n, 4),
        "mean_hit_at_1": round(sum(r["hit_at_1"] for r in rows) / n, 4),
    }
    return {"summary": summary, "per_query": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval evaluation (chapter 4, Listing 4.5)")
    parser.add_argument(
        "--docs",
        default="chapter04/data/policy_docs.jsonl",
        help="JSONL corpus with `id`, `title`, `text` fields",
    )
    parser.add_argument(
        "--eval",
        default="chapter04/data/rag_eval.jsonl",
        help="JSONL labelled queries: each row has `query` and `expected_doc_id`",
    )
    parser.add_argument("--k", type=int, default=3, help="Top-k cutoff for the metrics")
    parser.add_argument(
        "--backend",
        choices=sorted(EMBED_BACKENDS),
        default="st",
        help="Embedding backend (st = sentence-transformers, hash = token bag)",
    )
    parser.add_argument("--chunk_size", type=int, default=80, help="Chunk size in words")
    parser.add_argument("--chunk_overlap", type=int, default=20, help="Chunk overlap in words")
    parser.add_argument(
        "--output",
        default="chapter04/runs/rag_eval.json",
        help="Where to write the JSON report",
    )
    args = parser.parse_args()

    docs = read_jsonl_list(args.docs)
    queries = read_jsonl_list(args.eval)

    embed = EMBED_BACKENDS[args.backend]()
    rag = MinimalRAG(embed=embed, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    rag.ingest(docs)

    print(f"Indexed {len(rag.chunks)} chunks from {len(docs)} documents (backend={args.backend})")
    print(f"Evaluating {len(queries)} labelled queries at k={args.k}")

    started = time.time()
    report = evaluate(rag, queries, k=args.k)
    elapsed = time.time() - started

    summary = report["summary"]
    print()
    print("=" * 56)
    print("RAG RETRIEVAL EVAL")
    print("=" * 56)
    print(f"  k                       {summary['k']}")
    print(f"  queries                 {summary['queries']}")
    print(f"  mean Precision@k        {summary['mean_precision_at_k']:.3f}")
    print(f"  mean Hit@k              {summary['mean_hit_at_k']:.3f}")
    print(f"  mean Hit@1              {summary['mean_hit_at_1']:.3f}")
    print(f"  wall seconds            {elapsed:.2f}")
    print()
    print("Per-query (showing misses first):")
    misses = [r for r in report["per_query"] if r["hit_at_k"] < 1.0]
    hits = [r for r in report["per_query"] if r["hit_at_k"] >= 1.0]
    for row in misses + hits:
        marker = "MISS" if row["hit_at_k"] < 1.0 else "  ok"
        print(
            f"  [{marker}] expected={row['expected_doc_id']:24s} "
            f"top1={row['top_k_doc_ids'][0] if row['top_k_doc_ids'] else 'EMPTY':24s} "
            f"P@k={row['precision_at_k']:.2f}  query={row['query'][:60]!r}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": args.backend,
        "k": args.k,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "wall_seconds": round(elapsed, 2),
        **report,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
