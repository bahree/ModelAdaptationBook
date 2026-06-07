"""Listing 4.4 -- Minimal retrieval-augmented generation (RAG) pipeline.

A reader-friendly RAG implementation in roughly fifty lines of core
logic.  Documents are chunked with overlap, embedded with sentence
transformers (default) or a token-overlap fallback, and stored in a
numpy array as the index.  ``answer`` retrieves the top-k chunks and
hands them to a model callable for the final answer.

The script ships with two backends:

- ``st`` (default): sentence-transformers (CPU-friendly).
- ``hash``: token-overlap fallback used by tests and CPU smoke runs.

The model callable used by ``answer`` defaults to a deterministic stub
that prints the assembled prompt; replace it with a real LLM call (HF,
OpenAI, Anthropic) for production.

Run from code/:

    # Inspect retrieval against a query (no LLM call):
    python -m chapter04.rag_minimal retrieve \
        --docs chapter04/data/policy_docs.jsonl \
        --query "How do I rotate my API key?" --k 3

    # Full retrieval + stub answer:
    python -m chapter04.rag_minimal answer \
        --docs chapter04/data/policy_docs.jsonl \
        --query "What is the API key rotation grace period?" --k 3

    # Token-overlap fallback (no sentence-transformers required):
    python -m chapter04.rag_minimal retrieve \
        --docs chapter04/data/policy_docs.jsonl \
        --query "refund policy" --k 2 --backend hash
"""
from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from common.jsonl import read_jsonl_list


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def _token_bucket(token: str, dim: int) -> int:
    """Map a token to a bucket with a stable, process-independent hash.

    Python's built-in ``hash`` is salted per process (PYTHONHASHSEED), so using
    it here would make embeddings, and therefore retrieval ranking, change from
    run to run. blake2b gives the same bucket every time.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def hash_embed(texts: Sequence[str], dim: int = 128) -> np.ndarray:
    """Token-hash fallback embedder (no models required).

    Maps tokens to indices via a stable hash and accumulates a sparse
    bag-of-words vector that is L2-normalised.  Accuracy is well below a
    real embedder, but the implementation is deterministic and fast and
    is enough to exercise the pipeline in tests.
    """
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in _tokenize(text):
            vectors[i, _token_bucket(tok, dim)] += 1.0
        norm = float(np.linalg.norm(vectors[i]))
        if norm > 0:
            vectors[i] /= norm
    return vectors


def make_st_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Sentence-transformers embedding backend (cosine on normalised vectors)."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            embedder.encode(list(texts), normalize_embeddings=True), dtype=np.float32
        )

    return embed


# ---------------------------------------------------------------------------
# Minimal RAG pipeline (Listing 4.4)
# ---------------------------------------------------------------------------


@dataclass
class MinimalRAG:
    """A 50-line RAG pipeline: chunker, embedder, numpy index, retriever."""

    embed: Callable[[Sequence[str]], np.ndarray]
    chunk_size: int = 80
    chunk_overlap: int = 20
    chunks: List[str] = field(default_factory=list)
    metadata: List[Dict[str, str]] = field(default_factory=list)
    vectors: Optional[np.ndarray] = None

    def _chunk(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        if len(words) <= self.chunk_size:
            return [" ".join(words)]
        step = max(1, self.chunk_size - self.chunk_overlap)
        out: List[str] = []
        for i in range(0, len(words), step):
            out.append(" ".join(words[i : i + self.chunk_size]))
            if i + self.chunk_size >= len(words):
                break
        return out

    def ingest(self, documents: Sequence[Dict[str, str]]) -> None:
        """Add documents to the index.

        Each document is a dict with at least a ``text`` field; ``id`` and
        ``title`` are preserved on each chunk for citation.
        """
        new_chunks: List[str] = []
        new_meta: List[Dict[str, str]] = []
        for doc in documents:
            for piece in self._chunk(doc["text"]):
                new_chunks.append(piece)
                new_meta.append({"id": doc.get("id", ""), "title": doc.get("title", "")})
        if not new_chunks:
            return
        new_vectors = self.embed(new_chunks)
        self.chunks.extend(new_chunks)
        self.metadata.extend(new_meta)
        self.vectors = (
            new_vectors if self.vectors is None else np.vstack([self.vectors, new_vectors])
        )

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, object]]:
        """Return the top-``k`` chunks ranked by cosine similarity to ``query``."""
        if self.vectors is None or len(self.chunks) == 0:
            return []
        if k <= 0:
            return []
        q_vec = self.embed([query])[0]
        scores = self.vectors @ q_vec
        order = np.argsort(-scores)[: min(k, len(self.chunks))]
        return [
            {
                "score": float(scores[idx]),
                "chunk": self.chunks[idx],
                "metadata": self.metadata[idx],
            }
            for idx in order
        ]

    def answer(
        self,
        query: str,
        llm: Callable[[str], str],
        *,
        k: int = 5,
    ) -> Dict[str, object]:
        """Retrieve top-k chunks, assemble a grounded prompt, return the answer."""
        retrieved = self.retrieve(query, k=k)
        context = "\n\n".join(item["chunk"] for item in retrieved)
        prompt = (
            "Use the context below to answer the question. "
            "If the context does not contain the answer, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        response = llm(prompt)
        return {"query": query, "prompt": prompt, "retrieved": retrieved, "answer": response}


def stub_llm(prompt: str) -> str:
    """Deterministic stub that echoes the prompt for inspection."""
    snippet = prompt.split("\n\n", 2)[1] if "\n\n" in prompt else prompt
    return f"[stub answer based on context: {snippet[:140]}...]"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


EMBED_BACKENDS = {
    "st": lambda: make_st_embedder(),
    "hash": lambda: lambda texts: hash_embed(texts, dim=256),
}


def _build(args: argparse.Namespace) -> MinimalRAG:
    docs = read_jsonl_list(args.docs)
    embed = EMBED_BACKENDS[args.backend]()
    rag = MinimalRAG(embed=embed, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    rag.ingest(docs)
    return rag


def _cmd_retrieve(args: argparse.Namespace) -> None:
    rag = _build(args)
    print(f"Indexed {len(rag.chunks)} chunks from {args.docs}")
    results = rag.retrieve(args.query, k=args.k)
    for rank, item in enumerate(results, start=1):
        print()
        print(f"#{rank}  score={item['score']:.4f}  doc={item['metadata']['id']!r}")
        snippet = item["chunk"][:240].replace("\n", " ")
        print(f"      {snippet}...")


def _cmd_answer(args: argparse.Namespace) -> None:
    rag = _build(args)
    result = rag.answer(args.query, stub_llm, k=args.k)
    print(f"Indexed {len(rag.chunks)} chunks from {args.docs}")
    print()
    print("Top retrieved chunks:")
    for rank, item in enumerate(result["retrieved"], start=1):
        print(f"  #{rank} score={item['score']:.4f}  doc={item['metadata']['id']!r}")
    print()
    print("Stub answer:")
    print(result["answer"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RAG (chapter 4)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("retrieve", "answer"):
        p = sub.add_parser(name)
        p.add_argument(
            "--docs",
            default="chapter04/data/policy_docs.jsonl",
            help="JSONL file with `id`, `title`, `text` fields",
        )
        p.add_argument("--query", required=True, help="Natural-language question")
        p.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve")
        p.add_argument(
            "--backend",
            choices=sorted(EMBED_BACKENDS),
            default="st",
            help="Embedding backend (st = sentence-transformers, hash = token bag)",
        )
        p.add_argument("--chunk_size", type=int, default=80, help="Chunk size in words")
        p.add_argument("--chunk_overlap", type=int, default=20, help="Chunk overlap in words")
    args = parser.parse_args()

    if args.cmd == "retrieve":
        _cmd_retrieve(args)
    elif args.cmd == "answer":
        _cmd_answer(args)


if __name__ == "__main__":
    main()
