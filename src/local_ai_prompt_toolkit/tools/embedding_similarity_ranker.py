"""Offline vector cosine similarity ranker and semantic retrieval scoring utility."""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute exact cosine similarity between two numeric vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return round(dot_product / (norm_a * norm_b), 6)


def _bag_of_words_vectorize(query: str, corpus: List[str]) -> Tuple[List[float], List[List[float]]]:
    """Lightweight TF vectorizer for query and corpus strings without heavy external dependencies."""
    vocabulary: Dict[str, int] = {}
    
    def tokenize(s: str) -> List[str]:
        return [w.lower() for w in s.split() if w.strip()]

    all_texts = [query] + corpus
    for t in all_texts:
        for word in tokenize(t):
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)

    def vectorize(s: str) -> List[float]:
        vec = [0.0] * len(vocabulary)
        tokens = tokenize(s)
        for w in tokens:
            if w in vocabulary:
                vec[vocabulary[w]] += 1.0
        # Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm if norm > 0 else 0.0 for x in vec]

    query_vec = vectorize(query)
    doc_vecs = [vectorize(c) for c in corpus]
    return query_vec, doc_vecs


def execute_embedding_similarity_ranker(
    query_vector: Optional[List[float]] = None,
    document_vectors: Optional[List[Dict[str, Any]]] = None,
    query_text: Optional[str] = None,
    document_texts: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Rank documents by cosine similarity against a query vector or query text.

    Args:
        query_vector: Numerical vector representation of query.
        document_vectors: List of dicts with 'id' and 'vector'.
        query_text: Plain text query string (auto-vectorized if vectors omitted).
        document_texts: List of dicts with 'id' and 'text'.
        top_k: Number of highest ranking results to return.
        output_path: Optional destination to write JSON results.

    Returns:
        Structured dictionary listing ranked document IDs and similarity scores.
    """
    ranked_results: List[Dict[str, Any]] = []

    if query_vector is not None and document_vectors is not None:
        for doc in document_vectors:
            doc_id = doc.get("id", "doc")
            vec = doc.get("vector", [])
            sim = _cosine_similarity(query_vector, vec)
            ranked_results.append({
                "id": doc_id,
                "similarity_score": sim,
                "metadata": doc.get("metadata", {}),
            })

    elif query_text is not None and document_texts is not None:
        corpus = [d.get("text", "") for d in document_texts]
        q_vec, d_vecs = _bag_of_words_vectorize(query_text, corpus)
        for idx, doc in enumerate(document_texts):
            sim = _cosine_similarity(q_vec, d_vecs[idx])
            ranked_results.append({
                "id": doc.get("id", f"doc_{idx + 1}"),
                "similarity_score": sim,
                "text_sample": doc.get("text", "")[:100],
            })
    else:
        raise ValueError("Must provide either (query_vector, document_vectors) or (query_text, document_texts).")

    ranked_results.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_results = ranked_results[:top_k]

    result = {
        "status": "success",
        "top_k": top_k,
        "total_documents_scored": len(ranked_results),
        "results": top_results,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, mode="w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=2)

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Offline vector cosine similarity ranker and semantic retrieval scoring utility."
    )
    parser.add_argument("--query-text", "-q", help="Query string to evaluate")
    parser.add_argument("--docs-json", "-d", help="Path to JSON file containing document list")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of top candidates to return")
    parser.add_argument("--output", "-o", help="Path to write JSON ranking report")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.docs_json and os.path.isfile(args.docs_json):
            with open(args.docs_json, "r", encoding="utf-8") as f:
                docs = json.load(f)
        else:
            docs = [
                {"id": "doc1", "text": "Python systems engineering with zero-copy socket buffers."},
                {"id": "doc2", "text": "Frontend single page application react components."},
            ]

        res = execute_embedding_similarity_ranker(
            query_text=args.query_text or "systems sockets",
            document_texts=docs,
            top_k=args.top_k,
            output_path=args.output,
        )
        print(f"Top {len(res['results'])} documents scored:")
        for r in res["results"]:
            print(f"  [{r['similarity_score']:.4f}] {r['id']}")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
