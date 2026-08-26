"""Local RAG latency benchmark without Docker or enrichment calls."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.pipeline import run_query


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return round(ordered[min(int(len(ordered) * fraction), len(ordered) - 1)], 2)


def main() -> None:
    documents = load_documents()
    chunks = []
    for document in documents:
        _, children = chunk_hierarchical(document["text"], metadata=document["metadata"])
        chunks.extend({"text": child.text, "metadata": child.metadata} for child in children)

    search = HybridSearch()
    search.index(chunks)
    reranker = CrossEncoderReranker()
    query = "Nhân viên được nghỉ phép năm bao nhiêu ngày?"

    run_query(query, search, reranker)  # Warm up models and API connection.
    times = []
    for _ in range(5):
        started = time.perf_counter()
        run_query(query, search, reranker)
        times.append((time.perf_counter() - started) * 1000)

    print({"p50": percentile(times, 0.50), "p95": percentile(times, 0.95),
           "p99": percentile(times, 0.99), "samples_ms": [round(value, 2) for value in times]})


if __name__ == "__main__":
    main()
