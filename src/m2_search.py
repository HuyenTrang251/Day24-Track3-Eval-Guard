from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return f"{segmented} {text}"
    except Exception as e:
        print(f"  ⚠️  underthesea segment failed: {e}")
        return text  # fallback


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi
        self.documents = chunks
        self.corpus_tokens = []
        
        for chunk in chunks:
            # Tách từ tiếng Việt và split thành các token độc lập
            tokenized = segment_vietnamese(chunk["text"]).lower().split()
            self.corpus_tokens.append(tokenized)
            
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []
            
        tokenized_query = segment_vietnamese(query).lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sắp xếp và lấy top indices có score > 0
        indexed_scores = [(i, score) for i, score in enumerate(scores) if score > 0]
        sorted_indices = sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for idx, score in sorted_indices:
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc["text"],
                score=float(score),
                metadata=doc.get("metadata", {}),
                method="bm25"
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        # Thay thế kết nối host/port bằng location=":memory:" để chạy offline không cần Docker
        self.client = QdrantClient(location=":memory:")
        self._encoder = None


    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        # 1. Tạo mới collection
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        
        if not chunks:
            return
            
        # 2. Encode tất cả các chunks
        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        
        # 3. Tạo PointStruct để upsert vào Qdrant
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={**chunk.get("metadata", {}), "text": chunk["text"]}
            ))
            
        # 4. Lưu vào cơ sở dữ liệu Qdrant In-Memory
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not self.client.collection_exists(collection_name=collection):
            return []
            
        query_vector = self._get_encoder().encode(query).tolist()
        
        # Qdrant client >= 2.0 sử dụng query_points()
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k
        )
        
        results = []
        for pt in response.points:
            results.append(SearchResult(
                text=pt.payload["text"],
                score=pt.score,
                metadata={k: v for k, v in pt.payload.items() if k != "text"},
                method="dense"
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}  # text -> {"score": float, "result": SearchResult}
    
    for results in results_list:
        for rank, result in enumerate(results):
            text = result.text
            if text not in rrf_scores:
                rrf_scores[text] = {"score": 0.0, "result": result}
            # Cộng dồn điểm RRF dựa vào thứ hạng (rank bắt đầu từ 0)
            rrf_scores[text]["score"] += 1.0 / (k + rank + 1)
            
    # Sắp xếp giảm dần theo điểm RRF
    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    
    merged_results = []
    for item in sorted_docs:
        orig_result = item["result"]
        merged_results.append(SearchResult(
            text=orig_result.text,
            score=item["score"],
            metadata=orig_result.metadata,
            method="hybrid"
        ))
    return merged_results


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
