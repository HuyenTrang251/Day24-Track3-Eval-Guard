from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from numpy import dot
    from numpy.linalg import norm
    
    metadata = metadata or {}
    
    # 1. Tách văn bản thành các câu dựa trên dấu câu hoặc xuống dòng kép
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
        
    # 2. Khởi tạo mô hình embedding và encode các câu
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)
    
    chunks = []
    current_group = [sentences[0]]
    
    # 3. Duyệt qua các câu và tính cosine similarity giữa câu sau và câu trước
    for i in range(1, len(sentences)):
        sim = 0.0
        norm_a = norm(embeddings[i-1])
        norm_b = norm(embeddings[i])
        if norm_a > 0 and norm_b > 0:
            sim = dot(embeddings[i-1], embeddings[i]) / (norm_a * norm_b + 1e-9)
            
        # Nếu độ tương đồng thấp hơn threshold -> Tách sang chunk mới
        if sim < threshold:
            joined_text = " ".join(current_group)
            chunks.append(Chunk(
                text=joined_text,
                metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)}
            ))
            current_group = [sentences[i]]
        else:
            current_group.append(sentences[i])
            
    # Add group cuối cùng
    if current_group:
        joined_text = " ".join(current_group)
        chunks.append(Chunk(
            text=joined_text,
            metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)}
        ))
        
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    """
    metadata = metadata or {}
    
    # 1. Tách văn bản thành các đoạn văn
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    parents = []
    children = []
    
    prefix = metadata.get("source", "doc").replace(".", "_").replace(" ", "_")
    
    # 2. Gom paragraphs thành các parent chunks
    current_parent = ""
    for para in paragraphs:
        if len(current_parent) + len(para) > parent_size and current_parent:
            pid = f"{prefix}_p{len(parents)}"
            parents.append(Chunk(
                text=current_parent.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
            ))
            current_parent = ""
        current_parent += para + "\n\n"
        
    if current_parent.strip():
        pid = f"{prefix}_p{len(parents)}"
        parents.append(Chunk(
            text=current_parent.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
        ))
        
    # 3. Với mỗi parent chunk, cắt nhỏ ra thành các child chunks
    for parent in parents:
        pid = parent.metadata["parent_id"]
        parent_text = parent.text
        
        # Chia nhỏ parent_text thành các child chunks (cắt theo paragraph nhỏ hơn)
        parent_paras = [p.strip() for p in parent_text.split("\n\n") if p.strip()]
        current_child = ""
        
        for para in parent_paras:
            # Nếu para quá lớn so với child_size, ta chia theo câu để không bị cắt ngang câu/chữ
            if len(para) > child_size:
                if current_child:
                    children.append(Chunk(
                        text=current_child.strip(),
                        metadata={**metadata, "chunk_type": "child", "child_index": len(children)},
                        parent_id=pid
                    ))
                    current_child = ""
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip()]
                sent_accum = ""
                for s in sents:
                    if len(sent_accum) + len(s) > child_size and sent_accum:
                        children.append(Chunk(
                            text=sent_accum.strip(),
                            metadata={**metadata, "chunk_type": "child", "child_index": len(children)},
                            parent_id=pid
                        ))
                        sent_accum = ""
                    sent_accum += s + " "
                if sent_accum.strip():
                    children.append(Chunk(
                        text=sent_accum.strip(),
                        metadata={**metadata, "chunk_type": "child", "child_index": len(children)},
                        parent_id=pid
                    ))
            elif len(current_child) + len(para) > child_size and current_child:
                children.append(Chunk(
                    text=current_child.strip(),
                    metadata={**metadata, "chunk_type": "child", "child_index": len(children)},
                    parent_id=pid
                ))
                current_child = para + "\n\n"
            else:
                current_child += para + "\n\n"
                
        if current_child.strip():
            children.append(Chunk(
                text=current_child.strip(),
                metadata={**metadata, "chunk_type": "child", "child_index": len(children)},
                parent_id=pid
            ))
            
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo cấu trúc phần (header + content).
    """
    metadata = metadata or {}
    
    # 1. Tách văn bản theo các dòng chứa tiêu đề Markdown (# , ## , ### )
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    
    chunks = []
    current_header = "Intro"
    
    for item in sections:
        item_strip = item.strip()
        if not item_strip:
            continue
            
        # Nếu dòng đó match với header
        if re.match(r'^#{1,3}\s+', item_strip):
            current_header = item_strip
        else:
            # Đây là content của header trước đó
            # Gộp header hiện tại vào đầu content để làm giàu ngữ cảnh
            chunk_text = f"{current_header}\n\n{item_strip}"
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "section": current_header, "strategy": "structure", "chunk_index": len(chunks)}
            ))
            
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
