from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    import math
    import sys
    import types
    
    # Vá lỗi import của RAGAS với các bản LangChain mới
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        mock_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
        try:
            from langchain_google_vertexai import ChatVertexAI
            mock_vertex.ChatVertexAI = ChatVertexAI
        except ImportError:
            class ChatVertexAI:
                pass
            mock_vertex.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex
        
    # Hàm hỗ trợ dọn dẹp các giá trị NaN phát sinh khi tính toán lỗi
    def clean_nan(val):
        try:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return 0.0
            return float(val)
        except:
            return 0.0

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from langchain_openai import OpenAIEmbeddings

        # 1. Tạo Dataset từ dữ liệu đầu vào
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        
        # 2. Chạy đánh giá RAGAS (truyền thêm OpenAIEmbeddings để fix metric answer_relevancy)
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            embeddings=OpenAIEmbeddings()
        )
        
        df = result.to_pandas()
        
        # 3. Tạo danh sách EvalResult cho từng câu hỏi (tương thích cả cột mới và cũ của RAGAS)
        per_question = []
        for _, row in df.iterrows():
            q = row.get("question") if "question" in row else row.get("user_input", "")
            a = row.get("answer") if "answer" in row else row.get("response", "")
            ctx = row.get("contexts") if "contexts" in row else row.get("retrieved_contexts", [])
            gt = row.get("ground_truth") if "ground_truth" in row else row.get("reference", "")
            
            per_question.append(EvalResult(
                question=q,
                answer=a,
                contexts=ctx,
                ground_truth=gt,
                faithfulness=clean_nan(row.get("faithfulness")),
                answer_relevancy=clean_nan(row.get("answer_relevancy")),
                context_precision=clean_nan(row.get("context_precision")),
                context_recall=clean_nan(row.get("context_recall"))
            ))
            
        faithfulness_score = clean_nan(df["faithfulness"].mean() if "faithfulness" in df else 0.0)
        answer_relevancy_score = clean_nan(df["answer_relevancy"].mean() if "answer_relevancy" in df else 0.0)
        context_precision_score = clean_nan(df["context_precision"].mean() if "context_precision" in df else 0.0)
        context_recall_score = clean_nan(df["context_recall"].mean() if "context_recall" in df else 0.0)

        return {
            "faithfulness": faithfulness_score,
            "answer_relevancy": answer_relevancy_score,
            "context_precision": context_precision_score,
            "context_recall": context_recall_score,
            "per_question": per_question
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        # Trả về kết quả mặc định nếu không có OpenAI API Key hoặc lỗi kết nối
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": []
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    
    analyzed = []
    for res in eval_results:
        # Tính điểm trung bình của 4 chỉ số để đánh giá chất lượng tổng quát của câu hỏi
        avg_score = (res.faithfulness + res.answer_relevancy + res.context_precision + res.context_recall) / 4.0
        
        # Tìm metric nào tệ nhất (thấp điểm nhất) trong 4 metrics
        metrics_scores = {
            "faithfulness": res.faithfulness,
            "answer_relevancy": res.answer_relevancy,
            "context_precision": res.context_precision,
            "context_recall": res.context_recall
        }
        worst_metric = min(metrics_scores, key=metrics_scores.get)
        worst_score = metrics_scores[worst_metric]
        
        # Lấy kết quả chẩn đoán từ Diagnostic Tree
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        
        analyzed.append({
            "question": res.question,
            "avg_score": avg_score,
            "worst_metric": worst_metric,
            "score": worst_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix
        })
        
    # Sắp xếp tăng dần theo điểm trung bình (câu tệ nhất đứng trước)
    analyzed.sort(key=lambda x: x["avg_score"])
    
    # Lấy bottom_n câu hỏi tệ nhất
    results = []
    for item in analyzed[:bottom_n]:
        results.append({
            "question": item["question"],
            "worst_metric": item["worst_metric"],
            "score": item["score"],
            "diagnosis": item["diagnosis"],
            "suggested_fix": item["suggested_fix"]
        })
    return results


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
