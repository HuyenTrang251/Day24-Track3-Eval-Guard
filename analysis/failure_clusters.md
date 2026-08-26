# Failure Cluster Analysis - Phase A

**Sinh viên:** Nguyễn Thị Huyền Trang
**Ngày:** 26/08/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.933 | 0.517 | 0.800 |
| answer_relevancy | 0.839 | 0.667 | 0.591 |
| context_precision | 0.967 | 0.983 | 0.925 |
| context_recall | 0.900 | 0.779 | 0.650 |
| **avg_score** | **0.910** | **0.737** | **0.741** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question ID | avg_score | worst_metric |
|---:|---|---:|---:|---|
| 1 | multi_hop | 39 | 0.250 | faithfulness |
| 2 | multi_hop | 33 | 0.333 | faithfulness |
| 3 | adversarial | 44 | 0.375 | faithfulness |
| 4 | multi_hop | 30 | 0.375 | faithfulness |
| 5 | adversarial | 50 | 0.417 | faithfulness |
| 6 | factual | 7 | 0.572 | faithfulness |
| 7 | multi_hop | 21 | 0.625 | answer_relevancy |
| 8 | adversarial | 48 | 0.667 | answer_relevancy |
| 9 | multi_hop | 22 | 0.681 | context_recall |
| 10 | multi_hop | 31 | 0.683 | faithfulness |

---

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 2 | 14 | 2 | 18 |
| answer_relevancy | 13 | 2 | 2 | 17 |
| context_precision | 2 | 0 | 1 | 3 |
| context_recall | 3 | 4 | 5 | 12 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual và multi_hop cùng có 20 failure theo matrix; multi_hop là nhóm đáng ưu tiên vì 14/20 failure nằm ở faithfulness.  
**Dominant metric:** faithfulness.

**Lý do phân tích:** Các câu multi-hop cần tổng hợp nhiều quy định, ví dụ thâm niên, chức danh và policy phiên bản mới. Model có thể lấy được các đoạn context liên quan nhưng vẫn ghép chúng thành kết luận chưa được support trực tiếp. Trong khi đó, factual có nhiều answer-relevancy failure nhưng điểm trung bình vẫn cao hơn đáng kể. Vì vậy, cải thiện grounding cho multi-hop sẽ tạo tác động lớn nhất.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM ghép nhiều policy thành kết luận không có evidence trực tiếp | Thêm citations, prompt bắt buộc bám context và hạ temperature |
| context_recall | Thiếu chunk chứa điều kiện hoặc exception | Điều chỉnh chunking, hybrid retrieval và metadata filtering |
| context_precision | Retriever trả thêm context không liên quan | Tăng chất lượng reranking và filter theo policy/version |
| answer_relevancy | Câu trả lời không bám đúng ý hỏi | Cải thiện prompt template và thêm query decomposition |

---

## 6. Nhận xét về Adversarial Distribution

Adversarial có avg_score 0.741, thấp hơn factual 0.910 và gần multi-hop 0.737. Hai câu adversarial trong bottom 10 liên quan password policy và VPN cá nhân, cho thấy các version conflict hoặc quy định cấm dễ làm model trả lời sai. Nên ưu tiên metadata cho version hiện hành, đồng thời yêu cầu model nêu policy source khi câu hỏi chứa điều kiện dễ gây nhiễu.
