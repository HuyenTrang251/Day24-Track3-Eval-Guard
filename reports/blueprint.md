# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Thị Huyền Trang
**Ngày:** 26/08/2026

---

## Guard Stack Architecture

```text
User Input
    |
    v
[Presidio PII Scan]
    | block CCCD, CMND, số điện thoại Việt Nam, email
    v
[NeMo Input Rail / local input policy]
    | block jailbreak, prompt injection, yêu cầu PII và off-topic
    v
[RAG Pipeline Day 18]
    | M1 Chunk -> M2 Search -> M3 Rerank -> GPT-4o-mini
    v
[Output Rail]
    | redact PII trong câu trả lời trước khi trả về user
    v
User Response
```

---

## Latency Budget

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---:|---:|---:|---|
| Presidio PII | 11.87 | 15.50 | 15.50 | <10ms |
| NeMo Input Rail | 0.03 | 0.11 | 0.11 | <300ms |
| RAG Pipeline | 7145.27 | 7484.33 | 7484.33 | <2000ms |
| NeMo Output Rail | 10.13 | 11.82 | 11.82 | <300ms |
| **Total Guard** | **11.91** | **15.53** | **15.53** | **<500ms** |

**Budget OK?** Có. P95 của guard stack là 15.53ms, thấp hơn budget 500ms.

**Comment:** Output rail được đo sau một lượt warm-up trên local PII/output guard. RAG pipeline được benchmark trên 5 query sau warm-up, gồm local embedding/search/rerank và OpenAI answer generation. P95 RAG là 7484.33ms, vượt budget 2000ms; cần streaming, caching, smaller models hoặc tách async evaluation trước khi production. Khi chạy NeMo bằng remote LLM, cần đo lại riêng network và LLM latency vì chúng có thể trở thành bottleneck.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
- name: Unit and Guard Tests
  run: python -m pytest tests -q

- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: python -m pytest tests/test_phase_c.py -k adversarial_suite
  # Phải chặn đúng ít nhất 75% payload adversarial.

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency"
  # P95 guard stack phải nhỏ hơn 500ms.
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness | < 0.70 | Review prompt, citations và retrieval context |
| Adversarial block rate | < 80% | Bổ sung attack pattern và regression test |
| Guard P95 latency | > 600ms | Profile Presidio, NeMo và network calls |
| PII detected count | spike >10/hour | Security alert và kiểm tra nguồn request |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.807 |
| Worst metric | faithfulness |
| Dominant failure distribution | factual và multi_hop đồng hạng về số failure; multi_hop yếu faithfulness rõ nhất |
| Cohen's kappa | 1.000 trên 10 human-labelled cases |
| Adversarial pass rate | 19 / 20 (95%) |
| Guard P95 latency | 15.53 ms |

---

## Nhận xét và cải tiến

Factual questions có điểm tốt nhất, nhưng các câu multi-hop thường mất faithfulness khi mô hình phải kết hợp nhiều policy. Context precision đều cao, vì vậy ưu tiên là ràng buộc câu trả lời theo evidence thay vì chỉ thêm nhiều context. Có thể thêm citation-aware prompting, hạ temperature và yêu cầu model nêu rõ khi context thiếu thông tin. Với adversarial input, nên mở rộng pattern tiếng Việt cho giả mạo quyền hạn và yêu cầu tiết lộ dữ liệu. Trong production, cần chạy định kỳ regression suite, theo dõi P95 và dùng kết quả judge đã được calibrate với human labels trước khi biến nó thành release gate.
