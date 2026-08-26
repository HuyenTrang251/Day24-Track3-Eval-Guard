# LLM Judge Bias Report - Phase B

**Sinh viên:** Nguyễn Thị Huyền Trang
**Ngày:** 26/08/2026
**Judge model:** gpt-4o-mini

---

## 1. Pairwise Judge Results

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nghỉ khi kết hôn | B | Reference đầy đủ hơn model answer. |
| 2 | Mua thiết bị 55 triệu | B | CEO là approver đúng, không phải Director. |
| 3 | Thưởng Tết tối thiểu | B | Reference đầy đủ hơn câu trả lời ngắn. |
| 4 | Senior 9 năm thâm niên | tie | Hai lượt swap không đồng thuận. |
| 5 | Hoàn trả chi phí đào tạo | B | Reference đầy đủ hơn model answer. |

---

## 2. Swap-and-Average Results

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | B | B | B | Có |
| 2 | B | B | B | Có |
| 3 | B | B | B | Có |
| 4 | A | B | tie | Không |
| 5 | B | B | B | Có |

**Position bias rate:** 20% (= 1 case không consistent / 5 cases).

---

## 3. Cohen's kappa Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 6 label=1 và 4 label=0).  
**Judge labels:** Chấm nhị phân theo correctness của `model_answer` so với `ground_truth`; 1 = đúng, 0 = sai.

| Question ID | Human Label | Judge Label | Agree? |
|---|---:|---:|---|
| 1 | 1 | 1 | Có |
| 5 | 0 | 0 | Có |
| 12 | 1 | 1 | Có |
| 21 | 1 | 1 | Có |
| 23 | 1 | 1 | Có |
| 29 | 0 | 0 | Có |
| 33 | 1 | 1 | Có |
| 41 | 0 | 0 | Có |
| 46 | 1 | 1 | Có |
| 50 | 0 | 0 | Có |

**Cohen's kappa:** 1.000  
**Interpretation:** almost perfect agreement trên bộ 10 câu hiện tại.

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 0 / 4 cases
- B thắng + B dài hơn A: 4 / 4 cases  
- **Verbosity bias rate:** 100%

**Kết luận:** Judge chọn answer B dài hơn trong toàn bộ 4 case decisive của mẫu 5 cặp. Tuy nhiên B ở đây là reference answer thường có nhiều chi tiết policy hơn, nên chỉ số này cần được diễn giải thận trọng; không nên kết luận đây là verbosity bias thuần túy.

---

## 5. Nhận xét chung

Kappa 1.000 cho thấy binary correctness judge đồng thuận hoàn toàn với human labels trên 10 case hiện tại. Tuy nhiên, pairwise comparison với reference answer dài hơn dễ tạo lợi thế cho B, nên cần tách correctness evaluation khỏi pairwise preference khi dùng làm quality gate. Position bias 20% cho thấy swap-and-average vẫn cần thiết để phát hiện các case không ổn định. Trong production, nên mở rộng human-labelled set và review các case tie trước khi đưa kết quả judge vào quyết định release.
