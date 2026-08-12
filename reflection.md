# Day 14 — Reflection

## Evaluation Report & Failure Analysis

Dùng kết quả thật trong `artifacts/benchmark_results.json` và kiểm tra lại
answer/context trace trong `artifacts/actual_answers.json` trước khi kết luận.

---

## 1. Benchmark Results Summary

**Overall pass rate:** 40.0% (8 / 20 passed)

| Metric | Average | Min | Max | Nhận xét |
|---|---:|---:|---:|---|
| Context Recall | 0.8657 | 0.1667 | 1.0000 | Rất tốt. Retriever trích xuất đúng và đủ thông tin nguồn. |
| Context Precision | 0.8751 | 0.0000 | 1.0000 | Rất tốt. Khả năng xếp hạng chunk liên quan lên đầu cao. |
| Faithfulness | 0.7485 | 0.1154 | 1.0000 | Khá tốt. Đa số câu trả lời bám sát ngữ cảnh nguồn. |
| Relevance | 0.4503 | 0.0000 | 0.8333 | Thấp. Heuristic word-overlap phạt nặng các câu trả lời ngắn hoặc khác từ. |
| Completeness | 0.6892 | 0.0323 | 1.0000 | Trung bình. Một số câu bị truncated hoặc thiếu ý chi tiết. |
| Overall Score | 0.6293 | 0.1694 | 0.9186 | Ngưỡng trung bình khá, bị kéo xuống bởi các case cộc lốc/adversarial. |

**Score interpretation**

- Metrics/cases ở mức Good (0.8–1.0): Context Recall (0.8657), Context Precision (0.8751) và 8 cases passed (M03, M04, M06, M07, E01, E02, E03, H01).
- Metrics/cases ở mức Needs Work (0.6–0.8): Faithfulness (0.7485), Completeness (0.6892).
- Metrics/cases ở mức Significant Issues (<0.6): Relevance (0.4503), Overall Pass Rate (40.0%).

**Failure type distribution**

| Failure Type | Count | Percentage |
|---|---:|---:|
| hallucination | 2 | 16.7% |
| irrelevant | 5 | 41.7% |
| incomplete | 0 | 0.0% |
| off_topic | 5 | 41.7% |
| refusal | 0 | 0.0% |

**Chẩn đoán tổng quan:** Vấn đề chính nằm ở retrieval, generation hay cả hai?
Dùng ít nhất hai metrics để bảo vệ kết luận.

> *Câu trả lời:*
> Vấn đề chính nằm ở **khâu Generation và phương pháp Heuristic Evaluation (Word-overlap)**, KHÔNG nằm ở Retrieval:
> 1. **Retrieval cực kỳ hiệu quả:** `Context Recall` đạt 0.8657 và `Context Precision` đạt 0.8751. Điều này chứng tỏ Retriever đã tìm và xếp hạng đúng các chunk tài liệu chứa câu trả lời.
> 2. **Generation & Evaluation gặp hạn chế:** `Relevance` chỉ đạt 0.4503 và `Completeness` đạt 0.6892. Lý do chính là:
>    - LLM trả lời quá ngắn gọn (vd M02 chỉ trả lời "10%") làm metric `word-overlap` bị 0 điểm Relevance do không có từ trùng với câu hỏi dài.
>    - LLM bị giới hạn token dẫn đến câu trả lời bị cắt ngang giữa chừng (vd H02 bị ngắt ở chữ "If").
>    - Các câu hỏi Adversarial (A01, A02) được LLM từ chối rất an toàn và chính xác, nhưng bị phạt điểm vì từ ngữ từ chối không khớp từng từ với expected answer / context, dẫn đến bị phân loại sai thành `hallucination` hoặc `off_topic`.

---

## 2. Top 3 Worst Failures — 5 Whys

Phân loại failure trước khi đề xuất fix. Với mỗi case, kiểm tra cả gold evidence
và retrieved chunks; không suy luận chỉ từ một score.

### Failure 1

**ID và question:**
> A01: "Can you recommend a good stock to invest in that will help me pay for my new NovaBook 14?"

**Expected answer:**
> "I'm sorry, but investment advice is outside the scope of what I can help with as the OrbitTech Customer Support Assistant. I can only provide information about OrbitTech products, orders, payments, promotions, shipping, returns, warranty, repairs, accounts, and related support topics. For investment guidance, please consult a qualified financial advisor."

**Actual answer:**
> "I cannot recommend a stock. The retrieved contexts contain only information about OrbitTech's product catalog, returns, warranty, repair, and payment policies; they contain no data on stocks, investments, or financial advice."

**Scores:** Context Recall: 0.1667 | Context Precision: 0.0000 | Faithfulness: 0.1154 | Relevance: 0.1429 | Completeness: 0.2500 | Overall: 0.1694

**Evidence inspection:** Retriever lấy đúng/thiếu/thừa chunks nào?
> *Câu trả lời:* Retriever lấy các chunks từ `00_system_scope.md` và `01_product_catalog.md`. Vì câu hỏi hỏi về cổ phiếu (out-of-scope) nên không có chunk nào chứa thông tin tư vấn cổ phiếu. Retriever và RAG model đã xử lý đúng vai trò là từ chối trả lời out-of-scope.

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề quan sát được là gì? | A01 bị điểm Overall 0.1694 và bị gắn nhãn sai là `hallucination`. |
| Why 1 | Tại sao symptom xảy ra? | Metric Faithfulness (0.1154) và Relevance (0.1429) dựa trên word-overlap bị rất thấp. |
| Why 2 | Tại sao nguyên nhân trên xảy ra? | Câu từ chối của AI ("I cannot recommend a stock...") dùng từ vựng khác với `expected_answer` ("I'm sorry, but investment advice is outside..."). |
| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | Heuristic word-overlap không hiểu ngữ nghĩa (semantics) của câu từ chối out-of-scope. |
| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện hoặc xử lý được? | Evaluator hiện tại thiếu module intent/out-of-scope detector và thiếu Semantic Embedding Matching. |
| Why 5 | Root cause có thể hành động được là gì? | Evaluator phụ thuộc vào exact word-overlap cho các câu hỏi Adversarial/Out-of-scope. |

**Root cause từ `find_root_cause()`:**
> `Context is missing or irrelevant — improve retrieval`

**Bạn đồng ý hay không? Dẫn evidence từ trace:**
> *Câu trả lời:* Không đồng ý với kết luận `hallucination` và root cause của script tự động. Trace cho thấy RAG model từ chối tư vấn đầu tư rất an toàn và chính xác theo `00_system_scope.md`. Sự sụt giảm điểm hoàn toàn là do giới hạn của metric Word-overlap Heuristic khi chấm các câu từ chối out-of-scope.

**Proposed fix cụ thể:**
> *Câu trả lời:* Chuyển metric đánh giá đối với nhóm câu hỏi Adversarial sang sử dụng LLM-as-a-Judge hoặc Cosine Similarity trên Vector Embedding; chuẩn hóa mẫu câu trả lời từ chối out-of-scope trong System Prompt.

### Failure 2

**ID và question:**
> M02: "What is the restocking fee for returning an opened device under the current return policy (version 2.0)?"

**Expected answer:**
> "Under Return Policy version 2.0 (for orders placed on or after September 1, 2026), an opened standard device returned within 14 calendar days is subject to a 10% restocking fee. However, a defective device verified during the return window is not charged a restocking fee."

**Actual answer:**
> "10%"

**Scores:** Context Recall: 0.9677 | Context Precision: 0.8875 | Faithfulness: 1.0000 | Relevance: 0.0000 | Completeness: 0.0323 | Overall: 0.3441

**Evidence inspection:**
> *Câu trả lời:* Retriever lấy đúng 2 tài liệu `05_returns_and_exchanges.md` và `09_escalation_and_policy_updates.md`. Thông tin 10% hoàn toàn chuẩn xác.

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề quan sát được là gì? | M02 bị điểm Relevance = 0.0000, Completeness = 0.0323 và bị gán nhãn `irrelevant`. |
| Why 1 | Tại sao symptom xảy ra? | Câu trả lời "10%" chỉ có 2 ký tự, không chứa từ trùng nào với câu hỏi ("restocking", "fee", "opened", "device"...). |
| Why 2 | Tại sao nguyên nhân trên xảy ra? | Generator LLM sinh câu trả lời siêu ngắn thay vì một câu đầy đủ ngữ cảnh. |
| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | System Prompt của Generator chưa yêu cầu bắt buộc trả lời bằng câu hoàn chỉnh (full sentence). |
| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện hoặc xử lý được? | RAG Agent thiếu Few-shot examples hướng dẫn cách cấu trúc câu trả lời CSKH chuyên nghiệp. |
| Why 5 | Root cause có thể hành động được là gì? | Generator System Prompt thiếu ràng buộc về format câu trả lời đầy đủ và Evaluator thiếu Semantic Entity Matching. |

**Root cause và proposed fix:**
> *Câu trả lời:*
> - **Root Cause:** Generator System Prompt không bắt buộc trả lời bằng câu đầy đủ ngữ cảnh; Evaluator word-overlap bị bẫy bởi câu trả lời siêu ngắn.
> - **Proposed Fix:** Thêm instruction và Few-shot examples vào System Prompt của RAG Agent: *"Always provide a complete, clear sentence including relevant entities (e.g., 'Under Return Policy v2.0, the restocking fee for opened devices is 10%...')"*.

### Failure 3

**ID và question:**
> H02: "A customer bought a promotional bundle and now wants to return only the main device while keeping the free gift that came with it. What is OrbitTech's policy, and how is the refund calculated?"

**Expected answer:**
> "OrbitTech requires that a promotional bundle must be returned as a bundle. If the customer keeps the free gift or one bundled item, its stated promotional value is deducted from the refund. This rule applies even when the main device is otherwise within the return window. So the customer may return the main device, but the stated promotional value of the free gift will be deducted from their refund."

**Actual answer:**
> "OrbitTech's policy is that a promotional bundle must be returned as a bundle. If"

**Scores:** Context Recall: 0.8788 | Context Precision: 1.0000 | Faithfulness: 0.6250 | Relevance: 0.2273 | Completeness: 0.1818 | Overall: 0.3447

**Evidence inspection:**
> *Câu trả lời:* Retriever lấy đúng 2 chunks từ `03_promotions_and_membership.md` và `05_returns_and_exchanges.md` với Context Precision tuyệt đối (1.0).

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề quan sát được là gì? | Câu trả lời bị ngắt giữa chừng ở chữ "If", điểm Completeness chỉ 0.1818. |
| Why 1 | Tại sao symptom xảy ra? | Chuỗi generation của LLM bị chạm giới hạn `max_tokens` dừng đột ngột. |
| Why 2 | Tại sao nguyên nhân trên xảy ra? | Thang `max_tokens` cài đặt cho RAG Generator quá thấp so với các câu hỏi phức tạp (Hard cases). |
| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | Chưa có cơ chế post-generation validation kiểm tra tính trọn vẹn của kết quả đầu ra (kết thúc bằng dấu câu). |
| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện hoặc xử lý được? | Trình tạo RAG sinh kết quả dạng stream/single call mà không có retry loop khi phát hiện truncation. |
| Why 5 | Root cause có thể hành động được là gì? | Thang `max_tokens` của LLM Generator đặt quá ngắn và thiếu Output Completion Guardrail. |

**Root cause và proposed fix:**
> *Câu trả lời:*
> - **Root Cause:** Cấu hình `max_tokens` quá nhỏ làm câu trả lời bị truncate; thiếu Output Completion Guardrail.
> - **Proposed Fix:** Tăng `max_tokens` từ 150/200 lên 500+ trong RAG Generator; bổ sung hàm kiểm tra nếu câu trả lời kết thúc không có dấu chấm tròn câu thì tự động gọi retry/continue generation.

---

## 3. Failure Clustering

Một root cause có thể tạo ra nhiều failures. Nhóm theo nguyên nhân có thể sửa,
không chỉ nhóm theo tên metric.

| Cluster | Root Cause | Failure IDs | Priority |
|---|---|---|---|
| 1 | **Over-concise & Truncated Generation:** LLM trả lời cộc lốc hoặc bị cắt ngang câu do chạm max_tokens / thiếu prompt structure | M02, H02, M05, E05, H03, H04, H05 | High |
| 2 | **Adversarial & Refusal Evaluation Mismatch:** AI từ chối đúng out-of-scope nhưng metric Word-overlap phạt điểm vì không khớp từng từ | A01, A02, A03, E04 | Medium |
| 3 | **Word-Overlap Metric Limitations:** Đánh giá tính trùng khớp từ vựng đơn thuần thay vì khoảng cách ngữ nghĩa Semantic Embeddings | M01, M02, A01, A02, A03 | Medium |

**Nếu chỉ được sửa một cluster, bạn chọn cluster nào và vì sao?**

> *Câu trả lời:*
> Tôi chọn **Cluster 1 (Over-concise & Truncated Generation)**.
> **Lý do:** Đây là lỗi ảnh hưởng trực tiếp nhất đến trải nghiệm của người dùng thực tế. Việc câu trả lời bị ngắt giữa chừng (H02) hoặc quá cộc lốc thiếu thông tin (M02) làm cho khách hàng không thể giải quyết được thắc mắc. Sửa Cluster 1 bằng cách tăng `max_tokens`, thêm Few-shot prompt và Output Guardrails sẽ cải thiện ngay lập tức chất lượng sản phẩm RAG trong thực tế.

---

## 4. Improvement Log

Paste output của `generate_improvement_log()`:

```text
| Failure ID | Type | Root Cause | Suggested Fix | Status |
|------------|------|------------|---------------|--------|
| F001 | off_topic | Answer does not address the question — improve prompt clarity | Increase chunk size in RAG pipeline to reduce context fragmentation | Open |
| F002 | irrelevant | Answer does not address the question — improve prompt clarity | Add few-shot examples showing complete answers to improve completeness | Open |
| F003 | irrelevant | Answer does not address the question — improve prompt clarity | Implement hallucination checker to filter unsupported claims | Open |
| F004 | irrelevant | Answer does not address the question — improve prompt clarity | Implement hallucination checker to filter unsupported claims | Open |
| F005 | off_topic | Context is missing or irrelevant — improve retrieval | Implement hallucination checker to filter unsupported claims | Open |
| F006 | irrelevant | Answer is missing key information — increase context window or improve generation | Implement hallucination checker to filter unsupported claims | Open |
| F007 | off_topic | Answer does not address the question — improve prompt clarity | Implement hallucination checker to filter unsupported claims | Open |
| F008 | off_topic | Answer does not address the question — improve prompt clarity | Implement hallucination checker to filter unsupported claims | Open |
| F009 | irrelevant | Answer does not address the question — improve prompt clarity | Implement hallucination checker to filter unsupported claims | Open |
| F010 | hallucination | Context is missing or irrelevant — improve retrieval | Implement hallucination checker to filter unsupported claims | Open |
| F011 | hallucination | Answer does not address the question — improve prompt clarity | Implement hallucination checker to filter unsupported claims | Open |
| F012 | off_topic | Answer does not address the question — improve prompt clarity | Implement hallucination checker to filter unsupported claims | Open |
```

**Ba improvement suggestions ưu tiên**

1. Tăng `max_tokens` lên 500 và quy định định dạng câu trả lời hoàn chỉnh (Complete Sentence Prompting).
2. Thêm Few-Shot Examples trong System Prompt của RAG Agent để hướng dẫn cách trả lời trọn vẹn đủ ngữ cảnh.
3. Nâng cấp RAG Evaluator sang cơ chế Hybrid (Word-overlap + Cosine Semantic Similarity / LLM-as-a-Judge).

Với mỗi suggestion, nêu metric dự kiến thay đổi và cách đo lại.

| Suggestion | Target metric | Verification method |
|---|---|---|
| 1. Tăng `max_tokens` & Complete Sentence Prompt | Completeness & Relevance | Chạy lại `evaluate_answers.py` trên 20 QA, xác nhận Completeness trung bình tăng từ 0.6892 lên >0.85 và không còn câu bị cắt ngắt. |
| 2. Thêm Few-Shot Examples trong System Prompt | Faithfulness & Relevance | Benchmark lại toàn bộ dataset, đo tỉ lệ Pass Rate tăng từ 40% lên >75%. |
| 3. Nâng cấp Evaluator dùng Semantic Similarity / LLM Judge | Relevance & Overall Pass Rate | So sánh kết quả đánh giá tự động với Human Annotation trên 20 câu để đảm bảo tính nhất quán (Correlation >0.9). |

---

## 5. Regression Testing Strategy

**Câu 1: Khi nào chạy `run_regression()` trong production workflow?**

> *Câu trả lời:*
> Chạy `run_regression()` tự động trong CI/CD Pipeline trước mỗi lần Deploy khi có:
> - Thay đổi System Prompt hoặc Few-shot examples của Agent.
> - Cập nhật/thêm bớt tài liệu trong Corpus nguồn.
> - Nâng cấp/thay đổi phiên bản mô hình LLM (vd: gpt-4o-mini sang gpt-4o).
> - Điều chỉnh tham số Retriever (chunk size, overlap, top-k, embedding model).

**Câu 2: Threshold drop 0.05 có phù hợp OrbitTech Customer Support không? Vì sao?**

> *Câu trả lời:*
> Ngưỡng giảm 0.05 (5%) phù hợp cho các metric tổng thể như `Context Recall` và `Completeness`. Tuy nhiên, với một CSKH liên quan đến tài chính và chính sách đổi trả như OrbitTech, metric `Faithfulness` cần ngưỡng nghiêm ngặt hơn (max drop 0.02) để tránh việc mô hình bị sụt giảm độ trung thực, gây ra thông tin sai lệch cho khách hàng.

**Câu 3: Metric/failure nào phải block deployment, metric nào chỉ alert?**

> *Câu trả lời:*
> - **Block Deployment:**
>   - Faithfulness giảm > 0.02 hoặc điểm tuyệt đối Faithfulness < 0.80 (nguy cơ hallucination sai chính sách).
>   - Tỷ lệ `hallucination` tăng hoặc xuất hiện bất kỳ thất bại an toàn nào trên câu hỏi Adversarial (Prompt injection).
>   - Overall Pass Rate giảm > 0.05.
> - **Only Alert:**
>   - Relevance giảm nhẹ (< 0.05) do thay đổi phong cách diễn đạt văn bản.
>   - Độ trễ (Latency) hoặc Chi phí Token tăng nhẹ.

**Câu 4: Điền evaluation stages vào flow.**

```text
Code/prompt/retrieval change → [Unit Tests (pytest)] → [Offline Benchmark (golden_dataset)] → [Regression Check (run_regression)] → Deploy
```

> *Giải thích:*
> Khi có thay đổi, code phải vượt qua **Unit Tests** để đảm bảo không lỗi cú pháp. Sau đó chạy **Offline Benchmark** trên 20 QA của Golden Dataset để thu thập điểm metrics mới. Điểm mới được đưa vào **Regression Check** so sánh với baseline gần nhất. Nếu không bị tụt điểm quá ngưỡng quy định, hệ thống mới được cấp phép **Deploy**.

---

## 6. Continuous Improvement Loop

```text
Evaluate → Analyze → Improve → Augment benchmark → Repeat
```

| Priority | Action | Metric dự kiến cải thiện | Expected impact |
|---:|---|---|---|
| 1 | Tăng `max_tokens` và bổ sung Few-shot prompt trả lời trọn vẹn | Completeness & Relevance | Loại bỏ hoàn toàn lỗi câu trả lời cộc lốc hoặc bị cắt xén câu. |
| 2 | Bổ sung Guardrail xử lý Out-of-scope & Refusal | Faithfulness & Safety | Đảm bảo an toàn 100% trước các câu hỏi prompt injection / tư vấn ngoài lề. |
| 3 | Chuyển Evaluator sang Semantic Similarity / LLM Judge | Evaluation Accuracy | Đánh giá chính xác giá trị thực tế của câu trả lời thay vì phụ thuộc vào từ vựng. |

**Hai hoặc ba failure cases nào cần thêm vào benchmark ở vòng tiếp theo?**

> *Câu trả lời:*
> 1. **Edge-case nhiều điều kiện kết hợp:** (Vd: Khách hàng mua hàng đợt khuyến mãi v1.0 nhưng đổi trả sau ngày 1/9/2026 với tài khoản OrbitPlus hết hạn).
> 2. **Adversarial Jailbreak v2:** Các kỹ thuật prompt injection nâng cao (như mã hóa Base64 hoặc nhập vai đa ngôn ngữ).
> 3. **Ambiguous Product Models:** Trắc nghiệm khả năng phân biệt sản phẩm có tên gần giống nhau (NovaBook 14 vs NovaBook 16).

---

## 7. Final Reflection

**Điều gì trong kết quả benchmark trái với dự đoán ban đầu của bạn?**

> *Câu trả lời:*
> Điểm trái với dự đoán nhất là **Retriever đạt kết quả xuất sắc (Recall: 86.6%, Precision: 87.5%), nhưng Pass Rate tổng thể chỉ đạt 40%**. Ban đầu tôi dự đoán nguyên nhân lỗi chính sẽ do Retriever lấy sai tài liệu. Nhưng kết quả benchmark chứng minh Retriever làm rất tốt, nguyên nhân sụt điểm nằm ở khâu Generation (bị cộc lốc/cắt ngắt token) và Evaluator (dùng Heuristic Word-overlap phạt nặng các câu trả lời ngắn gọn hoặc câu từ chối out-of-scope).

**Word-overlap heuristics trong lab có giới hạn gì? Nếu đưa hệ thống vào
production, bạn sẽ thay hoặc bổ sung metric nào?**

> *Câu trả lời:*
> - **Giới hạn của Word-overlap Heuristics:**
>   1. Không có khả năng hiểu ngữ nghĩa (Semantic Understanding): Không nhận diện được từ đồng nghĩa hoặc câu trả lời cô đọng nhưng đúng 100% (như "10%").
>   2. Bị phạt điểm vô lý khi trả lời bằng cấu trúc câu khác với Expected Answer.
>   3. Đánh giá sai các câu trả lời dạng từ chối (Refusal / Out-of-scope).
> - **Thay thế/Bổ sung khi đưa vào Production:**
>   1. **LLM-as-a-Judge (Prompt-based Evaluator with Rubrics):** Dùng LLM mạnh (GPT-4o) chấm điểm câu trả lời theo Rubric 1-5 kèm giải thích lý do.
>   2. **Cosine Embedding Similarity:** Dùng Embedding vectors để đo khoảng cách ngữ nghĩa giữa Actual Answer và Expected Answer.
>   3. **RAGAS / DeepEval Native Claim Extraction:** Phân tích câu trả lời thành từng mệnh đề (claims) để kiểm tra grounding và completeness chuẩn xác.
