# Báo cáo cá nhân — Day 6

**Họ và tên:** Nguyễn Đình Trường  
**MSSV:** 2A202603837  
**Lớp:** L3B  
**Ngày chạy:** 2026-11-14

## Dữ liệu và pipeline

- **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang:** OpenStax, *Principles of Data Science* (2025), Rice University; giấy phép CC BY-NC-SA 4.0 ([nguồn](https://openstax.org/details/books/principles-data-science), [giấy phép](https://creativecommons.org/licenses/by-nc-sa/4.0/)). PDF có 56 trang vật lý; corpus chọn 38 trang nội dung vật lý 19–56 (77,152 ký tự sau chuẩn hóa), không tính trang bìa/mục lục/lời nói đầu 1–18. Manifest ghi quy tắc chọn trang, số ký tự, thời điểm truy cập và SHA-256.
- **Chuẩn hóa, lỗi và giới hạn:** Python 3.13.3, pypdf 6.21.4; chuẩn hóa Unicode NFC và khoảng trắng, giữ liên kết số trang, không OCR. Trang 1–2 không trích xuất được text, được ghi trong `ingestion_errors.csv` và nằm ngoài corpus đã chọn. Corpus chỉ là phần Chương 1 của PDF hiện có, không phải toàn bộ sách.
- **Embedding model/provider/version/số chiều, vector store/version, metric, lệnh chạy và chi phí:** OpenAI `text-embedding-3-small`, 1536 chiều, 37,062 input tokens; PostgreSQL 16.16 + pgvector 0.8.7, cosine distance, exact search. Chạy từ `starter/`: `python -m src.run --validate-only` (preflight, không gọi API/ghi DB), sau đó `python -m src.run` (pipeline đầy đủ). Thời gian các giai đoạn được ghi trong [`run_metadata.json`](../../starter/results/run_metadata.json). Ước tính phí embedding chung toàn lượt chạy là **$0.00074124 (~$0.00074)** theo giá tiêu chuẩn OpenAI `text-embedding-3-small` **$0.02/1M input tokens**; công thức: 37,062 × 0.02 / 1,000,000.
- **Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn:** metadata gồm `document_id`, `source`, `page`, `chapter`, `section` và character offsets. `section` lấy từ heading số mục tường minh trong văn bản (1.1–1.5), mang theo đến heading kế tiếp; dùng để giới hạn retrieval theo mục khi câu hỏi chỉ rõ section, không suy ra từ gold answer.

## Cấu hình và benchmark

- **A / B / C (tham số cụ thể, khác biệt duy nhất B–C):** A_fixed dùng cửa sổ 500 ký tự, overlap 50 (189 chunks). B_paragraph gom đoạn tối đa 700 ký tự, đoạn dài được chia cửa sổ (129 chunks). C_paragraph_filter dùng cùng chunks/vector với B (129 chunks), chỉ thêm filter `section` nếu câu hỏi nêu mục. Model, corpus, vector store, metric và benchmark giống nhau; khác biệt B–C duy nhất là filter.
- **Benchmark và evidence:** 20 câu đã tách trước khi chạy thành 15 dev / 5 test. Có 6 câu filter, trong đó 3 câu thuộc test. Gold evidence gồm source ID, trang vật lý và trích dẫn ngắn đã đối chiếu với PDF. Test chỉ gồm một chương, các chủ đề có thể xuất hiện ở cả hai split; nguy cơ overlap cao, không đại diện thống kê cho corpus khác.
- **Tiêu chí chọn chiến lược đã chốt trước khi xem test:** ưu tiên Evidence Hit@3 trên dev, sau đó Hit@1, audit chất lượng và tài nguyên/độ phức tạp. Không dùng test để chỉnh hoặc chọn cấu hình.

## Kết quả

Hit@1 và Evidence Hit@3 được tính từ gold source/page/evidence; đây là chỉ số retrieval, không phải độ chính xác câu trả lời do LLM sinh. Thời gian và chi phí không được đo tách riêng theo strategy/split; hai cột này lặp lại tổng của cả lượt chạy ở mỗi hàng, không phải số riêng của từng cấu hình.

Các phase có thời gian đo: ingestion **8.026 s** (đã bao gồm document embedding **6.196 s** và upsert **0.940 s**), query embedding **8.437 s**, retrieval **1.932 s**. Tổng cộng **18.395 s** cho các phase có timer; đây không phải wall-clock tổng tuyệt đối vì thời gian ghi kết quả và một số overhead không được đo. Không cộng document embedding/upsert thêm vào ingestion lần nữa. Chi phí embedding ước tính **$0.00074124 (~$0.00074) cho cả run**, không phải cho mỗi strategy; tra theo [bảng giá API OpenAI](https://developers.openai.com/api/docs/pricing) ngày 2026-11-14. Giá thực tế có thể khác theo billing, ưu đãi hoặc thuế; cần đối soát dashboard.

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian (toàn run) | Chi phí (toàn run) |
|---|---|---:|---:|---:|---|---|
| A_fixed | dev (15) | 7/15 (46.7%) | 11/15 (73.3%) | 189 | 18.395 s | $0.00074 |
| A_fixed | test (5) | 1/5 (20%) | 4/5 (80%) | 189 | 18.395 s | $0.00074 |
| B_paragraph | dev (15) | 9/15 (60%) | 12/15 (80%) | 129 | 18.395 s | $0.00074 |
| B_paragraph | test (5) | 2/5 (40%) | 4/5 (80%) | 129 | 18.395 s | $0.00074 |
| C_paragraph_filter | dev (15) | 9/15 (60%) | 12/15 (80%) | 129 | 18.395 s | $0.00074 |
| C_paragraph_filter | test (5) | 2/5 (40%) | 5/5 (100%) | 129 | 18.395 s | $0.00074 |

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---:|---:|---:|---|
| A_fixed | 1.0 (10) | 15/15 | N/A | 1/0/4 (5 câu), 1.6 |
| B_paragraph | 1.0 (10) | 15/15 | N/A | 0/1/4 (5 câu), 1.8 |
| C_paragraph_filter | 1.0 (10) | 15/15 | 9/9 | 0/0/5 (5 câu), 2.0 |

**Đường dẫn trace, nhãn thủ công và cách chọn mẫu:** 60 trace top-3 ở [`traces.jsonl`](../../starter/results/traces.jsonl), tách theo strategy tại [`traces_A_fixed.jsonl`](../../starter/results/traces_A_fixed.jsonl), [`traces_B_paragraph.jsonl`](../../starter/results/traces_B_paragraph.jsonl), [`traces_C_paragraph_filter.jsonl`](../../starter/results/traces_C_paragraph_filter.jsonl). Metrics/metadata lần chạy ở [`summary.json`](../../starter/results/summary.json) và [`run_metadata.json`](../../starter/results/run_metadata.json). Audit coherence lấy 10 trang vật lý 19–28 tại `start_char=0` cho mỗi strategy (30 mẫu). Metadata và grounding kiểm tra 5 câu test × top-3 cho mỗi strategy (45 hit); grounding được chấm theo query và ghi ở rank 1, metadata kiểm tra đủ 45 hit. Trợ lý AI tạo bản nháp nhãn, học viên đã tự đối chiếu trace/chunk với PDF/manifest và xác nhận nhãn trong [`manual_coherence.csv`](../../starter/results/manual_coherence.csv) và [`manual_grounding_metadata.csv`](../../starter/results/manual_grounding_metadata.csv); đây không phải đánh giá blind/độc lập.

## Phân tích và kết luận

- **So sánh A–B và B–C; 2 ví dụ sai có ID câu và nguồn:** A→B trên dev tăng Hit@1 từ 7/15 lên 9/15, Evidence Hit@3 từ 11/15 lên 12/15; test Hit@1 tăng 1/5 lên 2/5, Evidence Hit@3 giữ 4/5. Chunk count giảm 189→129 (31.7%), nhưng audit coherence vẫn 1.0/2 ở cả hai. B→C giữ nguyên dev; test Evidence Hit@3 tăng 4/5→5/5, còn Hit@1 giữ 2/5. Với `q16`/A_fixed, top-3 trên trang vật lý 37 nêu ví dụ Iris nhưng không chứa số 150 quan sát; gold evidence là trang 37, câu “totaling 150 items” (OpenStax, mục 1.3). Với `q20`/B_paragraph, top-3 có ví dụ `iloc[]` chọn cột ở trang 54 nhưng thiếu câu trả lời chọn theo cả hàng và cột; gold evidence nằm ở trang 51 (OpenStax, mục 1.5). Vì vậy grounding q20/B được chấm 1/2.
- **Chiến lược chọn, đánh đổi, điều kiện áp dụng:** chọn B_paragraph làm mặc định dựa trên dev: bằng C trên các metric dev, tốt hơn A về retrieval và tạo ít chunks hơn. Dùng C khi câu hỏi nêu section cụ thể; filter giúp giới hạn đúng mục trong 3/3 câu test có filter, nhưng mẫu nhỏ nên chỉ là quan sát. Không chọn cấu hình dựa vào kết quả test.
- **Hạn chế của dữ liệu, nhãn, test nhỏ, đo thời gian/chi phí:** chỉ dùng phần Chương 1 của một sách; 5 câu test và khả năng overlap giữa dev/test hạn chế độ khái quát. Chunk dài vẫn bị cắt theo giới hạn; coherence được lấy tại đầu trang nên có thể không đại diện toàn corpus. Nhãn được học viên xác nhận từ bản nháp AI, không blind; chi phí chưa đối soát dashboard và thời gian chưa tách theo strategy.
- **Nếu làm lại hoặc mở rộng, cải tiến gì?:** dùng corpus đa dạng/đầy đủ hơn và bộ test lớn hơn với chủ đề tách biệt; chọn mẫu audit phân tầng, nhờ người chấm thứ hai chấm blind; đo latency/chi phí theo từng strategy và ghi giá API tại thời điểm chạy; phân tích riêng truy vấn có/không có filter.
