# Báo cáo Day 7 — Agentic RAG

**Nguyễn Hải Quang · MSSV 03936 · K5-L3B**

Ngày chạy benchmark 16/11/2026 (UTC+7).

## Dữ liệu và benchmark

- **Nguồn.** Bản Công báo số 685 + 686 của Nghị định 13/2023/NĐ-CP có 37 trang PDF. Day06 đưa 36 trang nội dung vào kho và bỏ trang danh mục biểu mẫu. Day07 dùng **190 chunk B**, giữ ID, URL và số trang PDF. Bản Công báo được công bố công khai, dùng cho bài thực hành. Quyền tái phân phối PDF chưa được xác minh, nên cung cấp link nguồn để tái tạo. [Bản Công báo](https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-13-2023-nd-cp-39228.htm)
- **Vector store và embedding.** PostgreSQL 16.16, pgvector 0.8.7, bảng `day6_chunks` cấu hình B. Ollama 0.40.16 dùng `qwen3-embedding:0.6b` Q8_0, **1.024 chiều**, cùng model digest và chỉ dẫn embedding với Day06. Adapter kết nối chỉ đọc. [Cách kết nối hoặc dựng lại kho](../../starter/README.md#3-dùng-dữ-liệu-day-6) · [Phiên bản và digest](../../results/agent_dev/run.json)
- **Agent và split.** Model chat `qwen3.5:4b` chạy cục bộ qua Ollama. Dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng. Test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Giữ split của 20 câu có đáp án Day06, đổi câu dev `q10` sang một ý cùng Điều 16 có evidence trong một chunk và thêm 4 câu thiếu bằng chứng. [Dev](../../data/benchmark_dev.json) · [Test](../../data/benchmark_test.json)
- **Cách chấm.** Hit đòi hỏi đúng nguồn, đúng trang và chứa một đoạn gold đã lưu. Nhãn gold chỉ dùng khi chấm. Với câu thiếu bằng chứng, kiểm tra điều và biểu mẫu liên quan rồi rà từ khóa trên cả corpus. Chấm context và câu trả lời theo thang 0–2. Citation chỉ hợp lệ nếu nằm trong context và hỗ trợ phần câu trả lời được dẫn. Codex đọc trace và evidence để ghi nhãn, kèm lý do theo từng câu để học viên kiểm tra. [Tiêu chí](../../BENCHMARK.md)

## Retrieval và thử nghiệm

Cả ba phương pháp dùng cùng 190 chunk, lấy top 3, không áp dụng metadata filter hay rerank riêng. Ghép context theo thứ hạng với giới hạn **1.500 ký tự**, tính cả header nguồn. Lượt tìm đầu dùng câu hỏi gốc. Trong các lần chạy agent đã chấm, mỗi câu gọi search đúng một lần, nên chỉ số hit tính trên mọi lượt bằng hit của query gốc. Các tỷ lệ retrieval và context chỉ tính trên câu có đáp án.

| Cấu hình | Phương pháp | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Trung vị / chi phí API |
|---|---|---|---|---:|---:|---:|---:|
| Dense · baseline | pgvector, cosine | Mốc so sánh với vector Day06 | dev | 8/15 (53,33%) | 12/15 (80,00%) | 12/15 (80,00%) | 1.904,78 ms / 0 USD |
| TF-IDF · cải thiện 1 | TF-IDF, cosine từ khóa | Thử dùng từ khóa Điều hoặc tên trường để đẩy đoạn đúng lên đầu | dev | 9/15 (60,00%) | 11/15 (73,33%) | 10/15 (66,67%) | 1.971,20 ms / 0 USD |
| Hybrid RRF · cải thiện 2 | Dense + TF-IDF, tối đa 10 ứng viên mỗi nhánh, RRF `k=60` | Thử kết hợp tìm theo nghĩa và theo từ khóa | dev | 10/15 (66,67%) | 11/15 (73,33%) | 11/15 (73,33%) | 2.027,61 ms / 0 USD |
| Dense · baseline | pgvector, cosine | Giữ nguyên cấu hình dev | test | 2/5 (40,00%) | 2/5 (40,00%) | 2/5 (40,00%) | 2.162,77 ms / 0 USD |
| Hybrid RRF · đã chốt | Dense + TF-IDF, tối đa 10 ứng viên mỗi nhánh, RRF `k=60` | Giữ nguyên cấu hình dev | test | 2/5 (40,00%) | 3/5 (60,00%) | 2/5 (40,00%) | 2.181,76 ms / 0 USD |

Bảng trên đo lần chạy **agent thật**. Trung vị tính cả gọi LLM, retrieval và ghép context. Context Evidence Rate là số context cuối đủ bằng chứng theo nhãn đọc trace. Lần đo **retrieval riêng** dưới đây dừng sau khi ghép context, trước khi gọi LLM. Lần này không có token LLM, không có lỗi và chi phí API 0 USD trên cả ba cấu hình. Khởi tạo TF-IDF mất 8,839 ms và làm nóng embedding mất 3.729,615 ms, ghi riêng với thời gian truy vấn. [Số liệu retrieval riêng](../../results/retrieval_dev/summary.json) · [Trace retrieval](../../results/retrieval_dev/traces.jsonl)

| Retrieval riêng trên dev | Hit@1 | Evidence Hit@3 | Context đã ghép có gold | Trung vị |
|---|---:|---:|---:|---:|
| Dense | 8/15 (53,33%) | 12/15 (80,00%) | 12/15 (80,00%) | 26,485 ms |
| TF-IDF | 9/15 (60,00%) | 11/15 (73,33%) | 10/15 (66,67%) | 0,664 ms |
| Hybrid RRF | 10/15 (66,67%) | 11/15 (73,33%) | 11/15 (73,33%) | 27,792 ms |

- **Lỗi dev.** Ở `q01`, dense tìm đúng định nghĩa ở hạng 2 nhưng hybrid làm mất đoạn đúng khỏi top 3. Ở `q14`, TF-IDF tìm đúng hạng 3 mà đoạn đó không vừa context. Hybrid đưa đoạn đúng lên hạng 2 và agent trả lời đúng trong lần chạy dev đã lưu. Ở `q03`, cả ba agent đều từ chối dù có đủ bằng chứng trong context. Ở `q04`, dense và hybrid sinh thêm citation `[4]` khi context chỉ có ba đoạn nên câu trả lời bị từ chối.
- **Thay đổi trước khi chốt.** Lượt dev đầu của dense và hybrid ghi `q15` là đã trả lời dù model chỉ sinh `[1]`. Sau khi thêm kiểm tra nội dung và chạy lại đủ 51 lượt dev, hai cấu hình trả trạng thái thiếu bằng chứng. Điểm của `q15` vẫn là 0 vì đây là câu có đáp án. Trace giữ đầu ra gốc của model. [Kiểm thử hồi quy](../../tests/test_agentic.py)
- **Lý do chọn.** Dense giữ vai trò baseline và dẫn dev với 18/30 điểm câu trả lời. Hybrid và TF-IDF cùng 16/30 và cùng từ chối đúng 2/2 câu. Hybrid được chọn làm cấu hình đối chiếu vì context đủ bằng chứng ở 11/15 câu, cao hơn 10/15 của TF-IDF. Chốt lúc **10:58:13**, chạy test lúc **10:58:36** (UTC+7), giữ nguyên model, corpus, code, tham số và bộ câu hỏi đã lưu dấu vân tay. [Bản chốt](../../results/agent_dev/selection.json) · [Thông tin lần chạy test](../../results/agent_test/run.json)

## Agent và đánh giá

`agentic.py` đưa câu hỏi cho model chat. Khi model gọi `search`, Python chạy phương pháp retrieval đã chọn. Python gom các chunk theo thứ tự nhận, bỏ ID trùng, chọn các đoạn vừa ngân sách và trả context cho model. Model có thể đề xuất query mới ở lượt sau. Agent dừng khi có câu trả lời với citation hợp lệ về ID, khi model báo thiếu bằng chứng, hoặc khi vượt giới hạn tối đa 4 lượt search. Lỗi Ollama hoặc pgvector được lưu riêng. Kiểm tra ID ở runtime chỉ xác nhận nguồn nằm trong context, nên nhãn citation vẫn cần đọc nội dung. [Code agent](../../starter/agentic.py)

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 · TB · tổng | Citation hợp lệ | Câu trả lời có citation | Từ chối đúng | Từ chối nhầm |
|---|---:|---|---|---:|---:|---:|---:|
| Dense / dev | 17/17 | 17 câu × 1 lượt | 5/2/8 · 1,20/2 · 18/30 điểm | 9/11 | 11/11 | 1/2 | 5/15 |
| TF-IDF / dev | 17/17 | 17 câu × 1 lượt | 6/2/7 · 1,07/2 · 16/30 điểm | 8/10 | 9/9 | 2/2 | 6/15 |
| Hybrid RRF / dev | 17/17 | 17 câu × 1 lượt | 6/2/7 · 1,07/2 · 16/30 điểm | 8/10 | 9/9 | 2/2 | 6/15 |
| Dense / test | 7/7 | 7 câu × 1 lượt | 2/1/2 · 1,00/2 · 5/10 điểm | 2/6 | 5/5 | 2/2 | 0/5 |
| Hybrid RRF / test | 7/7 | 7 câu × 1 lượt | 2/1/2 · 1,00/2 · 5/10 điểm | 2/6 | 5/5 | 2/2 | 0/5 |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn và lý do |
|---|---|---|---|---|
| q16 | Có đáp án | Cả hai tìm được đoạn trang 17 ở hạng 1. | Cả hai nêu cần sự đồng ý của trẻ và cha, mẹ hoặc người giám hộ, dẫn trang 17. | Context 2, trả lời 2. Nguồn hỗ trợ đúng điều kiện. |
| q17 | Có đáp án | Cả hai tìm được đoạn trang 18 ở hạng 1. | Cả hai nêu phải kèm lý do chậm, muộn, dẫn trang 18. | Context 2, trả lời 2. Citation đúng. |
| q18 | Có đáp án | Cả hai đưa các đoạn về hồ sơ đánh giá tác động trang 21 vào context. | Cả hai liệt kê nội dung hồ sơ, dẫn hai đoạn trang 21. | Context 0, trả lời 0. Hai citation không hỗ trợ thông báo sau khi chuyển thành công. |
| q19 | Có đáp án | Cả hai lấy phần đầu Mẫu 02 ở trang 32, thiếu đoạn có địa chỉ trụ sở. | Cả hai trả lời tên tổ chức, doanh nghiệp và dẫn trang 32. | Context 0, trả lời 0. Dẫn đúng mẫu nhưng sai trường. |
| q20 | Có đáp án | Dense ưu tiên Điều 23 ở trang 18. Hybrid có đoạn đúng của Mẫu 03 ở hạng 3, nhưng bỏ khỏi context vì quá ngân sách. | Cả hai mô tả biện pháp giảm thiểu theo Điều 23, dẫn trang 18. | Context 1, trả lời 1. Nêu được ý biện pháp nhưng thiếu tên trường **Biện pháp áp dụng** và citation đúng mẫu. |
| q23 | Thiếu bằng chứng | Cả hai tìm hướng dẫn gửi phiếu và phần đầu Mẫu 01. | Cả hai báo thiếu bằng chứng, không đưa ra mức phí. | Từ chối đúng. |
| q24 | Thiếu bằng chứng | Cả hai tìm quy định chung về bảo vệ dữ liệu. | Cả hai báo thiếu bằng chứng, không đưa ra thuật toán. | Từ chối đúng. |

| Cấu hình / split | Context 0/1/2 | Token prompt / sinh | Lỗi | Kết quả từng câu |
|---|---|---:|---:|---|
| Dense / dev | 2/1/12 | 23.915 / 1.390 | 0/17 | [Trace](../../results/agent_dev/dense/traces.json) · [Nhãn](../../results/agent_dev/dense/reviews.json) · [Số liệu](../../results/agent_dev/dense/summary.json) |
| TF-IDF / dev | 5/0/10 | 23.529 / 1.448 | 0/17 | [Trace](../../results/agent_dev/tfidf/traces.json) · [Nhãn](../../results/agent_dev/tfidf/reviews.json) · [Số liệu](../../results/agent_dev/tfidf/summary.json) |
| Hybrid RRF / dev | 2/2/11 | 23.621 / 1.414 | 0/17 | [Trace](../../results/agent_dev/hybrid_rrf/traces.json) · [Nhãn](../../results/agent_dev/hybrid_rrf/reviews.json) · [Số liệu](../../results/agent_dev/hybrid_rrf/summary.json) |
| Dense / test | 2/1/2 | 9.976 / 774 | 0/7 | [Trace](../../results/agent_test/dense/traces.json) · [Nhãn](../../results/agent_test/dense/reviews.json) · [Số liệu](../../results/agent_test/dense/summary.json) |
| Hybrid RRF / test | 2/1/2 | 9.876 / 876 | 0/7 | [Trace](../../results/agent_test/hybrid_rrf/traces.json) · [Nhãn](../../results/agent_test/hybrid_rrf/reviews.json) · [Số liệu](../../results/agent_test/hybrid_rrf/summary.json) |

**Kết luận.** Trên test, hybrid tăng Evidence Hit@3 từ 2/5 lên 3/5 nhưng cả hai vẫn chỉ có đủ bằng chứng trong context ở 2/5 câu, trả lời đúng đủ 2/5 câu và hợp lệ 2/6 citation. Ở `q20`, hai đoạn đầu chiếm 1.089 ký tự, còn 411 ký tự. Thêm đoạn đúng và dấu phân cách cần 514 ký tự, nên model không nhận được đoạn đó. Ở `q18` và `q19`, cần cải thiện kết quả truy xuất đúng điều khoản hoặc trường biểu mẫu. Mẫu test có 5 câu có đáp án trên một văn bản. Đây là kết quả của một lần chạy đã chốt.

**Tái lập.** Mở Docker Desktop và Ollama, dùng volume và image Day06 theo lệnh PowerShell trong README. `baseline.py --compare` chạy retrieval riêng, `agentic.py --split dev` chạy ba agent, `agentic.py --split test --selection results/agent_dev/selection.json` chạy hai cấu hình đã chốt, và `agent_metrics.py --details` tổng hợp trace với nhãn. Dùng thư mục output mới khi chạy lại. [Lệnh đầy đủ](../../starter/README.md#chạy-lại-bằng-docker) · [Code tổng hợp](../../starter/agent_metrics.py)

## Thử thách nâng cao (nếu thực hiện)

Phạm vi báo cáo là phần cơ bản.
