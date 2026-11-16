# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Chạy agent với Ollama và tool `search` trên kho Day06 |
| `baseline.py` | Chạy TF-IDF offline hoặc so sánh dense, TF-IDF và hybrid trên dev |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |
| `day6_retriever.py` | Adapter dùng lại embedding và truy vấn của Day06, có lệnh kiểm tra kết nối |
| `retrieval_benchmark.py` | Hàm RRF, trace và metric dùng từ `baseline.py` |

## 1. Chạy agent

`agentic.py` dùng model chat `qwen3.5:4b` qua Ollama. Model phát tool call `search`, Python dùng adapter để tìm chunk trong pgvector rồi gửi context cho model. Lượt gọi đầu dùng nguyên câu hỏi. Các lượt tìm tiếp dùng query do model đề xuất. Trace giữ cả query model yêu cầu và query thực thi. Có thể đổi model hỗ trợ tool calling với `--model`. Đặt `DAY6_ROOT`, `DATABASE_URL` và `OLLAMA_BASE_URL` theo phần kết nối bên dưới.

```bash
python3 -B starter/agentic.py --query 'Nghị định chia dữ liệu cá nhân thành hai nhóm nào?'
```

Nếu chạy bằng Docker, dùng cùng lệnh ở phần kiểm tra kho Day06 và thay lệnh Python cuối bằng:

```text
python -B /day7/starter/agentic.py --query 'Nghị định chia dữ liệu cá nhân thành hai nhóm nào?'
```

JSON có `status`, câu trả lời, citation, `tool_call_count`, `model_calls` và `events`. Các tham số `--max-searches`, `--top-k` và `--budget` điều chỉnh điều kiện dừng và context. Kiểm tra trực tiếp đoạn dẫn để xác nhận citation hỗ trợ câu trả lời.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

Với benchmark của bài, dùng `python -B starter/baseline.py --compare --output results/retrieval_dev_rerun` sau khi đặt các biến môi trường ở mục 3. Nếu dùng Docker, giữ phần cấu hình container ở mục 4 và thay lệnh Python cuối bằng `python -B starter/baseline.py --compare --output /results/retrieval_dev_rerun`. [Báo cáo và kết quả dev](../report/individual/03936_NguyenHaiQuang.md#retrieval-và-thử-nghiệm)

## 3. Dùng dữ liệu Day 6

### Kiểm tra kho pgvector Day 6 của bài này

`day6_retriever.py` dùng code embedding và pgvector trong repo Day 6. Adapter kết nối chỉ đọc, dùng index B (190 chunk trong snapshot đã nộp) và giữ nguyên ID, văn bản, nguồn, trang và metadata. Khi truyền filter `section` hoặc `form_number`, adapter áp dụng cách lọc C trên cùng index B.

Mở Docker Desktop và Ollama. Từ gốc repo Day 7, chạy trong PowerShell. Thay `DAY6_ROOT` nếu hai repo không nằm cạnh nhau. [Repo Day06 để tái tạo kho](https://github.com/qwerty12321434/K5-L3B-DAY06-NguyenHaiQuang-03936-DataIngestion)

```powershell
$env:DAY6_ROOT = (Resolve-Path '..\K5-L3B-DAY06-NguyenHaiQuang-03936-DataIngestion').Path
docker compose -f "$env:DAY6_ROOT\starter\docker-compose.yml" -p day6_student_starter up -d --wait
$env:DATABASE_URL = 'postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab'
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
python -m pip install 'psycopg[binary]==3.3.8'
python -B starter/day6_retriever.py --top-k 3
```

Khi thành công, JSON có `"status": "ok"`, model `qwen3-embedding:0.6b`, `dimensions: 1024`, số chunk trong index B và danh sách kết quả có rank, score, nguồn và trang. Có thể truyền `--query "..."` để thử câu khác.

Nếu dùng lại image `day6-ingestion:local` đã build cho Day 6, có thể chạy thay hai lệnh Python ở trên bằng:

```powershell
docker run --rm --network day6_student_starter_default `
  --mount "type=bind,source=$($PWD.Path),target=/day7,readonly" `
  --mount "type=bind,source=$env:DAY6_ROOT,target=/day6,readonly" `
  -e 'DAY6_ROOT=/day6' `
  -e 'DATABASE_URL=postgresql://day6:day6_local_only@postgres:5432/day6_lab' `
  -e 'OLLAMA_BASE_URL=http://host.docker.internal:11434' `
  day6-ingestion:local python -B /day7/starter/day6_retriever.py --top-k 3
```

Kiểm tra `ollama list` nếu thiếu model. Nếu volume PostgreSQL chưa có index B, chạy bước **Chạy ingest A/B/C bằng Docker** trong `starter/README.md` của Day 6 để tái tạo index từ corpus Day 6.

### Xuất chunk và câu hỏi để chạy baseline offline

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 4. Tổng hợp đánh giá agent

### Chọn cấu hình cho test

Chạy cả ba phương pháp trên dev với cùng model và context budget. Dense là baseline. Chọn cấu hình đối chiếu giữa TF-IDF và hybrid theo thứ tự: tổng điểm câu trả lời có bằng chứng trên 15 câu có đáp án, số từ chối đúng, số context chấm 2, rồi thời gian trung vị thấp hơn. Lượt lỗi không được cộng điểm và ghi riêng. Khi đối chiếu trên dev, nếu dense dẫn đầu thì ghi rõ dense vẫn là mốc tốt nhất trên dev theo tiêu chí này. Chốt model, corpus, code và tham số trước khi chạy cùng 7 câu test cho hai cấu hình.

### Chạy lại bằng Docker

Mở Docker Desktop và Ollama. Từ gốc repo Day07 trong PowerShell, dùng lại image `day6-ingestion:local` và volume Day06. Hai lệnh bên dưới ghi lần chạy lại vào thư mục riêng.

```powershell
$day6Repo = 'D:\Workspace\K5-L3B-DAY06-NguyenHaiQuang-03936-DataIngestion'
docker compose -f "$day6Repo\starter\docker-compose.yml" -p day6_student_starter up -d --wait
$runArgs = @(
  'run', '--rm', '--network', 'day6_student_starter_default',
  '--mount', "type=bind,source=$($PWD.Path),target=/day7,readonly",
  '--mount', "type=bind,source=$(Join-Path $PWD 'results'),target=/results",
  '--mount', "type=bind,source=$day6Repo,target=/day6,readonly",
  '-e', 'DAY6_ROOT=/day6',
  '-e', 'DATABASE_URL=postgresql://day6:day6_local_only@postgres:5432/day6_lab',
  '-e', 'OLLAMA_BASE_URL=http://host.docker.internal:11434',
  '--workdir', '/day7', 'day6-ingestion:local', 'python', '-B', 'starter/agentic.py'
)
docker @runArgs --split dev --output /results/agent_dev_rerun
docker @runArgs --split test --selection /day7/results/agent_dev/selection.json --output /results/agent_test_rerun
```

`run.json` lưu dấu vân tay của code, model, corpus và benchmark. Lệnh test kiểm tra cấu hình và SHA-256 của file câu hỏi theo `selection.json`, rồi tự chạy hai phương pháp đã ghi trong đó. Dùng đúng bản Day06, model Ollama và file câu hỏi đã chốt.

### Xem và tổng hợp nhãn

Trong mỗi thư mục phương pháp, `traces.json` là các câu chạy thật và `reviews.json` là nhãn đọc tay của lần chạy đó. Đọc `model_calls` để xem câu model sinh, số token và context thực gửi. Đối chiếu kết quả mới với nguồn trước khi dùng nhãn cho một lần chạy lại.

```powershell
python -X utf8 -B starter/agent_metrics.py `
  --split dev --cases data/benchmark_dev.json `
  --traces results/agent_dev/dense/traces.json `
  --reviews results/agent_dev/dense/reviews.json `
  --details --output results/agent_dev/dense/summary.json
```

`--details` tổng hợp xếp hạng theo gold và context theo nhãn `context_grade` do người đọc chấm. Mỗi review kèm lý do cho câu trả lời, context áp dụng và từng citation. File tổng hợp có phân bố lượt search, thời gian, token, chi phí API và lỗi.

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
