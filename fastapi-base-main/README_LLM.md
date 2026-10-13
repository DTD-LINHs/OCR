# Phần LLM: sửa lỗi OCR và nhận xét bài làm

Phần này thêm 2 chức năng vào backend:

| Endpoint | Việc | Ai gọi |
|---|---|---|
| `POST /api/v1/llm/correct-ocr` | Sửa lỗi **nhận dạng** trong output OCR. Không sửa lỗi của sinh viên. | Sau khi OCR service trả kết quả |
| `POST /api/v1/llm/review` | Nhận xét bài làm + điểm gợi ý, dựa trên code và kết quả test | Sau khi sandbox chạy xong |
| `GET /api/v1/llm/info` | Xem đang dùng provider/model nào | Debug |

Luồng tổng thể:

```
Ảnh → [OCR] → code thô → [LLM /correct-ocr] → code sạch → (giảng viên duyệt) → [Sandbox] → kết quả test → [LLM /review] → nhận xét
```

---

## 1. Các file đã thêm / sửa

```
app/
├── api/v1/api_llm.py        MỚI  – 3 endpoint (router được bộ khung tự đăng ký)
├── schemas/sche_llm.py      MỚI  – format JSON vào/ra của 2 endpoint
├── services/srv_llm.py      MỚI  – gọi LLM, đọc JSON, kiểm tra an toàn, chế độ mock
├── services/llm_prompts.py  MỚI  – toàn bộ prompt (sửa prompt ở đây, không cần đụng logic)
└── core/config.py           SỬA  – thêm các biến LLM_*
tests/
├── conftest.py              MỚI
└── test_llm.py              MỚI  – 21 test, không cần DB, không cần LLM thật
.env.example                 SỬA  – thêm khối cấu hình LLM
README_LLM.md                MỚI  – file này
dev_llm_server.py            MỚI  – chạy riêng phần LLM, không cần PostgreSQL (cổng 8001)
```

Không sửa file nào của người khác ngoài `config.py` và `.env.example` (chỉ **thêm** dòng, không đổi dòng cũ).

---

## 2. Cấu hình (`.env`)

```env
LLM_PROVIDER=mock                        # mock | openai
LLM_BASE_URL=http://localhost:11434/v1   # địa chỉ API
LLM_API_KEY=                             # để trống nếu chạy local
LLM_MODEL=qwen2.5-coder:7b
LLM_TEMPERATURE=0                        # 0 = ổn định, lần nào chạy cũng ra gần giống nhau
LLM_TIMEOUT_SECONDS=180
LLM_JSON_MODE=true                       # server nào không hỗ trợ thì code tự thử lại không dùng
LLM_MIN_SIMILARITY=0.7                   # ngưỡng phát hiện LLM "viết lại" code
```

- **`mock`**: không gọi LLM thật. `/correct-ocr` chỉ áp 2 luật đơn giản (`a-b` → `a_b`, từ khóa viết hoa → chữ thường), `/review` chấm theo tỷ lệ test đạt. Dùng để **bạn OCR, bạn sandbox và frontend tích hợp ngay**, không cần API key.
- **`openai`**: gọi bất kỳ API nào theo chuẩn *OpenAI Chat Completions*. Đổi model chỉ cần đổi 3 dòng:

| Dùng gì | `LLM_BASE_URL` | `LLM_MODEL` (ví dụ) | `LLM_API_KEY` |
|---|---|---|---|
| Ollama chạy trên máy | `http://localhost:11434/v1` | `qwen2.5-coder:7b` | để trống |
| Ollama, app chạy trong docker-compose | `http://host.docker.internal:11434/v1` | `qwen2.5-coder:7b` | để trống |
| OpenAI | `https://api.openai.com/v1` | tên model | key |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | tên model | key |
| **OpenRouter (nhóm đang dùng)** | `https://openrouter.ai/api/v1` | `deepseek/deepseek-v4.1-flash`, `openai/gpt-4o-mini`... | `OPENROUTER_API_KEY` hoặc `LLM_API_KEY` |
| Groq / vLLM / LM Studio | URL của dịch vụ, có `/v1` | tên model | tùy |

> Lưu ý quyền riêng tư: dùng API bên ngoài nghĩa là bài làm của sinh viên được gửi ra ngoài. Demo thì được, khi đưa vào web trường nên cân nhắc model chạy local.

---

## 3. Chạy thử

**Chạy test** (không cần DB, không cần LLM):
```bash
pip install -r requirements.txt
pytest tests/test_llm.py -v
```

**Chạy cả app** (bộ khung cần PostgreSQL nên dùng docker):
```bash
cp .env.example .env        # sửa LLM_* nếu cần
docker compose up --build
```
Mở http://localhost:8000/docs, tìm nhóm **v1 - api_llm**, bấm *Try it out*.

---

## 4. Format dữ liệu (để chốt với nhóm)

Mọi response đều bọc trong `DataResponse` của bộ khung: `{http_code, success, message, metadata, data}`. Phần dưới là nội dung của `data`.

### 4.1. `POST /api/v1/llm/correct-ocr`

**Request**, do OCR service gửi:
```json
{
  "ocr_text": "retusn Ans\n...",
  "lines": [{"text": "retusn Ans", "score": 0.81}],
  "language": "python",
  "problem": "Viết chương trình tính giai thừa"
}
```
Chỉ `ocr_text` là bắt buộc. `lines[].score` (độ tin cậy của PaddleOCR) và `problem` giúp LLM sửa chính xác hơn.

**Gửi kèm ảnh gốc (khuyến nghị nếu model nhận ảnh):** thêm `"image_base64": "<base64 của ảnh>"`
(hoặc data URL `data:image/jpeg;base64,...`, hoặc `"image_url": "https://..."`). Khi có ảnh, LLM đối chiếu
OCR với chữ viết thật nên sửa được cả dòng OCR bỏ sót, nhưng vẫn phải giữ nguyên lỗi của sinh viên.
Nên resize ảnh về cạnh dài khoảng 1024–1600px trước khi gửi (giảm chi phí, giới hạn ~7.5MB).
Model phải hỗ trợ đầu vào ảnh (trên OpenRouter: mục *Modalities* có biểu tượng ảnh).

**Response**:
```json
{
  "code": "return Ans\n...",
  "changes": [{"line": 6, "before": "retusn Ans", "after": "return Ans",
               "reason": "OCR nhầm r/s trong từ khóa", "confidence": "high"}],
  "uncertain": [{"line": 3, "text": "n / 2", "note": "Có thể sinh viên dùng sai / thay vì %"}],
  "syntax_ok": true,
  "syntax_error": null,
  "similarity": 0.91,
  "needs_review": true,
  "review_reasons": ["Có 1 chỗ LLM không chắc là lỗi OCR hay lỗi của sinh viên"],
  "used_image": false,
  "model": "qwen2.5-coder:7b"
}
```

| Trường | Frontend nên dùng thế nào |
|---|---|
| `code` | Hiện cạnh ảnh gốc, cho giảng viên sửa tay, rồi mới gửi sang sandbox |
| `changes` | Highlight các dòng LLM đã sửa (`before` → `after`), tô màu theo `confidence` |
| `uncertain` | Đánh dấu chỗ cần giảng viên đối chiếu với ảnh |
| `needs_review` + `review_reasons` | Hiện cảnh báo "Cần kiểm tra" |
| `syntax_ok` / `syntax_error` | Biết trước code có chạy được không, trước khi gửi sandbox |

### 4.2. `POST /api/v1/llm/review`

**Request**, gửi sau khi sandbox chạy:
```json
{
  "problem": "Viết chương trình tính giai thừa của n",
  "code": "n = int(input())\n...",
  "test_results": [
    {"name": "n=5", "input": "5", "expected": "120", "actual": "120", "passed": true, "error": null, "hidden": false},
    {"name": "n=0", "input": "0", "expected": "1", "actual": "", "passed": false,
     "error": "RecursionError: ...", "hidden": true}
  ],
  "compile_error": null,
  "rubric": "Đúng kết quả 7đ, xử lý n=0 1đ, trình bày 2đ",
  "max_score": 10
}
```
- `hidden: true`: test ẩn, **không** gửi input/expected cho LLM, để nhận xét không lộ test ẩn cho sinh viên.
- `compile_error`: nếu code không chạy được, sandbox gửi lỗi vào đây và để `test_results` rỗng.

**Response**: gồm đúng 3 phần theo tiêu chí chấm, cộng điểm gợi ý và cảnh báo của server.
```json
{
  "general": {
    "meets_requirements": "no",
    "requirements_comment": "Kết quả luôn bằng 0 nên chưa đúng yêu cầu tính n!",
    "standard_comment": "Dòng 1 có chuỗi gợi ý trong input(), khi chấm tự động chuỗi này bị in ra output",
    "optimization_comment": "Một vòng lặp O(n) là phù hợp",
    "complexity": "O(n)",
    "issues": [
      {"line": 1, "code_snippet": "input(\"Nhập n: \")", "type": "io_format", "severity": "medium",
       "explanation": "...", "suggestion": "Bỏ chuỗi trong input()"},
      {"line": 2, "code_snippet": "result = 0", "type": "logic", "severity": "high",
       "explanation": "Nhân với 0 luôn ra 0", "suggestion": "Khởi tạo bằng 1"}
    ]
  },
  "solution_guide": {
    "approach": "Nhân dồn các số từ 1 đến n",
    "steps": ["Đọc n", "Khởi tạo kết quả bằng 1", "Nhân dồn", "In kết quả"],
    "pseudocode": "ĐỌC n\nkq <- 1\nLẶP i TỪ 1 ĐẾN n: kq <- kq * i\nIN kq",
    "expected_complexity": "O(n)"
  },
  "conclusion": {"verdict": "incorrect", "summary": "Bài sai vì khởi tạo result = 0 ..."},
  "suggested_score": 3,
  "max_score": 10,
  "score_explanation": "...",
  "feedback_for_student": "...",
  "passed_tests": 0,
  "total_tests": 2,
  "warnings": [],
  "model": "gpt-4o"
}
```

| Phần | Tiêu chí chấm | Trường |
|---|---|---|
| 1. Đánh giá chung | Đúng yêu cầu đề bài | `general.meets_requirements`, `requirements_comment` |
| | Đúng chuẩn ngôn ngữ / lập trình thi đấu (đọc stdin, in đúng format, **không có chuỗi gợi ý trong `input()`**, không in debug) | `general.standard_comment`, issue `type: io_format` / `standard` |
| | Tối ưu code, gợi ý đoạn cần sửa | `general.optimization_comment`, `complexity`, `issues` (có `line` + `code_snippet` + `suggestion`) |
| 2. Hướng dẫn giải | Phương pháp, các bước, **chỉ mã giả, không đưa code** | `solution_guide.approach`, `steps`, `pseudocode` |
| 3. Kết luận | Đúng/sai trong 1 đoạn ngắn | `conclusion.verdict` (`correct` / `partially_correct` / `incorrect`), `conclusion.summary` |

`passed_tests` / `total_tests` do server tự đếm, không phụ thuộc LLM. `suggested_score` luôn bị giới hạn trong `[0, max_score]`.

---

## 5. Bên trong `srv_llm.py` làm gì

### `correct_ocr()`
1. Dựng prompt (`llm_prompts.py`): quy tắc + **1 ví dụ mẫu** + output OCR có đánh số dòng và score.
2. Gọi LLM, đọc JSON. Nếu LLM bọc JSON trong ```` ```json ```` hay có chữ thừa, `extract_json()` vẫn lấy ra được.
3. **Kiểm tra an toàn**, phần quan trọng nhất. Hệ thống dùng để chấm điểm, nên LLM tuyệt đối không được sửa lỗi của sinh viên:
   - `similarity`: so code sau khi sửa với bản OCR, bỏ qua khoảng trắng và hoa/thường. Nếu dưới `LLM_MIN_SIMILARITY` thì LLM có thể đã viết lại code, nên bật `needs_review`.
   - Số dòng tăng lên: LLM có thể đã thêm code.
   - Có thay đổi `confidence: low` hoặc có `uncertain`: cần người xem.
   - `syntax_ok`: dùng `compile()` để kiểm tra cú pháp, **không chạy code**.
   - Nếu LLM không tự liệt kê `changes`, server tự tính diff từng dòng.

### `review()`
1. Dựng prompt gồm đề bài, code có số dòng, kết quả test, rubric, thang điểm. Test ẩn chỉ gửi dạng
   `hidden test N: PASS/FAIL`: không gửi input/expected, **không gửi cả tên test** (tên như `n=0` đã lộ input).
   Prompt yêu cầu LLM trả đúng 3 phần: đánh giá chung, hướng dẫn giải (chỉ mã giả), kết luận.
2. Gọi LLM, đọc JSON. Thiếu `general` / `solution_guide` / `conclusion` thì gọi lại 1 lần, vẫn thiếu thì trả 502
   và ghi log output thô (tối đa 2000 ký tự). `/correct-ocr` làm tương tự khi thiếu `code`.
3. Chuẩn hóa: giá trị `type`/`severity`/`verdict` lạ thì đổi về mặc định; điểm bị kẹp trong `[0, max_score]`.
4. **Server tự kiểm tra**, không tin hoàn toàn vào LLM, kết quả đưa vào `warnings`:
   - Phần hướng dẫn giải có chứa code Python chạy được thay vì mã giả không.
   - Code có chuỗi gợi ý trong `input()` mà nhận xét không nêu ra không.
   - Kết luận có mâu thuẫn với kết quả test không (vd. đạt hết test mà kết luận `incorrect`).
   - Nhận xét cho sinh viên có làm lộ dữ liệu test ẩn không.

### Prompt: chỉnh ở `llm_prompts.py`
- `CORRECT_OCR_SYSTEM`: các quy tắc **SỬA / KHÔNG SỬA / KHÔNG CHẮC**. Muốn LLM sửa nhiều hơn hay ít hơn thì sửa ở đây.
- `CORRECT_OCR_EXAMPLE_*`: ví dụ mẫu. Ví dụ cố ý chứa một lỗi logic của sinh viên (`n / 2` thay vì `n % 2`) được **giữ nguyên**, để dạy LLM phân biệt.
- `REVIEW_SYSTEM`: cách chấm, giọng văn nhận xét, yêu cầu viết tiếng Việt.

---

## 6. Lỗi trả về

| HTTP | Khi nào |
|---|---|
| 422 | Request thiếu trường bắt buộc / sai kiểu |
| 502 | LLM trả lỗi, trả không phải JSON, hoặc thiếu trường bắt buộc (`code`; `general`/`solution_guide`/`conclusion`) sau khi đã thử lại 1 lần |
| 503 | Không kết nối được tới `LLM_BASE_URL` (chưa bật Ollama, sai URL...) |

---

## 7. Hạn chế đã biết

- **Kiểm tra lộ test ẩn trong `warnings` bỏ qua giá trị ngắn.** Server chỉ cảnh báo khi input/expected của test ẩn
  dài từ 3 ký tự trở lên và xuất hiện nguyên văn trong `feedback_for_student`. Với test như `n=0 -> 1`
  (giá trị `"0"`, `"1"`), nếu bắt cả giá trị ngắn thì gần như nhận xét nào cũng bị cảnh báo nhầm, nên server
  **không tự phát hiện** được. Cũng không bắt được khi LLM diễn đạt lại (vd. "trường hợp n bằng 0 phải ra 1").
  Biện pháp hiện có: không gửi input/expected/tên test ẩn cho LLM, và dặn trong prompt; giảng viên vẫn nên đọc lại
  phần nhận xét trước khi gửi cho sinh viên.
- **Model có thể tự nêu các trường hợp biên phổ biến** (vd. `n = 0`, `0! = 1`) dựa trên kiến thức chung, không phải
  do lộ test ẩn: đã thử bỏ cả tên test ẩn lẫn ràng buộc `0 <= n <= 20` khỏi đề bài, model vẫn nhắc `n = 0`.
  Nhóm chọn giữ nguyên prompt (dặn chặt hơn sẽ làm nhận xét kém hữu ích); **nhận xét cần được giảng viên duyệt
  trước khi gửi sinh viên.**
- **Thời gian phản hồi phụ thuộc nhà cung cấp mà OpenRouter định tuyến tới.** Đo với `deepseek/deepseek-v4.1-flash`:
  thường 5.5–17 giây mỗi request, có lần 83 giây (request `/review`, không có ảnh). Vì vậy timeout mặc định là 180 giây.

---

## 8. Việc tiếp theo

- [ ] Chốt format ở mục 4 với bạn làm OCR và bạn làm sandbox.
- [ ] Chọn LLM cho demo, đổi `LLM_PROVIDER=openai` + 3 biến còn lại.
- [ ] Đánh giá `/correct-ocr` trên 55 ảnh của dataset: CER trước/sau LLM, và kiểm tra **11 ảnh có lỗi logic** (ảnh 29, 45–54) xem LLM có lỡ sửa mất lỗi của sinh viên không.
- [ ] (Đã tạm bỏ) Kiểm tra đạo code giữa các bài trong cùng contest. Nếu làm lại, nên làm bằng thuật toán so khớp
      token/AST (kiểu MOSS) thay vì gọi LLM cho từng cặp bài, vì số cặp tăng theo bình phương số bài.
- [ ] Endpoint hiện chưa yêu cầu đăng nhập (cho dễ demo). Khi cần, thêm `dependencies=[Depends(AuthenticateRequired())]` vào `APIRouter(...)` trong `api_llm.py` (class có sẵn ở `app/utils/login_manager.py`).
