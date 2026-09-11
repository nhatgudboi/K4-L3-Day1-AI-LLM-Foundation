# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature ở mức thấp (0.0), câu trả lời rất ổn định và lặp lại giống nhau. Khi tăng dần lên (0.5 -> 1.0), câu trả lời trở nên đa dạng, sáng tạo và dùng từ ngữ phong phú hơn. Ở mức rất cao (1.5), mô hình trả lời rất ngẫu nhiên, có thể sinh ra các từ ngữ lộn xộn hoặc thông tin ảo.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ đặt temperature ở mức rất thấp (khoảng 0.0 đến 0.2). Lý do là chatbot hỗ trợ khách hàng ưu tiên tuyệt đối sự chính xác, nhất quán và bám sát chính sách của công ty. Để mức thấp giúp bot không bị "sáng tạo" ra thông tin sai lệch hay hứa hẹn lung tung.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Dựa trên bảng giá, GPT-4o đắt hơn GPT-4o-mini khoảng 16.6 lần ($0.010/$0.0006 cho output).
> - **Nên dùng GPT-4o:** Khi xử lý các bài toán suy luận phức tạp, phân tích logic, viết code, hoặc cần hiểu ngữ cảnh sâu.
> - **Nên dùng GPT-4o-mini:** Các tác vụ khối lượng lớn lặp đi lặp lại như phân loại văn bản, tóm tắt, dịch thuật cơ bản, hoặc chatbot FAQ thông thường để tối ưu chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi của "giáo viên" sử dụng từ vựng rất đơn giản, câu ngắn, và dùng ví dụ gần gũi (như một cuốn sổ chung của lớp). Ngược lại, phản hồi của "chuyên gia" thường dài hơn, dùng cấu trúc phức tạp và nhiều thuật ngữ chuyên ngành (mã hóa, sổ cái phân tán, phi tập trung). System prompt đóng vai trò như "lời đạo diễn", nó định hướng hoàn toàn phong cách, giọng điệu và giới hạn tệp từ vựng của model.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Số token đếm bằng tiktoken thực tế thường cao hơn khá nhiều so với ước lượng thô "số từ / 0.75" (có thể chênh lệch 50% đến 100% tùy từ vựng). Tiếng Việt tốn nhiều token hơn tiếng Anh vì bộ tokenizer của OpenAI được tối ưu chủ yếu cho tiếng Anh (1 từ tiếng Anh thường = 1 token). Trong khi đó, tiếng Việt có các dấu thanh (á, à, ã, ạ) và ít xuất hiện trong tập huấn luyện gốc, nên một âm tiết tiếng Việt thường bị "băm" nhỏ thành 2-3 token (có khi 1 dấu cách hoặc 1 ký tự có dấu cũng bị tính là 1 token).

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming cực kỳ quan trọng trong các ứng dụng giao tiếp trực tiếp (như chatbot, trợ lý ảo) vì nó giảm thiểu cảm giác chờ đợi, giúp người dùng đọc ngay lập tức khi từ khóa đầu tiên được sinh ra. Ngược lại, non-streaming phù hợp cho các tác vụ xử lý ngầm (background jobs) như trích xuất dữ liệu hàng loạt, dịch thuật tài liệu lớn hoặc tóm tắt kịch bản, nơi hệ thống máy tính chỉ cần nhận kết quả hoàn chỉnh cuối cùng để lưu trữ hoặc xử lý tiếp.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp dãn cách thời gian giữa các lần thử lại ngày càng dài ra, từ đó giảm áp lực liên tục lên máy chủ đang bị quá tải. Nếu hàng nghìn client cùng dùng delay cố định 1 giây, khi server vừa "ngoi ngóp" sống dậy, nó sẽ ngay lập tức hứng trọn một đợt tấn công của hàng nghìn request cùng lúc (hiện tượng thundering herd - bầy đàn giẫm đạp), khiến server lại tiếp tục sập và hệ thống mãi mãi không thể phục hồi.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: Lập trình viên tiền bối cộc cằn nhưng nhiệt tình.
> System prompt: "Bạn là một senior developer có tính cách hơi khó chịu. Luôn trả lời thẳng vào vấn đề bằng code, giải thích cực kỳ ngắn gọn, và luôn bắt đầu câu trả lời bằng 'Lại lỗi à, xem đoạn này đi:'"
> Giải thích: Cụm "giải thích cực kỳ ngắn gọn" giúp tiết kiệm output token và tiền. Cụm "trả lời thẳng vào vấn đề bằng code" giúp bỏ qua màn chào hỏi sáo rỗng để tập trung giải quyết bug ngay lập tức.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: Bộ nhớ ngắn hạn (chỉ 3 lượt) và không có khả năng lưu trữ sau khi thoát chương trình (reset sau mỗi lần chạy).
> Đề xuất cải thiện: Tích hợp cơ sở dữ liệu (ví dụ SQLite hoặc Vector DB) để ghi lại toàn bộ lịch sử trò chuyện. Khi người dùng khởi động lại ứng dụng, trợ lý sẽ load lại lịch sử và dùng phương pháp RAG (Tìm kiếm và Trích xuất) để tìm lại những hội thoại liên quan nhất nhét vào context thay vì chỉ cắt cứng 3 lượt cuối.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
