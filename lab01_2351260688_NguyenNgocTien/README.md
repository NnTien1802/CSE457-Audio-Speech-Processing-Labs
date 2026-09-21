# CSE457 — Lab 1 với datalab1.mp3

- [Notebook đã chạy A–G](CSE457_Lab1_VSCode_datalab1.ipynb)
- [Báo cáo Markdown, hình, số liệu và 7 câu trả lời](report_Lab01.md)
- [Kết quả kiểm tra và tham số](Lab01_output/verification.json)

## Chạy lại

1. Mở thư mục này trong VS Code. Notebook và `datalab1.mp3` phải nằm cùng thư mục.
2. Dùng Python 3.10 hoặc 3.11, cài thư viện bằng `python -m pip install -r requirements.txt`.
3. Chọn kernel tương ứng trong notebook và bấm **Run All**.
4. Xem `report_Lab01.md`; hình, bảng CSV và WAV được tạo trong `Lab01_output/`.

Notebook chứa trình phát các đoạn so sánh với `normalize=False`. Giữ cùng âm lượng thiết bị khi nghe. Nhận xét về cảm nhận trong báo cáo là dự đoán theo phép đo, chưa phải kết quả khảo sát người nghe.

Notebook là mã nguồn chính và chạy độc lập. Dữ liệu đầu vào ở `datalab1.mp3`; toàn bộ đầu ra cần giữ nằm trong `Lab01_output/`.

Các tệp phụ được chuyển vào [luu_tru_khong_nop/](luu_tru_khong_nop/README.md): mã Python trùng lặp, đánh giá cũ, bản sao notebook trước sửa, công cụ kiểm tra và cấu hình VS Code. Thư mục này không cần cho việc chạy bài hoặc nộp bài, và đã được loại khỏi Git bằng `.gitignore`.
