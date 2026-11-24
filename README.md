# Đề tài 13 – Đánh giá rủi ro tín dụng BNPL

Chuyên đề ML/DS/AI – Master of IDT, UEH (2026 S2). Học viên: Dương Lê Khắc (525102180006).

Dữ liệu: [Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) (Kaggle). Dữ liệu không được đưa vào repo; notebook tự tìm thư mục dữ liệu dưới `/kaggle/input`.

## Cấu trúc
- `src/utils.py` – hàm dùng chung (đọc dữ liệu, gộp bảng, Gini stability, bootstrap, DeLong, calibration, nhóm hành vi).
- `notebooks/01`–`09` – từng bước: EDA, đặc trưng, mô hình mất cân bằng, scorecard & hạn mức, cảnh báo sớm, kiểm định, nộp bài, rủi ro "ngoài dấu hiệu rõ ràng", thử nghiệm bảng phụ.
- `notebooks/Kaggle_All_In_One.ipynb` – bản gộp chạy trọn trên Kaggle (Internet tắt), xuất `submission.csv`.
- `reports/` – tóm tắt kết quả và tổng quan nghiên cứu.

## Thiết kế chính
- Chia theo thời gian (WEEK_NUM): train 0–63, valid 64–77, test 78–91; kiểm định cửa sổ trượt.
- So sánh LR / RF / LightGBM với các cách xử lý mất cân bằng; hiệu chỉnh isotonic; scorecard PDO 20.
- Lọc rò rỉ: bỏ cột ngày tương lai, cột thay đổi tỉ lệ thiếu dữ liệu giữa các giai đoạn, và thuộc tính nhạy cảm (giới tính).

## Chạy
Mở `Kaggle_All_In_One.ipynb` trên Kaggle, thêm dữ liệu cuộc thi, Run All (~2,5 giờ CPU).
