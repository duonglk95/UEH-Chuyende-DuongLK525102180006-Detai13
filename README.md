# Đề tài 13 – Đánh giá rủi ro tín dụng BNPL

Chuyên đề ML/DS/AI – Master of IDT, UEH (2026 S2). Học viên: Dương Lê Khắc (525102180006).

Dữ liệu: [Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) (Kaggle).

## Cấu trúc
- `Detai_13/SourceCode/utils.py`: hàm dùng chung (đọc dữ liệu, gộp bảng, Gini stability, bootstrap, DeLong, calibration, nhóm hành vi).
- `Detai_13/SourceCode/01` đến `09`: từng bước: EDA, đặc trưng, mô hình mất cân bằng, scorecard & hạn mức, cảnh báo sớm, kiểm định, nộp bài, rủi ro "ngoài dấu hiệu rõ ràng", thử nghiệm bảng phụ.
- `Detai_13/SourceCode/Kaggle_All_In_One.ipynb`: Pipeline chạy trên Kaggle.
- `Detai_13/BaoCao/`: báo cáo Word (`BaoCao_Detai13.docx`), tóm tắt kết quả, tổng quan nghiên cứu.
- `Detai_13/Data/README.txt`: nguồn và cách lấy dữ liệu.

## Thiết kế chính
- Chia theo thời gian (WEEK_NUM): train 0–63, valid 64–77, test 78–91; kiểm định cửa sổ trượt.
- So sánh LR / RF / LightGBM với các cách xử lý mất cân bằng; hiệu chỉnh isotonic; scorecard PDO 20.
- Lọc rò rỉ: bỏ cột ngày tương lai, cột thay đổi tỉ lệ thiếu dữ liệu giữa các giai đoạn, và thuộc tính nhạy cảm (giới tính).

## Chạy
Mở `Kaggle_All_In_One.ipynb` trên Kaggle, thêm bộ dữ liệu Home Credit làm Input, Run All (khoảng 2 giờ 30 phút trên CPU).
