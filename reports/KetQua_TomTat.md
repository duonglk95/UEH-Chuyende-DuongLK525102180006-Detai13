# Kết quả Đề tài 13 – Version 5 trên Kaggle (toàn bộ dữ liệu, 2 giờ 16 phút)

Nguồn: đầu ra các ô code của notebook "Topic 13 DLK", Version 5 (Internet tắt, chạy lại toàn bộ). Nhãn `target` là dữ liệu thật của Home Credit; chỉ dùng biến dẫn xuất từ cột thật (gộp bảng, đổi ngày thành số ngày, 3 tỷ lệ, gộp credit bureau). Chia theo thời gian: train tuần 0-63 (1.282.082 hồ sơ, vỡ nợ 3,32%), valid 64-77 (92.870; 2,43%), test 78-91 (151.707; 2,12%).

## 1. Độ chính xác trên tập test, khoảng tin cậy bootstrap 95% (500 lần)

| Mô hình | AUC | CI95 | Gini | Gini stability* |
|---|---|---|---|---|
| **LightGBM gốc + credit bureau (376 biến) – mô hình cuối** | **0,8711** | [0,8660; 0,8766] | **0,7423** | **0,7180** |
| LightGBM sqrt(scale_pos_weight) | 0,8507 | [0,8450; 0,8569] | 0,7015 | 0,6799 |
| LightGBM không xử lý mất cân bằng | 0,8504 | [0,8445; 0,8567] | 0,7009 | 0,6790 |
| LightGBM scale_pos_weight | 0,8492 | [0,8433; 0,8556] | 0,6984 | 0,6773 |
| Random Forest class_weight | 0,7997 | [0,7923; 0,8068] | 0,5995 | 0,5622 |
| Random Forest | 0,7968 | [0,7892; 0,8038] | 0,5935 | 0,5568 |
| Logistic Regression (WoE) – các biến thể | 0,7924 – 0,7953 | | 0,585 – 0,591 | 0,547 – 0,553 |

*Chỉ số của cuộc thi: trung bình Gini theo tuần + 88 × min(0, độ dốc) − 0,5 × độ lệch chuẩn phần dư.

- Kiểm định DeLong: mô hình cuối tốt hơn mọi mô hình còn lại có ý nghĩa thống kê (so với LightGBM đặc trưng gốc: chênh AUC 0,0207, z = 12,86, p ≈ 7×10⁻³⁸; so với Logistic Regression: chênh ≈ 0,077, z ≈ 25,7).
- Xử lý mất cân bằng (class weight, undersampling, SMOTE, scale_pos_weight) không cải thiện khả năng xếp hạng của bất kỳ mô hình nào; chỉ làm lệch xác suất.

## 2. Độ ổn định theo thời gian – kiểm định cửa sổ thời gian (khoảng đệm 2 tuần)

| Cửa sổ | Train | Kiểm định | AUC | Gini tuần TB |
|---|---|---|---|---|
| Mở rộng | 0-37 | 40-51 | 0,8324 | 0,662 |
| Mở rộng | 0-49 | 52-63 | 0,8354 | 0,668 |
| Mở rộng | 0-61 | 64-77 | 0,8536 | 0,694 |
| Mở rộng | 0-75 | 78-91 | 0,8708 | 0,735 |
| Trượt 40 tuần | 10-49 / 22-61 / 36-75 | 52-63 / 64-77 / 78-91 | 0,8325 / 0,8520 / 0,8668 | |

- Cửa sổ mở rộng: AUC trung bình 0,8481 ± 0,0178 (thấp nhất 0,8324); cửa sổ trượt: 0,8459 ± 0,0167. Dữ liệu cũ vẫn có ích (cửa sổ mở rộng nhỉnh hơn) → chưa cần bỏ dữ liệu cũ khi huấn luyện lại.
- Gini tuần không có xu hướng giảm (độ dốc +0,001).

## 3. Hiệu chỉnh xác suất (mô hình cuối, isotonic fit trên valid)

| | ECE | Hệ số độ dốc | Hệ số chặn | PD TB | Vỡ nợ thực |
|---|---|---|---|---|---|
| Trước hiệu chỉnh | 0,0153 | 1,139 | −0,288 | 3,65% | 2,12% |
| Sau isotonic | **0,0039** | 1,061 | −0,027 | 2,51% | 2,12% |

Hạng rủi ro (test, mô hình cuối) giảm đơn điệu: E 11,26% | D 2,69% | C 0,72% | B 0,12% | A 0,02%.

## 4. Chính sách hạn mức (mô hình đặc trưng gốc, giả định biên 6%, LGD 60%)

Duyệt 88,3%, vỡ nợ nhóm duyệt 1,05% (toàn bộ 2,12%); lợi nhuận theo hạn mức từng hạng 62,0 tỷ VND trên tổng hạn mức 1.089,9 tỷ; ngưỡng PD tối ưu (valid) 0,079 → test duyệt 93,4%, vỡ nợ 1,32%. Gini tuần TB 0,694, PSI train → test 0,034.

## 5. Kiểm tra rò rỉ và trôi dữ liệu

- **Hoán vị nhãn**: AUC test 0,5065 (kỳ vọng 0,5) → quy trình không rò rỉ nhãn.
- **Adversarial validation (train vs test): AUC = 1,00** → có biến phân biệt hoàn toàn hai giai đoạn. Nguyên nhân chính là các cột chỉ có dữ liệu ở giai đoạn sau (gain adversarial: `description_5085714M` 47,0%, `contractssum_5085716L` 28,1%, `requesttype_4525192L` 5,7%, `birthdate_574D` 2,3%...); 14 biến PSI > 0,25. Mô hình vẫn ổn định trên test, nhưng đây là rủi ro khi triển khai → **đã sửa ở bản v6**: loại cột có chênh lệch tỷ lệ thiếu dữ liệu giữa train và valid ≥ 30 điểm phần trăm (không dùng tập test để chọn).

## 6. Thuộc tính nhạy cảm (giới tính)

| | AUC test | Tỷ lệ duyệt nữ | Tỷ lệ duyệt nam | Vỡ nợ nhóm duyệt nữ / nam |
|---|---|---|---|---|
| Có giới tính | 0,8711 | 92,2% | 85,9% | 0,89% / 1,19% |
| Không giới tính | 0,8692 | 90,5% | 89,0% | 0,82% / 1,35% |

Bỏ giới tính làm AUC giảm 0,0019 (DeLong p = 0,012: có ý nghĩa thống kê nhưng rất nhỏ) và thu hẹp chênh lệch tỷ lệ duyệt từ 6,3 xuống 1,5 điểm phần trăm → **bản v6 không dùng giới tính**.

## 7. Kiểm định độc lập trên tập test ẩn của Kaggle

Late Submission Version 5 (nộp 08:59 ngày 23/11/2026) **lỗi Out of Memory** khi Kaggle chạy lại trên tập test ẩn: notebook gộp giữ dữ liệu của mọi phần trước trong bộ nhớ rồi mới tạo đặc trưng cho tập test ẩn (lớn hơn nhiều so với 10 dòng mẫu). Bản v6 đã sửa: mỗi phần giải phóng bộ nhớ của phần trước, phần tạo `submission.csv` đặt cuối cùng và dự đoán theo khối 100.000 hồ sơ. Chưa có điểm public/private.

## 8. Bản v6 (chờ chạy)

Loại biến trôi độ sẵn có; bỏ giới tính; thêm notebook 08 – đặc điểm đi cùng rủi ro cao ngoài các dấu hiệu túng thiếu hiển nhiên (mô hình đối chứng, SHAP, rủi ro quan sát/kỳ vọng O/E với kiểm định z, phân khúc rủi ro ẩn ở khách "trông an toàn").
