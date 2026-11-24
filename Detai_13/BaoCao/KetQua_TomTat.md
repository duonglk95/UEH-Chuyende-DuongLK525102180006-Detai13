# Kết quả Đề tài 13 – Version 8 trên Kaggle (bản v6.2, toàn bộ dữ liệu, 2 giờ 27 phút)

Nguồn: đầu ra các ô code của notebook "Topic 13 DLK", Version 8 (Internet tắt, chạy lại toàn bộ, hoàn tất ngày 24/11/2026). Nhãn `target` là dữ liệu thật của Home Credit; chỉ dùng biến dẫn xuất từ cột thật (gộp bảng, đổi ngày thành số ngày, 3 tỷ lệ, gộp credit bureau). Chia theo thời gian: train tuần 0-63 (1.282.082 hồ sơ, vỡ nợ 3,32%), valid 64-77 (92.870; 2,43%), test 78-91 (151.707; 2,12%).

So với Version 5, bản này đã: loại 20 cột ngày có giá trị sau ngày quyết định (trừ cột hạn trả/đáo hạn), loại 21 cột có tỷ lệ thiếu dữ liệu chênh ≥ 30 điểm phần trăm giữa train và valid, bỏ giới tính. Vì vậy AUC giảm nhẹ so với V5 (0,8711 → 0,8689) – đây là cái giá của việc chặn rò rỉ, không phải mô hình kém đi.

## 1. Độ chính xác trên tập test, khoảng tin cậy bootstrap 95%

| Mô hình | AUC | CI95 | Gini | Gini stability* |
|---|---|---|---|---|
| **LightGBM gốc + credit bureau (354 biến) – mô hình cuối** | **0,8689** | [0,8635; 0,8744] | **0,7378** | **0,7153** |
| LightGBM sqrt(scale_pos_weight) | 0,8456 | [0,8395; 0,8518] | 0,6911 | 0,6709 |
| LightGBM scale_pos_weight | 0,8454 | [0,8395; 0,8515] | 0,6908 | 0,6716 |
| LightGBM không xử lý mất cân bằng | 0,8451 | [0,8388; 0,8512] | 0,6901 | 0,6692 |
| Random Forest class_weight | 0,7972 | [0,7900; 0,8045] | 0,5943 | 0,5586 |
| Random Forest | 0,7955 | [0,7883; 0,8034] | 0,5909 | 0,5550 |
| Logistic Regression (WoE) – 4 biến thể | 0,7882 – 0,7888 | | 0,576 – 0,578 | 0,540 – 0,541 |

*Chỉ số của cuộc thi: trung bình Gini theo tuần + 88 × min(0, độ dốc) − 0,5 × độ lệch chuẩn phần dư.

- DeLong: mô hình cuối tốt hơn mọi mô hình còn lại có ý nghĩa thống kê (so với LightGBM đặc trưng gốc: chênh 0,0238, z = 13,99, p ≈ 2×10⁻⁴⁴; so với Logistic Regression: chênh ≈ 0,080, z ≈ 25,8).
- Xử lý mất cân bằng không cải thiện xếp hạng (LightGBM: 0,8451 / 0,8454 / 0,8456; LR: 0,7882 – 0,7888) mà làm lệch xác suất (Brier 0,020 → 0,179 với scale_pos_weight đầy đủ = 29,1).
- Đặc trưng credit bureau (120 biến từ `credit_bureau_a_1`) là đóng góp lớn nhất: AUC valid 0,8287 → 0,8558, test 0,8451 → 0,8689, Gini tuần thấp nhất 0,641 → 0,678.

## 2. Độ ổn định theo thời gian – kiểm định cửa sổ thời gian (khoảng đệm 2 tuần)

| Cửa sổ | Train | Kiểm định | AUC | Gini tuần TB |
|---|---|---|---|---|
| Mở rộng | 0-37 | 40-51 | 0,8316 | 0,660 |
| Mở rộng | 0-49 | 52-63 | 0,8344 | 0,666 |
| Mở rộng | 0-61 | 64-77 | 0,8503 | 0,687 |
| Mở rộng | 0-75 | 78-91 | 0,8657 | 0,725 |
| Trượt 40 tuần | 10-49 / 22-61 / 36-75 | 52-63 / 64-77 / 78-91 | 0,8325 / 0,8501 / 0,8642 | |

- Cửa sổ mở rộng: AUC trung bình 0,8455 ± 0,0158; cửa sổ trượt cho kết quả gần như bằng → dữ liệu cũ vẫn có ích, chưa cần bỏ khi huấn luyện lại.
- Gini tuần của mô hình cuối không giảm theo thời gian (độ dốc +0,0016, độ lệch chuẩn phần dư 0,031).

## 3. Hiệu chỉnh xác suất (mô hình cuối, isotonic fit trên valid)

| | ECE | Hệ số độ dốc | Hệ số chặn | PD TB | Vỡ nợ thực |
|---|---|---|---|---|---|
| Trước hiệu chỉnh | 0,0151 | 1,110 | −0,350 | 3,63% | 2,12% |
| Sau isotonic | **0,0043** | 1,056 | −0,056 | 2,55% | 2,12% |

Hạng rủi ro (test, mô hình cuối) giảm đơn điệu: E 10,95% | D 2,80% | C 0,69% | B 0,14% | A 0,02%.

## 4. Scorecard và chính sách hạn mức (mô hình đặc trưng gốc; biên 6%, LGD 60%)

| Hạng | Điểm | Tỷ trọng | Vỡ nợ | Quyết định | Hạn mức | Lợi nhuận (tỷ VND) |
|---|---|---|---|---|---|---|
| A | 684 – 753 | 6,4% | 0,09% | Duyệt | 20 triệu | 11,6 |
| B | 634 – 664 | 30,8% | 0,20% | Duyệt | 12 triệu | 32,9 |
| C | 597 – 633 | 27,1% | 0,84% | Duyệt | 6 triệu | 13,4 |
| D | 573 – 595 | 22,0% | 2,57% | Duyệt hạn chế | 2 triệu | 2,9 |
| E | 447 – 573 | 13,7% | 9,22% | Từ chối | 0 | 0 |
| **Tổng** | | | 2,12% | | 1.069,6 tỷ | **60,8** |

- Tối ưu ngưỡng PD trên valid (0,087) rồi áp lên test: duyệt 92,6%, vỡ nợ nhóm duyệt 1,28%, lợi nhuận 61,3 tỷ. Với hạn mức cố định 6 triệu: 43,4 tỷ → hạn mức theo hạng mang lại thêm ~18 tỷ.
- Gini tuần TB 0,684 (thấp nhất 0,643); PSI điểm dự đoán train → test 0,018 (ổn định).

## 5. Dấu hiệu hành vi trước vỡ nợ (notebook 05)

| Nhóm hành vi | Số biến | AUC riêng nhóm (test) | Độ phủ cờ | Vỡ nợ khi có cờ / không | Lift |
|---|---|---|---|---|---|
| Trễ hạn | 69 | 0,779 | 21,7% | 6,02% / 1,04% | 2,84 |
| Vay nhiều nơi | 74 | 0,740 | 18,0% | 5,63% / 1,35% | 2,66 |
| Gánh nặng nợ | 42 | 0,667 | 21,8% | 4,05% / 1,58% | 1,91 |
| Đảo nợ / gia hạn | 3 | 0,567 | 14,0% | 4,04% / 1,81% | 1,91 |

Tỷ trọng gain của mô hình cuối: trễ hạn 28,8%, vay nhiều nơi 19,4%, gánh nặng nợ 9,0%, đảo nợ 0,5%, biến khác 42,4%. Dữ liệu chỉ có 3 biến về gia hạn hợp đồng (`prolongationcount`) nên nhóm "đảo nợ" chỉ đo được gián tiếp.

## 6. Ngoài các dấu hiệu túng thiếu hiển nhiên (notebook 08)

- Mô hình chỉ dùng 188 biến "hiển nhiên" (trễ hạn, vay nhiều nơi, gánh nặng nợ, đảo nợ): AUC 0,8530; thêm 166 biến còn lại lên 0,8689 (chênh 0,0159, z = 11,1, p ≈ 2×10⁻²⁸). Biến không hiển nhiên chiếm 47,3% tỷ trọng SHAP.
- Đặc điểm đi cùng rủi ro (O/E = quan sát / kỳ vọng theo tỷ lệ chung, kiểm định z): nhóm hưu trí (O/E 0,68, z = −10,0) và công chức (0,78) rủi ro thấp hơn mức chung; người làm thuê "EMPLOYED" cao hơn (1,10, z = 2,9). Rủi ro cao hơn ở người trẻ nhất (O/E 1,08) và ở người đã vay khoản giá trị lớn (price cao nhất: 1,03). Số người dùng chung số điện thoại di động, sản phẩm bán chéo, loại điểm bán và số kỳ thanh toán cũng có ý nghĩa.
- Tuổi và tình trạng hôn nhân được đánh dấu là thuộc tính nhạy cảm: chỉ báo cáo, khuyến nghị không dùng để từ chối.
- Nhóm "trông an toàn" (60% hồ sơ, 99.826 ở test) có tỷ lệ vỡ nợ 0,45%; cây quyết định tìm rủi ro ẩn trong nhóm này chỉ đạt AUC 0,607 → rủi ro còn lại phân tán, không có phân khúc ẩn rõ rệt.

## 7. Kiểm tra rò rỉ và trôi dữ liệu

- **Hoán vị nhãn**: AUC test 0,4548 (kỳ vọng ~0,5) → không rò rỉ nhãn.
- **Adversarial validation (train vs test): AUC vẫn = 1,00.** Biến phân biệt chính là `description_5085714M` (62,3% gain, phân loại khách của credit bureau), `applprev_1__postype_4733339M__first` (12,5%), `person_1__safeguarantyflag_411L__mean` (9,8%) – các cột phân loại có mã mới xuất hiện ở giai đoạn sau, không bị bộ lọc tỷ lệ thiếu dữ liệu bắt được. 14 biến PSI > 0,25. Dù vậy, điểm dự đoán ổn định (PSI 0,018) và Gini tuần không giảm → trôi ở cấp biến không làm hỏng dự đoán, nhưng cần theo dõi khi triển khai. Hướng tiếp: lọc thêm theo PSI/adversarial trên valid.

## 8. Thử nghiệm bảng phụ (notebook 09)

Thêm bảng thuế (`tax_registry_a/b/c_1`) và `applprev_2`: 18 biến mới, bộ lọc chống rò rỉ loại 12, giữ 6. AUC 0,8689 → 0,8695 (DeLong z = 1,16, p = 0,246) → **không cải thiện có ý nghĩa**, nên không đưa vào mô hình nộp bài.

## 9. Kiểm định độc lập trên tập test ẩn của Kaggle

Version 8 chạy xong trên dữ liệu mẫu (submission.csv 10 dòng) và đã được nộp Late Submission lúc 12:25 ngày 24/11/2026; Kaggle đang chạy lại trên tập test ẩn. Version 5 trước đó lỗi Out of Memory; bản v6.2 đã giải phóng bộ nhớ từng phần, đặt phần nộp bài cuối cùng và dự đoán theo khối. Điểm public/private: chờ cập nhật.
