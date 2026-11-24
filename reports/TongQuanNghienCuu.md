# Tổng quan nghiên cứu và định vị kết quả (bản nháp)

> Bản nháp để học viên đọc, kiểm tra nguồn và viết lại bằng lời của mình. Các con số của đề tài lấy từ Version 5 (mô hình cuối, tập test tuần 78–91).

## 1. Mô hình học máy so với Logistic Regression trong chấm điểm tín dụng

Các nghiên cứu benchmark lớn (Baesens và cộng sự, 2003; Lessmann và cộng sự, 2015) là mốc tham chiếu của lĩnh vực. Lessmann và cộng sự kết luận nhiều bộ phân loại dự báo vỡ nợ chính xác hơn đáng kể so với Logistic Regression – chuẩn của ngành – và khuyến nghị dùng Random Forest làm mốc so sánh. Một tổng quan gần đây lưu ý nhiều kết quả công bố phóng đại khoảng cách giữa gradient boosting và Logistic Regression vì chỉ mô hình boosting được tinh chỉnh tham số.

*Định vị:* đề tài cho kết quả cùng chiều (LightGBM AUC 0,871 so với Logistic Regression 0,794–0,795; DeLong z ≈ 25,7, p xấp xỉ 0). Do Logistic Regression của đề tài chưa được tinh chỉnh (C = 0,1 cố định, 60 biến theo IV), khoảng cách này cần được diễn giải thận trọng – đây là một hạn chế cần nêu.

## 2. Xử lý dữ liệu mất cân bằng

Một benchmark 23 phương pháp trên dữ liệu tín dụng mất cân bằng (arXiv:1907.12996) cho thấy các phương pháp ensemble như stochastic gradient boosting và random forest hoạt động tốt ngay cả khi mất cân bằng nặng. Nghiên cứu PSD2 (arXiv:2011.10367) cho thấy SMOTE/BorderlineSMOTE không cải thiện, thậm chí làm giảm, hiệu quả của Gradient Boosting.

*Định vị:* đề tài xác nhận kết luận này trên dữ liệu thực quy mô lớn: class weight, undersampling, SMOTE và scale_pos_weight không cải thiện AUC của bất kỳ mô hình nào (khác biệt ≤ 0,003), chỉ làm lệch xác suất (Brier tăng từ 0,020 lên 0,17–0,20). Đề tài còn chỉ ra một lỗi phương pháp dễ gặp: kết quả kém của scale_pos_weight (AUC 0,72) là do early stopping theo dõi cả logloss, không phải do trọng số.

## 3. Giá trị của dữ liệu tín dụng bên ngoài (credit bureau)

Liu & Liang (Journal of Risk Model Validation) cho thấy trong cho vay thay thế, điểm tín dụng đơn lẻ chỉ đạt AUC khoảng 0,6; thêm biến khoản vay và hơn 300 biến credit bureau chọn lọc cải thiện đáng kể. Một nghiên cứu khác trên doanh nghiệp nhỏ ghi nhận AUC tăng từ khoảng 0,78 lên 0,83 khi tích hợp dữ liệu tín dụng cá nhân của chủ doanh nghiệp; các lần tra cứu tín dụng gần đây nằm trong nhóm yếu tố dự báo mạnh nhất. Đội đứng thứ 5 cuộc thi Home Credit 2018 (deepsense.ai) báo cáo AUC riêng từng bảng: hồ sơ vay trước 0,63, bureau 0,61, lịch sử trả góp 0,58.

*Định vị:* kết quả của đề tài nhất quán: thêm 120 đặc trưng gộp từ `credit_bureau_a_1` tăng AUC test từ 0,850 lên 0,871 (DeLong z = 12,9; p ≈ 7×10⁻³⁸) và Gini theo tuần từ 0,694 lên 0,736; nhóm trễ hạn (30,6% gain) và vay nhiều nơi (18,1%) đóng góp lớn nhất.

## 4. Tra cứu tín dụng, vay chồng (loan stacking) và dấu hiệu túng thiếu

Báo cáo của Ủy ban Cải cách Pháp luật Úc (ALRC) ghi nhận nhiều lần tra cứu có thể chỉ là khách hàng so sánh ưu đãi, nhưng cũng có thể phản ánh việc bị tổ chức khác từ chối; đa số đồng thuận rằng số lần tra cứu cao gắn với vỡ nợ. Experian mô tả vay chồng xảy ra vì khoảng thời gian từ duyệt đến giải ngân ngắn hơn thời gian khoản vay được báo cáo lên trung tâm tín dụng. Equifax giới thiệu thước đo "Credit Abuse Risk" nhằm phát hiện rủi ro ẩn trong các nhóm "trông an toàn".

*Định vị:* notebook 05 lượng hóa các tín hiệu này (lift ≈ 2,7 cho trễ hạn và vay nhiều nơi; 9,9% vỡ nợ – gấp 4,7 lần mức chung – khi ba dấu hiệu cùng xuất hiện); notebook 08 đi xa hơn, tìm rủi ro **vượt** mức mà các dấu hiệu túng thiếu giải thích được (O/E, kiểm định z) và phân khúc rủi ro ẩn ở khách "trông an toàn" – tương tự ý tưởng của Equifax nhưng minh bạch về phương pháp.

## 5. Các nghiên cứu trên cùng bộ dữ liệu Home Credit – Credit Risk Model Stability (2024)

- Nhóm Erdős Institute (Medina & Kwong) so sánh XGBoost, MLP và LightGBM; LightGBM cho độ ổn định tốt nhất; dùng thống kê DPD có trọng số.
- Một tiền bài báo (Nayak, arXiv:2603.06733) dùng chia theo thời gian, thêm bảng thuế và thông tin cá nhân, báo cáo AUC-ROC 0,912 cho mô hình CCI; tuy nhiên không công bố số liệu mô hình cơ sở và chưa qua bình duyệt, cũng không mô tả việc loại cột ngày sau quyết định – nên chưa thể dùng làm mốc so sánh trực tiếp.
- Một số báo cáo sinh viên công bố AUC ≥ 0,97 trên các tập Home Credit; mức này khó đạt được nếu không có rò rỉ dữ liệu.

*Định vị:* với chia theo thời gian, loại cột phát sinh sau quyết định và cột trôi độ sẵn có, AUC 0,85–0,87 là mức hợp lý. Kiểm tra hoán vị nhãn (AUC 0,507) cho thấy quy trình không rò rỉ nhãn.

## 6. Đóng góp riêng của đề tài

1. **Kiểm định ổn định và ý nghĩa thống kê**: khoảng tin cậy bootstrap, kiểm định DeLong cho mọi cặp mô hình, chỉ số ổn định Gini của cuộc thi, kiểm định cửa sổ thời gian (mở rộng và trượt, 4 khối, khoảng đệm 2 tuần).
2. **Phát hiện và loại biến trôi độ sẵn có** nhờ adversarial validation (AUC = 1,0 trước khi xử lý) – một rủi ro triển khai ít được nêu trong các lời giải cuộc thi.
3. **Đánh đổi độ chính xác – công bằng**: bỏ giới tính làm AUC giảm 0,0019 (p = 0,012) nhưng thu hẹp chênh lệch tỷ lệ duyệt nữ/nam từ 6,3 xuống 1,5 điểm phần trăm.
4. **Tín hiệu cảnh báo sớm và rủi ro vượt kỳ vọng**: lượng hóa hành vi trước vỡ nợ và đặc điểm đi cùng rủi ro cao ngoài các dấu hiệu túng thiếu hiển nhiên.

## 7. Hạn chế và hướng mở rộng

- Logistic Regression chưa tinh chỉnh; chưa thử ensemble nhiều mô hình.
- Chưa dùng `credit_bureau_a_2` (lịch sử theo tháng của hợp đồng bên ngoài) do giới hạn bộ nhớ; bảng thuế và `applprev_2` được thử ở notebook 09.
- Chỉ có nhãn vỡ nợ; các hành vi trễ hạn/vay chồng được dùng như chỉ báo dẫn trước, không phải nhãn tương lai.
- Tham số lợi nhuận (biên 6%, LGD 60%) là giả định nghiệp vụ.

## Tài liệu tham khảo (cần kiểm tra và định dạng lại theo chuẩn của khoa)

- Baesens, B., et al. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. *Journal of the Operational Research Society*.
- Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*.
- Predicting credit default probabilities using machine learning techniques in the face of unequal class distributions. arXiv:1907.12996. https://arxiv.org/abs/1907.12996
- PSD2 Explainable AI Model for Credit Scoring. arXiv:2011.10367. https://arxiv.org/abs/2011.10367
- Liu, Z., & Liang, H. Enhancing default prediction in alternative lending: leveraging credit bureau data and machine learning. *Journal of Risk Model Validation*. https://www.risk.net/node/7961668
- Journal of Risk Model Validation – small business default with owners' personal credit data. https://www.risk.net/node/7962833
- deepsense.ai. Wait, so loans need to be repaid? The Home Credit risk prediction competition on Kaggle. https://deepsense.ai/wait-so-loans-need-to-be-repaid-the-home-credit-risk-prediction-competition-on-kaggle/
- Australian Law Reform Commission (2010). Credit reporting – inquiry information. https://www.alrc.gov.au/?p=8258
- Harding, G. (2016). Stacking, the invisible threat in online lending. Experian. https://www.experian.com/blogs/insights/2016/07/stacking-and-online-lending/
- Equifax. Credit Abuse Risk whitepaper. https://assets.equifax.com/marketing/US/assets/credit-abuse-risk-whitepaper.pdf
- Medina, B. A., & Kwong, S. H. Home Credit Risk Stability II. Erdős Institute. https://erdosinstitute.org/_files/ugd/cfa4f1_3feaabf1305a40028d506842843e189c.pdf
- Nayak, S. Calibrated Credit Intelligence: Shift-Robust and Fair Risk Scoring with Bayesian Uncertainty and Gradient Boosting. arXiv:2603.06733. https://arxiv.org/abs/2603.06733
- Home Credit Group (2024). Home Credit – Credit Risk Model Stability. Kaggle. https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability
