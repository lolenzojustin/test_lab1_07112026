# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`= "cab" ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= "ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
# Record này nói về cả bức ảnh traffic, vì task là image_classification và sử dụng model yolo11n-cls.pt để phân loại toàn ảnh. Lớp cab có rank = 1, score = 0.510915 là dự đoán được xếp hạng cao nhất cho ảnh; record không có tọa độ hộp bao để xác định một chiếc taxi riêng trong ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
# Danh sách tên lớp có sẵn, đi kèm taxonomy/checkpoint, không phải model tự nghĩ ra. Trong bài này, checkpoint yolo11n-cls.pt sử dụng taxonomy ImageNet-1K gồm 1.000 lớp. Model dự đoán và xếp hạng các lớp trong danh sách đó cho ảnh đầu vào.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
# Cần ghi đủ vì class_id xác định mã lớp trong bộ nhãn, class_name giúp người đọc hiểu tên lớp, còn taxonomy_name cho biết đang dùng bộ nhãn nào. Các bộ nhãn có thể dùng mã khác nhau cho cùng một tên lớp hoặc cùng một mã cho các lớp khác nhau, nên ghi đủ sẽ tránh nhầm lẫn khi đối chiếu kết quả.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
# Guideline cần nêu rõ cách chọn chủ thể chính, ví dụ ưu tiên vật nổi bật nhất, chiếm diện tích lớn nhất hoặc phù hợp mục tiêu bài toán. Nếu có nhiều chủ thể ngang nhau, cần có quy tắc ưu tiên; trường hợp vẫn không rõ thì hỏi người phụ trách.
6. Vì sao không được lấy `score` của model làm ground truth?
# Vì score chỉ thể hiện mức độ tự tin của model, điểm cao vẫn có thể dự đoán sai. Ground truth phải là nhãn do con người gán và kiểm tra theo guideline, không thể lấy điểm của model để xác nhận nhãn đúng.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person ; `score`= 0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`= 113.58 ; `bbox_height`= 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
# Góc trên-trái của box ở (385.33, 69.24), góc dưới-phải ở (498.92, 348.92), tính từ góc trên-trái của ảnh. Box rộng 113.58 pixel và cao 279.68 pixel theo JSON.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
# Chưa xác định được từ record này. Cần xem kết quả ô so sánh ngưỡng 0.20 / 0.35 / 0.60 trong notebook để lấy số prediction tại từng ngưỡng.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
# Reviewer phải kiểm tra thêm các hộp được giữ lại, đặc biệt là dự đoán sai lớp, hộp trùng hoặc hộp bao không đúng vật thể.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
# Vẽ box sát phần vật thể nhìn thấy, bao đủ phần đó và hạn chế lấy thừa nền; không tự đoán phần bị che khuất nếu guideline chưa quy định.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
# Ở mép trái ảnh kitchen có một người chỉ lộ một phần tay. Tôi sẽ hỏi Lab Coach xem trường hợp này có cần gán nhãn person và vẽ box thế nào, vì phần nhìn thấy quá ít và chưa có quy tắc rõ ràng.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-001 ; `class_name`= person ; `score`= 0.899318 ; số điểm polygon=`polygon_point_count`= 348 ; 3–5 điểm đầu của `polygon_xy`= [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
# Polygon mô tả đường biên và vùng vật thể chiếm trong ảnh, thể hiện hình dạng chi tiết hơn box. Ví dụ với người trong ảnh, polygon bám theo đầu, thân, tay và chân, còn box chỉ là hình chữ nhật bao quanh.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
# instance_id dùng để phân biệt từng đối tượng trong bộ kết quả lab. Hai người cùng lớp person vẫn có instance_id riêng. Nó không phải class ID vì không đại diện cho loại vật thể, và không phải tracking ID vì không dùng để theo dõi cùng một vật thể qua nhiều khung hình.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
# Vẽ mask sát đường biên phần vật thể nhìn thấy, không lấy nền hoặc vật khác vào mask; không tự vẽ thêm phần bị che nếu guideline chưa quy định.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
# Trên bàn phía trước ảnh kitchen, các bát nằm sát nhau và che một phần mặt bàn. Tôi áp dụng guideline chỉ vẽ phần nhìn thấy: mask bàn cần loại phần các bát che lên, còn mỗi bát có mask riêng. Nếu guideline chưa quy định cách xử lý phần bị che, tôi sẽ hỏi Lab Coach trước khi gán nhãn.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):
| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 	Một nhãn lớp cho toàn ảnh, được chọn theo guideline. | Ảnh traffic có nhiều loại xe nhưng model xếp cab cao nhất, score 0.510915; khó chọn một lớp đại diện. |Chọn lớp theo quy tắc ưu tiên chủ thể, hỏi Lab Coach nếu chưa rõ.  | Kiểm tra lớp có đúng taxonomy và quy tắc chọn chủ thể; không lấy score làm bằng chứng nhãn đúng. |
| Phát hiện vật thể | Mỗi vật thể có một nhãn lớp và box bao quanh theo guideline. | Trong detection_predictions.png, người ở mép trái chỉ lộ một phần tay nhưng được dự đoán là person khoảng 0.61. | Áp dụng quy tắc về vật bị cắt mép; hỏi Lab Coach nếu chưa rõ có cần gán nhãn. | Kiểm tra đúng lớp, box sát vật thể, có bỏ sót hoặc trùng hộp không. |
| Instance segmentation |Mỗi đối tượng có nhãn lớp và mask/polygon riêng.  | segmentation_prediction.png, các bát trên bàn nằm sát nhau, che mặt bàn nên khó xác định biên từng vùng. |Vẽ riêng từng đối tượng, bám phần biên nhìn thấy theo guideline; hỏi khi không rõ.  | Kiểm tra mask có ăn sang nền/vật khác, thiếu vùng hoặc gộp nhầm các đối tượng không; yêu cầu sửa nếu sai. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
# Chỉ sử dụng các ảnh COCO công khai được cung cấp trong notebook, giữ thông tin nguồn ảnh và không đưa ảnh cá nhân, dữ liệu nội bộ hoặc thông tin nhạy cảm lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
#  Tôi sẽ dừng xử lý, không tải lên hoặc chia sẻ dữ liệu đó và báo cho Lab Coach hoặc giảng viên để được hướng dẫn

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
