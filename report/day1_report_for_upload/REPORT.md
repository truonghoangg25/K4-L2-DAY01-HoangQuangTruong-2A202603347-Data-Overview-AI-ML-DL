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

1. `class_id`=468; `class_name`=cab; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về cả bức ảnh vì image classification dự đoán một hoặc các class cho toàn bộ ảnh và không cung cấp thông tin vị trí như bounding box hay polygon của từng object.
3. Class list đến từ taxonomy của checkpoint/model, trong trường hợp này là ImageNet-1K, chứ không phải model tự tạo ra class mới.
4. class_id là mã số của class, class_name là tên class, còn taxonomy_name xác định bộ taxonomy mà class đó thuộc về. Ba thông tin này giúp xác định class rõ ràng và tránh nhầm lẫn giữa các taxonomy khác nhau.
5. Guideline cần quy định tiêu chí chọn một class đại diện cho toàn bộ ảnh một cách nhất quán, chẳng hạn dựa trên chủ thể hoặc nội dung chính của ảnh, để các annotator/reviewer đưa ra cùng một quyết định.
6. score là mức độ tin cậy của model đối với prediction, còn Ground Truth là nhãn được xác định theo quy trình/guideline của dataset. Vì vậy score cao không đồng nghĩa prediction chắc chắn đúng và score không thể thay thế Ground Truth.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=bowl; `score`=0.701283; `bbox_xyxy`=[155.02, 168.67, 182.38, 184.18] ; `bbox_width`=27.36 ; `bbox_height`=15.51
2. Lớp bowl, score 0.701283, box từ góc trên-trái từ khoảng pixel(x= 32.65, y=342.12), box từ góc dưới-phải từ khoảng pixel (x=100.16, y=384.93). Rộng khoảng 67.51px và cao khoảng 42.81px
3. Khi threshold=0.20 (20%) thì prediction=18 (nhiều hơn), còn khi thershold=0.60 (60%) thì prediction=6 (ít hơn)
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), reviewer sẽ phải xem thêm nhiều hộp hơn, kể cả hộp nhiễu/score thấp, nên khối lượng kiểm tra tăng.
5. Quy tắc box: Box phải ôm sát vật thể, không cắt mất phần nhìn thấy của object; không vẽ một box chung cho hai object tách rời.
6. Trên ảnh `kitchen`, vật bowl có threshold 0.72 bị cắt mép. Tôi chọn guideline nếu rule đã có (ví dụ “còn ≥ 50% object trong khung thì vẫn gán”); nếu không chắc object là gì vì che khuất nặng thì hỏi Lab Coach, không tự bịa lớp.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1.`instance_id`=kitchen-001; `class_name`=person  `score`=0.899318; số điểm polygon=`polygon_point_count`=348; 3–5 điểm đầu của `polygon_xy`= [[446.0, 70.0], [446.0, 72.0], [445.0, 73.0], [445.0, 74.0], [444.0, 75.0]]
2. So với box, polygon cho biết biên và hình dạng của từng vật thể chi tiết hơn. Bounding box chỉ biểu diễn vật thể bằng một hình chữ nhật bao quanh, còn polygon mô tả đường biên theo nhiều điểm nên có thể bám sát hình dạng thực tế của vật thể hơn.
3. instance_id là mã định danh của một đối tượng cụ thể trong các prediction của ảnh. Nó không phải class_id: class_id xác định loại vật thể, ví dụ person có class_id=0, còn instance_id=kitchen-001 xác định instance cụ thể đó. Nó cũng không phải tracking ID vì JSON này chỉ chứa prediction của ảnh, không thể hiện việc theo dõi cùng một đối tượng qua nhiều frame/thời điểm.
4. Quy tắc vẽ biên mask: vẽ polygon bám theo ranh giới nhìn thấy của vật thể, không lấy phần nền bên ngoài và không tự suy đoán phần bị che khuất nếu không có đủ thông tin để xác định chính xác.
5. Chỗ hai đồ dùng chạm nhau trên bàn kitchen: nếu guideline chưa nói rõ tách biên thế nào thì escalation; nếu guideline đã bảo “tách theo khe nhìn thấy giữa hai object” thì làm theo guideline.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

|-------------|---------------------------------------------------|-------------------------------------------------------|-----------------------|-----------------------|
|   Tác vụ    |   Ground truth trông như thế nào? (đơn vị nhãn)   |   Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON  |   Annotator làm gì?   |   Reviewer xem gì?    |
|-------------|---------------------------------------------------|-------------------------------------------------------|-----------------------|-----------------------|
|Phân loại ảnh|    Một lớp cho cả ảnh theo taxonomy ImageNet-1K   |          Top-1 có thể không khớp trực giác            |  Đọc guideline chọn   | Đúng taxonomy,phạm vi |
|             |                                                   |          cảnh giao thông (prediction ≠ GT)            |  đúng 1 lớp cho ảnh   |  cấp ảnh, không chép  |
|             |                                                   |                                                       |                       |   score thành nhãn    |
|-------------|---------------------------------------------------|-------------------------------------------------------|-----------------------|-----------------------|
|  Phát hiện  |         Mỗi object: class + box xyxy              |     Hạ threshold ra thêm hộp nhiễu; object cắt mép    |    Vẽ box theo rule   |  Box có ôm sát không, |
|   vật thể   |                                                   |                                                       |    đủ object trong    |   thiếu/thừa object,  |
|             |                                                   |                                                       |     guideline         |    ca che khuất       |
|-------------|---------------------------------------------------|-------------------------------------------------------|-----------------------|-----------------------|
|  Instance   |         Mỗi instance: class + polygon             |              Biên hai vật chạm nhau / vùng mờ         |    Vẽ mask bám bi     |  Mask có tràn object  |
|segmentation |                                                   |                                                       |     từng instance     |khác không, đủ instance|
|             |                                                   |                                                       |                       |       cùng lớp        |
|-------------|---------------------------------------------------|-------------------------------------------------------|-----------------------|-----------------------|



## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ dùng ảnh COCO công khai có sẵn trong notebook; không tải ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Dừng thao tác và báo Lab Coach / GV phụ trách lab.

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
