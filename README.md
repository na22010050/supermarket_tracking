# Dự án Cuối kỳ: Theo dõi số lượng khách và doanh thu trong siêu thị mini (YOLOv8)

Đây là dự án cuối kỳ môn Thị giác Máy tính, sử dụng YOLOv8 để theo dõi số lượng khách hàng và tính toán doanh thu (mô phỏng) tại một siêu thị mini.

##Các tính năng chính

* **Nhận diện 17 loại sản phẩm:** Mô hình được huấn luyện để nhận diện 17 loại sản phẩm phổ biến tại Việt Nam (Mì Hảo Hảo, Coca, Sữa TH, ...).
* **Đếm khách hàng:** Sử dụng mô hình YOLOv8 gốc để phát hiện và đếm người.
* **Tính doanh thu:** Gán giá cho từng sản phẩm và tính tổng doanh thu tích lũy.
* **Tracking (Theo dõi):** Sử dụng thuật toán BoTSoRT (đã tích hợp) để gán ID duy nhất, đảm bảo không bị đếm lặp khách hàng hoặc sản phẩm khi đứng yên.

##  Cài đặt

### 1. Tải Dữ Liệu (Quan trọng)

Do dataset (6000+ ảnh) quá lớn, bạn cần tải file `merged_dataset.zip` từ Google Drive và giải nén vào thư mục gốc của dự án.

* **Link tải Dataset và models:** [https://drive.google.com/drive/folders/1aYXgOA-JAueX7_q2JfNNnvzR-cOt5Rdq?usp=sharing]

### 2. Tải Mô hình (Model)

Mô hình `best.pt` đã được huấn luyện (đã bao gồm trong thư mục `models/` của repo này).

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```
### 4.Chạy Demo
Đặt một file video (ví dụ video.mp4) vào thư mục D:\supermarket.
Mở file predict.py và sửa lại dòng video_path cho đúng.
Chạy lệnh:
```Bash
python predict.py
```