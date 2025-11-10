from ultralytics import YOLO
import os

# --- PHẦN CẤU HÌNH ---

# 1. Tải mô hình đã huấn luyện
# (Đây là nơi mô hình của bạn đã được lưu ở lần train trước)
model_path = r'D:\supermarket\models\best.pt'

# 2. CHỈ ĐỊNH ẢNH TEST CỦA BẠN
# !!! (QUAN TRỌNG) Hãy đổi 'ten_file_anh_cua_ban.jpg' thành tên 1 file ảnh có thật !!!
image_to_test = r'D:\supermarket\merged_dataset\test\images\TH_TRUE_MILK_z4081090917936_74f94267b7aba4c0de2f5053fc16f5f1_jpg.rf.4a05e9a377a356f8f8e7c28b473c427e.jpg'

# 3. Nơi lưu kết quả
output_project_path = r'D:\supermarket\runs\predict_test'

# -----------------------

# Kiểm tra xem mô hình có tồn tại không
if not os.path.exists(model_path):
    print(f"LOI: Khong tim thay mo hinh tai: {model_path}")
    print("Ban co chac la da chay train.py thanh cong chua?")
    exit()

# Kiểm tra xem ảnh test có tồn tại không
if 'ten_file_anh_cua_ban.jpg' in image_to_test:
    print("--- --------------------------------------------------- ---")
    print("!!! CANH BAO: Ban chua doi ten file anh test !!!")
    print(f"Hay mo file 'check_model.py' va sua dong 'image_to_test'")
    print("--- --------------------------------------------------- ---")
    exit()

if not os.path.exists(image_to_test):
    print(f"LOI: Khong tim thay anh test tai: {image_to_test}")
    exit()

# Tải mô hình
model = YOLO(model_path) 

# Chạy dự đoán
print(f"--- Dang kiem tra mo hinh tren anh: {image_to_test} ---")
results = model.predict(
    source=image_to_test, 
    save=True,        # <-- Tự động lưu ảnh kết quả
    conf=0.4,         # <-- Giảm độ tự tin xuống 40% (để thấy được cả các dự đoán yếu)
    
    # --- ĐÂY LÀ PHẦN KIỂM SOÁT ĐẦU RA ---
    project=output_project_path, # Chỉ định thư mục cha
    name='test_image_results'    # Chỉ định thư mục con
    # ------------------------------------
)

# In kết quả ra terminal
print("\n--- Ket qua tim thay ---")
save_dir = ""
for r in results:
    save_dir = r.save_dir
    for box in r.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"  + Phat hien: {class_name:<18} (Do tu tin: {confidence:.2f})")

print("\n--- Da xong! ---")
print(f"Anh ket qua da duoc luu tai thu muc:")
print(f"{os.path.abspath(save_dir)}")