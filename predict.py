from ultralytics import YOLO
import cv2
import math
import os

# --- PHẦN CẤU HÌNH ---

# 1. Đường dẫn đến MÔ HÌNH TỔNG 
model_path = r'D:\supermarket\models\best.pt'

# 2. Đường dẫn đến VIDEO TEST 
video_path = r'D:\supermarket\video_test.mp4'  

# 3. Nơi lưu video KẾT QUẢ
output_project_path = r'D:\supermarket\runs\predict_video'

# 4. Bảng giá cho 17 CLASS 
PRICE_LIST = {
    # Rau củ
    'banana': 5000,
    'tomato': 3000,
    'egg': 3000,
    # Đồ ăn
    'kitkat': 12000,
    'lays': 15000,
    'haohao': 5000,
    'indomie_goreng': 7000,
    # Đồ uống
    'coca': 10000,
    'th_true_milk': 8000,  # (Giữ giá 8000 như bạn đã sửa)
    'chocolate_sachet': 4000,
    'coffee_sachet': 3000,
    'juice_sachet': 3500,
    # Gia dụng
    'close_up': 40000,
    'colgate': 45000,
    'pepsodent': 35000,
    'sensodyne': 80000,
    'omo_chai': 110000,
}

# --- PHẦN CODE XỬ LÝ ---

# Tải mô hình
if not os.path.exists(model_path):
    print(f"LOI: Khong tim thay mo hinh tai: {model_path}")
    print("Hay dam bao ban da copy file 'best.pt' vao 'D:\\supermarket\\models\\'")
    exit()

print(f"Dang tai mo hinh tu: {model_path}")
grocery_model = YOLO(model_path)
person_model = YOLO('yolov8n.pt') # Dùng bản 'n' gốc để đếm người

# Mở file video
if not os.path.exists(video_path):
    print(f"LOI: Khong tim thay video test tai: {video_path}")
    print("Hay copy file video test cua ban vao D:\\supermarket va dat ten la 'video_test.mp4'")
    exit()

cap = cv2.VideoCapture(video_path)

# Cài đặt file video đầu ra
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Đảm bảo thư mục đầu ra tồn tại
os.makedirs(output_project_path, exist_ok=True)
output_video_path = os.path.join(output_project_path, 'output_video_demo.mp4')
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

print(f"Video dau ra se duoc luu tai: {output_video_path}")
print("--- DANG XU LY VIDEO ---")
print("!!! Nhan phim 'q' tren cua so video de thoat som !!!")

# Biến lưu trạng thái
TOTAL_REVENUE = 0
TOTAL_CUSTOMERS = 0
counted_item_ids = set()
counted_person_ids = set()

# Lặp qua từng khung hình
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 1. XỬ LÝ DOANH THU (Tracking hàng hóa)
    grocery_results = grocery_model.track(
        frame, 
        persist=True, 
        verbose=False, 
        conf=0.5,
        save=False  # <-- Ngăn lưu file rác
    )

    if grocery_results[0].boxes.id is not None:
        item_track_ids = grocery_results[0].boxes.id.int().tolist()
        item_classes = grocery_results[0].boxes.cls.int().tolist()

        for track_id, class_id in zip(item_track_ids, item_classes):
            if track_id not in counted_item_ids:
                class_name = grocery_model.names[class_id]
                
                if class_name in PRICE_LIST:
                    TOTAL_REVENUE += PRICE_LIST[class_name]
                    counted_item_ids.add(track_id) # Đánh dấu là "đã đếm"
                    print(f"[DOANH THU] Da tinh tien: {class_name} (ID: {track_id}), Tong moi: {TOTAL_REVENUE}")

    annotated_frame = grocery_results[0].plot()

    # 2. XỬ LÝ ĐẾM KHÁCH (Tracking người)
    person_results = person_model.track(
        frame, 
        classes=[0], 
        persist=True, 
        verbose=False, 
        conf=0.5,
        save=False  # <-- Ngăn lưu file rác
    )
    
    if person_results[0].boxes.id is not None:
        person_track_ids = person_results[0].boxes.id.int().tolist()
        
        for track_id in person_track_ids:
            if track_id not in counted_person_ids:
                counted_person_ids.add(track_id) # Đánh dấu là "đã đếm"
                print(f"[SO KHACH] Phat hien khach moi (ID: {track_id})")

        TOTAL_CUSTOMERS = len(counted_person_ids)

    annotated_frame = person_results[0].plot(img=annotated_frame)

    # 3. HIỂN THỊ ĐẦU RA LÊN VIDEO
    text_revenue = f'Tong Doanh Thu: {TOTAL_REVENUE} VND'
    text_customer = f'Tong So Khach: {TOTAL_CUSTOMERS}'
    
    # Vẽ hộp nền mờ
    cv2.rectangle(annotated_frame, (5, 5), (480, 100), (0, 0, 0), -1)
    
    # Vẽ văn bản
    cv2.putText(annotated_frame, text_revenue, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, text_customer, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Ghi file video
    out.write(annotated_frame)
    
    # --- PHẦN XEM TRỰC TIẾP  ---
    cv2.imshow('Demo Du An', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # ---------------------------------------------

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows() # Dòng này nằm BÊN NGOÀI là đúng

print("\n--- HOAN THANH XU LY VIDEO! ---")
print(f"Tong doanh thu cuoi cung: {TOTAL_REVENUE}")
print(f"Tong so khach da dem: {TOTAL_CUSTOMERS}")
print(f"Video ket qua da duoc luu tai: {output_video_path}")