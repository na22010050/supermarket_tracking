from ultralytics import YOLO
import cv2
import math
import os

# --- PHẦN CẤU HÌNH ---

# 1. Đường dẫn đến MÔ HÌNH TỔNG 
model_path = r'D:\supermarket\models\best.pt'

# 2. NGUỒN VIDEO
video_source = 0  # 0 cho webcam mặc định

# 3. Nơi lưu video KẾT QUẢ
output_project_path = r'D:\supermarket\runs\predict_live_demo'

# 4. Bảng giá
PRICE_LIST = {
    # Rau củ
    'banana': 5000, 'tomato': 3000, 'egg': 3000,
    # Đồ ăn
    'kitkat': 12000, 'lays': 15000, 'haohao': 5000, 'indomie_goreng': 7000,
    # Đồ uống
    'coca': 10000, 'th_true_milk': 8000,
    'chocolate_sachet': 4000, 'coffee_sachet': 3000, 'juice_sachet': 3500,
    # Gia dụng
    'close_up': 40000, 'colgate': 45000, 'pepsodent': 35000,
    'sensodyne': 80000, 'omo_chai': 110000,
}

# --- PHẦN CODE XỬ LÝ ---

# Tải mô hình
if not os.path.exists(model_path):
    print(f"LOI: Khong tim thay mo hinh tai: {model_path}")
    exit()

print(f"Dang tai mo hinh tu: {model_path}")
grocery_model = YOLO(model_path)
person_model = YOLO('yolov8n.pt')

# Mở WEBCAM
print(f"--- Dang mo webcam (ID: {video_source}) ---")
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print(f"LOI: Khong the mo webcam.")
    exit()

# Cài đặt file video đầu ra
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) 

os.makedirs(output_project_path, exist_ok=True)
output_video_path = os.path.join(output_project_path, 'output_live_demo.mp4')
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

print(f"Video live se duoc LUU LAI tai: {output_video_path}")
print("--- DANG CHAY LIVE (Che do Chinh Xac Cao / Rat Cham) ---")
print("!!! Nhan phim 'q' tren cua so video de thoat !!!")

# Biến lưu trạng thái
TOTAL_REVENUE = 0
TOTAL_CUSTOMERS = 0
counted_item_ids = set()
counted_person_ids = set()

# Lặp qua từng khung hình (live)
while True:
    success, frame = cap.read()
    if not success:
        print("Loi: Mat ket noi voi webcam.")
        break

    # 1. XỬ LÝ DOANH THU 
    grocery_results = grocery_model.track(
        frame, 
        persist=True, 
        verbose=False, 
        conf=0.4,   
        save=False,
        imgsz=640   # <-- CHÍNH XÁC CAO (GÂY CHẬM)
    )

    if grocery_results[0].boxes.id is not None:
        item_track_ids = grocery_results[0].boxes.id.int().tolist()
        item_classes = grocery_results[0].boxes.cls.int().tolist()
        for track_id, class_id in zip(item_track_ids, item_classes):
            if track_id not in counted_item_ids:
                class_name = grocery_model.names[class_id]
                if class_name in PRICE_LIST:
                    TOTAL_REVENUE += PRICE_LIST[class_name]
                    counted_item_ids.add(track_id)
                    print(f"[DOANH THU] Da tinh tien: {class_name} (ID: {track_id}), Tong moi: {TOTAL_REVENUE}")
    annotated_frame = grocery_results[0].plot()

    # 2. XỬ LÝ ĐẾM KHÁCH (ĐÃ ĐỔI imgsz=640 ĐỂ CHÍNH XÁC HƠN)
    person_results = person_model.track(
        frame, 
        classes=[0], 
        persist=True, 
        verbose=False, 
        conf=0.4,  # <-- Giảm conf một chút
        save=False,
        imgsz=640  # <-- CHÍNH XÁC CAO (GÂY CHẬM)
    )
    
    if person_results[0].boxes.id is not None:
        person_track_ids = person_results[0].boxes.id.int().tolist()
        for track_id in person_track_ids:
            if track_id not in counted_person_ids:
                counted_person_ids.add(track_id)
                print(f"[SO KHACH] Phat hien khach moi (ID: {track_id})")
        TOTAL_CUSTOMERS = len(counted_person_ids)
    annotated_frame = person_results[0].plot(img=annotated_frame)

    # 3. HIỂN THỊ ĐẦU RA LÊN VIDEO
    text_revenue = f'Tong Doanh Thu: {TOTAL_REVENUE} VND'
    text_customer = f'Tong So Khach: {TOTAL_CUSTOMERS}'
    cv2.rectangle(annotated_frame, (5, 5), (480, 100), (0, 0, 0), -1)
    cv2.putText(annotated_frame, text_revenue, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, text_customer, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    out.write(annotated_frame) # Ghi lại file demo
    
    # --- PHẦN XEM TRỰC TIẾP (Vẫn giữ resize) ---
    h, w = annotated_frame.shape[:2]
    display_width = 960
    if w > display_width:
        r = display_width / float(w)
        display_height = int(h * r)
        display_frame = cv2.resize(annotated_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
    else:
        display_frame = annotated_frame

    cv2.imshow('Demo Du An LIVE (Che do Chinh Xac Cao)', display_frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # ---------------------------------------------

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()

print("\n--- DA DUNG LIVE DEMO! ---")
print(f"Tong doanh thu cuoi cung: {TOTAL_REVENUE}")
print(f"Tong so khach da dem: {TOTAL_CUSTOMERS}")
print(f"Video demo live da duoc luu tai: {output_video_path}")