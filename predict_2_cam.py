from ultralytics import YOLO
import cv2
import math
import os
from collections import defaultdict # Dùng để lưu lịch sử tracking

# ============================================================
# 🧩 PHẦN CẤU HÌNH
# ============================================================

# 1️⃣ Đường dẫn video
VIDEO_PATH_PERSON = r'D:\supermarket\testvideo\testnguoi.mp4'  # <-- VIDEO ĐẾM KHÁCH
VIDEO_PATH_PRODUCT = r'D:\supermarket\testvideo\2cam.mp4' # <-- VIDEO TÍNH TIỀN

# 2️⃣ Đường dẫn mô hình
MODEL_PATH_PRODUCT = r'D:\supermarket\models\best.pt' # Mô hình sản phẩm
MODEL_PATH_PERSON = 'yolov8m.pt' # Mô hình người

# 3️⃣ Thư mục lưu video đầu ra
OUTPUT_PROJECT_PATH = r'D:\supermarket\runs\predict_2_cam_video'

# 4️⃣ Bảng giá cho 17 class sản phẩm (Giữ nguyên)
PRICE_LIST = {
    'banana': 3000,
    'tomato': 3000,
    'egg': 3000,
    'kitkat': 12000,
    'lays': 15000,
    'haohao': 5000,
    'indomie_goreng': 7000,
    'coca': 10000,
    'th_true_milk': 8000,
    'chocolate_sachet': 4000,
    'coffee_sachet': 3000,
    'juice_sachet': 3500,
    'close_up': 40000,
    'colgate': 45000,
    'pepsodent': 35000,
    'sensodyne': 80000,
    'omo_chai': 110000,
}

# Cấu hình Vùng Quan Tâm (ROI)
ROI_LINE_PERCENTAGE = 0.5 

# ============================================================
# ⚙️ TẢI MÔ HÌNH
# ============================================================

print(f"Đang tải mô hình sản phẩm từ: {MODEL_PATH_PRODUCT}")
grocery_model = YOLO(MODEL_PATH_PRODUCT)
print(f"Đang tải mô hình người từ: {MODEL_PATH_PERSON}")
person_model = YOLO(MODEL_PATH_PERSON)

# ============================================================
# 🎥 MỞ 2 NGUỒN VIDEO
# ============================================================

if not os.path.exists(VIDEO_PATH_PERSON):
    print(f"LỖI: Không tìm thấy video đếm khách: {VIDEO_PATH_PERSON}")
    exit()
if not os.path.exists(VIDEO_PATH_PRODUCT):
    print(f"LỖI: Không tìm thấy video tính tiền: {VIDEO_PATH_PRODUCT}")
    exit()

cap_person = cv2.VideoCapture(VIDEO_PATH_PERSON)
cap_product = cv2.VideoCapture(VIDEO_PATH_PRODUCT)

# Cài đặt 2 file video đầu ra
os.makedirs(OUTPUT_PROJECT_PATH, exist_ok=True)

w_p = int(cap_person.get(cv2.CAP_PROP_FRAME_WIDTH))
h_p = int(cap_person.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_p = int(cap_person.get(cv2.CAP_PROP_FPS))
out_person_path = os.path.join(OUTPUT_PROJECT_PATH, 'output_cam_khach.mp4')
out_person = cv2.VideoWriter(out_person_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_p, (w_p, h_p))

w_pr = int(cap_product.get(cv2.CAP_PROP_FRAME_WIDTH))
h_pr = int(cap_product.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_pr = int(cap_product.get(cv2.CAP_PROP_FPS))
out_product_path = os.path.join(OUTPUT_PROJECT_PATH, 'output_cam_tinhtien.mp4')
out_product = cv2.VideoWriter(out_product_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_pr, (w_pr, h_pr))

print(f"🎬 Video đếm khách sẽ lưu tại: {out_person_path}")
print(f"🎬 Video tính tiền sẽ lưu tại: {out_product_path}")
print("--- ĐANG XỬ LÝ 2 VIDEO ---")
print("💡 Nhấn 'q' để thoát sớm khi đang xem video.")

# ============================================================
# 🔢 BIẾN TRẠNG THÁI
# ============================================================

TOTAL_REVENUE = 0
TOTAL_CUSTOMERS = 0
counted_item_ids = set()
counted_person_ids = set() 
track_history = defaultdict(lambda: []) 

# Tính toán vị trí đường kẻ DỌC (trục X)
ROI_LINE_X = int(w_p * ROI_LINE_PERCENTAGE)

# ============================================================
# 🧮 VÒNG LẶP XỬ LÝ
# ============================================================

while True:
    success_p, frame_p = cap_person.read()
    success_pr, frame_pr = cap_product.read()

    if not success_p and not success_pr:
        print("Cả 2 video đã xử lý xong.")
        break

    # --------------------------------------------------------
    # 1️⃣ XỬ LÝ VIDEO ĐẾM KHÁCH (CHỈ ĐẾM PHẢI -> TRÁI)
    # --------------------------------------------------------
    if success_p:
        annotated_frame_p = frame_p.copy()
        
        person_results = person_model.track(
            frame_p, classes=[0], persist=True, verbose=False, 
            conf=0.3, save=False, imgsz=640
        )
        
        # Vẽ vạch DỌC
        cv2.line(annotated_frame_p, (ROI_LINE_X, 0), (ROI_LINE_X, h_p), (0, 0, 255), 3)

        if person_results[0].boxes.id is not None:
            boxes = person_results[0].boxes.xyxy.cpu()
            track_ids = person_results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x_center = (box[0] + box[2]) / 2
                
                history_x = track_history[track_id]
                history_x.append(x_center)
                
                if len(history_x) > 2:
                    history_x.pop(0)

                if len(history_x) == 2 and track_id not in counted_person_ids:
                    x_prev = history_x[0]
                    x_curr = history_x[1]
                    
                    # --- (ĐÃ SỬA) CHỈ KIỂM TRA HƯỚNG TỪ PHẢI SANG TRÁI ---
                    
                    # 1. KIỂM TRA ĐI TỪ PHẢI SANG TRÁI (Right -> Left)
                    if x_prev >= ROI_LINE_X and x_curr < ROI_LINE_X:
                        TOTAL_CUSTOMERS += 1
                        counted_person_ids.add(track_id)
                        print(f"[SỐ KHÁCH] Phat hien khach moi (ID: {track_id}) (Phải -> Trái). Tổng: {TOTAL_CUSTOMERS}")
                    
                    # 2. KIỂM TRA ĐI TỪ TRÁI SANG PHẢI (Left -> Right) -> BỎ QUA
                    # elif x_prev < ROI_LINE_X and x_curr >= ROI_LINE_X:
                    #     pass # Không đếm chiều này

            annotated_frame_p = person_results[0].plot(img=annotated_frame_p)

        # Hiển thị thông tin
        text_customer = f'Tong So Khach: {TOTAL_CUSTOMERS}'
        cv2.rectangle(annotated_frame_p, (5, 5), (400, 50), (0, 0, 0), -1)
        cv2.putText(annotated_frame_p, text_customer, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        out_person.write(annotated_frame_p)
        
        # Resize và Hiển thị
        h, w = annotated_frame_p.shape[:2]
        if w > 800:
            r = 800 / float(w)
            display_p = cv2.resize(annotated_frame_p, (800, int(h * r)), interpolation=cv2.INTER_AREA)
        else:
            display_p = annotated_frame_p
        cv2.imshow('Cam 1: Dem Khach', display_p)


    # --------------------------------------------------------
    # 2️⃣ XỬ LÝ VIDEO TÍNH TIỀN 
    # --------------------------------------------------------
    if success_pr:
        annotated_frame_pr = frame_pr.copy()
        
        grocery_results = grocery_model.track(
            frame_pr, persist=True, verbose=False, 
            conf=0.4, save=False, imgsz=640
        )

        if grocery_results[0].boxes.id is not None:
            item_track_ids = grocery_results[0].boxes.id.int().tolist()
            item_classes = grocery_results[0].boxes.cls.int().tolist()

            for track_id, class_id in zip(item_track_ids, item_classes):
                if track_id not in counted_item_ids:
                    if class_id < len(grocery_model.names):
                        class_name = grocery_model.names[class_id]
                        if class_name in PRICE_LIST:
                            TOTAL_REVENUE += PRICE_LIST[class_name]
                            counted_item_ids.add(track_id)
                            print(f"[DOANH THU] + {class_name} ({PRICE_LIST[class_name]} VND) → Tổng: {TOTAL_REVENUE:,} VND")
                    else:
                        print(f"CANH BAO: Phat hien class_id khong hop le: {class_id}")
        
        annotated_frame_pr = grocery_results[0].plot(img=annotated_frame_pr)
        
        # Hiển thị thông tin
        text_revenue = f'Tong Doanh Thu: {TOTAL_REVENUE:,} VND'
        cv2.rectangle(annotated_frame_pr, (5, 5), (540, 50), (0, 0, 0), -1)
        cv2.putText(annotated_frame_pr, text_revenue, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        out_product.write(annotated_frame_pr)
        
        # Resize và Hiển thị
        h, w = annotated_frame_pr.shape[:2]
        if w > 800:
            r = 800 / float(w)
            display_pr = cv2.resize(annotated_frame_pr, (800, int(h * r)), interpolation=cv2.INTER_AREA)
        else:
            display_pr = annotated_frame_pr
        cv2.imshow('Cam 2: Tinh Doanh Thu', display_pr)

    # --------------------------------------------------------
    # Thoát nếu nhấn 'q'
    # --------------------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# 🏁 KẾT THÚC
# ============================================================

cap_person.release()
cap_product.release()
out_person.release()
out_product.release()
cv2.destroyAllWindows()

print("\n✅ --- HOÀN THÀNH XỬ LÝ 2 VIDEO! ---")
print(f"Tổng doanh thu cuối cùng: {TOTAL_REVENUE:,} VND")
print(f"Tổng số khách đã đếm: {TOTAL_CUSTOMERS}")
print(f"📁 Video kết quả lưu tại: {OUTPUT_PROJECT_PATH}")