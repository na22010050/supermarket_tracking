from ultralytics import YOLO
import cv2
import math
import os

# ============================================================
# 🧩 PHẦN CẤU HÌNH
# ============================================================

# 1️⃣ Đường dẫn đến mô hình sản phẩm (mô hình bạn tự huấn luyện)
MODEL_PATH = r'D:\supermarket\models\best.pt'

# 2️⃣ Đường dẫn video test
VIDEO_PATH = r'D:\supermarket\testvideo\1cam.mp4'

# 3️⃣ Thư mục lưu video đầu ra
OUTPUT_PROJECT_PATH = r'D:\supermarket\runs\predict_video'

# 4️⃣ Bảng giá cho 17 class sản phẩm
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

# ============================================================
# ⚙️ TẢI MÔ HÌNH
# ============================================================

if not os.path.exists(MODEL_PATH):
    print(f"LỖI: Không tìm thấy mô hình tại: {MODEL_PATH}")
    exit()

print(f"Đang tải mô hình sản phẩm từ: {MODEL_PATH}")
grocery_model = YOLO(MODEL_PATH)

# ⚙️ Mô hình người (đếm khách)
PERSON_MODEL_PATH = 'yolov8m.pt'   # có thể đổi sang yolov8l.pt để chính xác hơn
PERSON_CONF = 0.3
PERSON_IMGSZ = 640

# Kiểm tra nếu chưa có file thì tự tải
if not os.path.exists(PERSON_MODEL_PATH):
    print(f"⚠️ Không tìm thấy {PERSON_MODEL_PATH}, đang tải từ Ultralytics...")
    person_model = YOLO(PERSON_MODEL_PATH)  # sẽ tự tải về
else:
    person_model = YOLO(PERSON_MODEL_PATH)

# ============================================================
# 🎥 XỬ LÝ VIDEO
# ============================================================

if not os.path.exists(VIDEO_PATH):
    print(f"LỖI: Không tìm thấy video test tại: {VIDEO_PATH}")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

os.makedirs(OUTPUT_PROJECT_PATH, exist_ok=True)
output_video_path = os.path.join(OUTPUT_PROJECT_PATH, 'output_video_demo_conf30.mp4')
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

print(f"🎬 Video đầu ra sẽ lưu tại: {output_video_path}")
print("--- ĐANG XỬ LÝ VIDEO (conf=0.3 / imgsz=640) ---")
print("💡 Nhấn 'q' để thoát sớm khi đang xem video.")

# ============================================================
# 🔢 BIẾN TRẠNG THÁI
# ============================================================

TOTAL_REVENUE = 0
TOTAL_CUSTOMERS = 0
counted_item_ids = set()
counted_person_ids = set()

# ============================================================
# 🧮 VÒNG LẶP XỬ LÝ TỪNG KHUNG HÌNH
# ============================================================

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --------------------------------------------------------
    # 1️⃣ XỬ LÝ DOANH THU (MÔ HÌNH SẢN PHẨM)
    # --------------------------------------------------------
    grocery_results = grocery_model.track(
        frame, 
        persist=True, 
        verbose=False, 
        conf=0.8,
        save=False,
        imgsz=640
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
                    print(f"[DOANH THU] + {class_name} ({PRICE_LIST[class_name]} VND) → Tổng: {TOTAL_REVENUE:,} VND")

    annotated_frame = grocery_results[0].plot() 

    # --------------------------------------------------------
    # 2️⃣ XỬ LÝ ĐẾM KHÁCH (MÔ HÌNH NGƯỜI)
    # --------------------------------------------------------
    person_results = person_model.track(
        frame, 
        classes=[0], 
        persist=True, 
        verbose=False, 
        conf=PERSON_CONF,
        save=False,
        imgsz=PERSON_IMGSZ
    )

    if person_results[0].boxes.id is not None:
        person_track_ids = person_results[0].boxes.id.int().tolist()
        for track_id in person_track_ids:
            if track_id not in counted_person_ids:
                counted_person_ids.add(track_id)
                print(f"[SỐ KHÁCH] Phát hiện khách mới (ID: {track_id})")
        TOTAL_CUSTOMERS = len(counted_person_ids)

    annotated_frame = person_results[0].plot(img=annotated_frame)

    # --------------------------------------------------------
    # 3️⃣ HIỂN THỊ VÀ GHI VIDEO
    # --------------------------------------------------------
    text_revenue = f'Tong Doanh Thu: {TOTAL_REVENUE:,} VND'
    text_customer = f'Tong So Khach: {TOTAL_CUSTOMERS}'

    cv2.rectangle(annotated_frame, (5, 5), (540, 100), (0, 0, 0), -1)
    cv2.putText(annotated_frame, text_revenue, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, text_customer, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    out.write(annotated_frame)

    # Resize hiển thị cho phù hợp màn hình
    h, w = annotated_frame.shape[:2]
    display_width = 960
    if w > display_width:
        r = display_width / float(w)
        display_height = int(h * r)
        display_frame = cv2.resize(annotated_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
    else:
        display_frame = annotated_frame

    cv2.imshow('Demo Du An - Supermarket Detection', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# 🏁 KẾT THÚC
# ============================================================

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n✅ --- HOÀN THÀNH XỬ LÝ VIDEO! ---")
print(f"Tổng doanh thu cuối cùng: {TOTAL_REVENUE:,} VND")
print(f"Tổng số khách đã đếm: {TOTAL_CUSTOMERS}")
print(f"📁 Video kết quả lưu tại: {output_video_path}")
