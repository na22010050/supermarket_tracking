from ultralytics import YOLO
import torch # Thư viện để kiểm tra GPU
import os # Thêm thư viện os

# 1. Tải một mô hình YOLOv8 gốc
model = YOLO('yolov8n.pt') # Dùng bản 'n' (nano) để chạy nhanh nhất trên CPU

# 2. Bắt đầu huấn luyện
if __name__ == '__main__':
    
    # Kiểm tra xem có GPU (card màn hình) không
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Dang su dung thiet bi: {device} ---")
    
    # Đường dẫn đến file YAML TỔNG mà script vừa tạo ra
    master_yaml_file = 'D:/supermarket/merged_dataset/data.yaml'
    
    # --- ĐÂY LÀ PHẦN KIỂM SOÁT NƠI LƯU ---
    project_path = 'D:/supermarket/runs/train'
    run_name = 'supermarket_master_model'
    # ------------------------------------

    # Bắt đầu huấn luyện
    results = model.train(
        data=master_yaml_file, 
        epochs=5,      # Đang để 5 epochs để chạy test. Sau này hãy tăng lên 100-150
        imgsz=320,
        batch=4,       # Nếu lỗi 'Out of Memory' trên CPU, giảm xuống 2
        device=device,
        
        # Thêm 2 dòng này để ép YOLO lưu vào đúng thư mục
        project=project_path,
        name=run_name
    )

    print("--- HOAN THANH HUAN LUYEN! ---")
    
    # Lấy đường dẫn LƯU THỰC TẾ từ kết quả (chính xác 100%)
    actual_save_dir = results.save_dir
    best_model_path = os.path.join(actual_save_dir, 'weights', 'best.pt')

    print(f"Mo hinh tot nhat da duoc luu tai:")
    print(f"{best_model_path}")