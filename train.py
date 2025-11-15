from ultralytics import YOLO
import torch 
import os

# 1. Tải mô hình 'Small' (chính xác hơn 'Nano')
model = YOLO('yolov8s.pt') 

# 2. Bắt đầu huấn luyện
if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Dang su dung thiet bi: {device} ---")
    
    # --- ĐÃ SỬA: Đường dẫn đến dataset (Cũ + Mới) ---
    data_file = r'D:\supermarket\merged_dataset\merged_dataset\data.yaml'
    
    if not os.path.exists(data_file):
        print(f"LOI: Khong tim thay file {data_file}")
        print("Hay kiem tra lai duong dan 'merged_dataset\\merged_dataset' da dung chua.")
        exit()

    print(f"--- Su dung dataset TONG HOP tai: {data_file} ---")

    # --- ĐÃ SỬA: Nơi lưu mô hình mới ---
    project_path = 'D:/supermarket/runs/train'
    run_name = 'model_17class_plus_moree_v1' # Tên mô hình mới
    
    # Bắt đầu huấn luyện
    results = model.train(
        data=data_file, 
        epochs=50,      # <-- Đặt 150 để train cho 'chín' (sẽ rất lâu trên CPU)
        imgsz=640,       # <-- Dùng 640 để đạt độ chính xác cao
        batch=2,         # <-- Giữ batch=2 cho an toàn trên CPU
        device=device,
        
        # Chỉ định nơi lưu
        project=project_path,
        name=run_name
    )

    print("--- HOAN THANH HUAN LUYEN! ---")
    
    # Lấy đường dẫn LƯU THỰC TẾ từ kết quả
    actual_save_dir = results.save_dir
    best_model_path = os.path.join(actual_save_dir, 'weights', 'best.pt')

    print(f"Mo hinh moi (tot hon) da duoc luu tai:")
    print(f"{best_model_path}")
    print("\n--- BUOC TIEP THEO ---")
    print(f"1. Copy file 'best.pt' o tren.")
    print(f"2. Dan (Paste) vao thu muc 'D:\\supermarket\\models\\' (de ghi de file cu).")
    print(f"3. Chay file 'predict.py' de kiem tra.")