import os
import shutil
import yaml

print("--- Bat dau qua trinh THEM 'moree' vao 'merged_dataset' ---")

# --- PHẦN CẤU HÌNH ---
# 1. Thư mục data CŨ (Đã gộp 17 class)
BASE_DATASET_DIR = r'D:\supermarket\merged_dataset\merged_dataset'

# 2. Thư mục data MỚI 
ADDON_DATASET_DIR = r'D:\supermarket\moree'
# ---------------------

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Tải "Bản Đồ Tổng" (17 class) từ merged_dataset
try:
    base_yaml_path = os.path.join(BASE_DATASET_DIR, 'data.yaml')
    base_config = load_yaml(base_yaml_path)
    MASTER_CLASSES_MAP = {name: i for i, name in enumerate(base_config['names'])}
    print(f"Da tai ban do 17 class tu 'merged_dataset' (vi du: 'banana' = {MASTER_CLASSES_MAP.get('banana')})")
except Exception as e:
    print(f"LOI NANG: Khong tim thay file 'data.yaml' goc trong {BASE_DATASET_DIR}. Dung lai. {e}")
    exit()

# Tải "data.yaml" của 'moree'
try:
    addon_yaml_path = os.path.join(ADDON_DATASET_DIR, 'data.yaml')
    addon_config = load_yaml(addon_yaml_path)
    addon_classes = addon_config['names']
    print(f"Da tai file config 'moree' voi {len(addon_classes)} class.")
except Exception as e:
    print(f"LOI NANG: Khong tim thay file 'data.yaml' trong {ADDON_DATASET_DIR}. Dung lai. {e}")
    exit()

# Tạo "Từ điển dịch" ID
remap = {}
for local_id, class_name in enumerate(addon_classes):
    if class_name in MASTER_CLASSES_MAP:
        master_id = MASTER_CLASSES_MAP[class_name]
        remap[local_id] = master_id
        print(f"  + 'moree' class '{class_name}' (ID {local_id}) se duoc dich thanh Master ID {master_id}")
    else:
        print(f"  - CANH BAO: Class '{class_name}' (tu 'moree') khong co trong 17 class goc. Se bo qua.")

# Lặp qua các thư mục (train, valid, test) của 'moree' và gộp
total_files_added = 0
for split in ['train', 'valid', 'test']:
    print(f"\n--- Dang xu ly thu muc: {split} ---")
    
    # Đường dẫn thư mục con của 'moree'
    addon_img_dir = os.path.join(ADDON_DATASET_DIR, split, 'images')
    addon_label_dir = os.path.join(ADDON_DATASET_DIR, split, 'labels')
    
    # Đường dẫn thư mục con của 'merged_dataset' (nơi sẽ gộp vào)
    base_img_dir = os.path.join(BASE_DATASET_DIR, split, 'images')
    base_label_dir = os.path.join(BASE_DATASET_DIR, split, 'labels')
    
    if not os.path.exists(addon_label_dir) or not os.path.exists(addon_img_dir):
        print(f"  Khong tim thay thu muc '{split}' trong 'moree'. Bo qua.")
        continue
        
    file_count = 0
    for label_file in os.listdir(addon_label_dir):
        if not label_file.endswith('.txt'): continue
        
        base_name = os.path.splitext(label_file)[0]
        
        # Tìm ảnh tương ứng
        img_file = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            if os.path.exists(os.path.join(addon_img_dir, base_name + ext)):
                img_file = base_name + ext
                break
        
        if not img_file:
            print(f"    Khong tim thay anh cho label {label_file}. Bo qua.")
            continue

        # Đường dẫn file gốc (trong 'moree')
        old_label_path = os.path.join(addon_label_dir, label_file)
        old_img_path = os.path.join(addon_img_dir, img_file)
        
        # Đường dẫn file MỚI (trong 'merged_dataset')
        # Thêm tiền tố 'moree_' để tránh TRÙNG TÊN FILE
        new_file_basename = f"moree_{base_name}"
        new_label_path = os.path.join(base_label_dir, new_file_basename + ".txt")
        new_img_path = os.path.join(base_img_dir, new_file_basename + os.path.splitext(img_file)[1])
        
        # "Dịch" file label và ghi file mới
        try:
            with open(old_label_path, 'r') as f_in, open(new_label_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    local_id = int(parts[0])
                    if local_id in remap: # Chỉ ghi nếu class đó hợp lệ
                        master_id = remap[local_id]
                        f_out.write(f"{master_id} {' '.join(parts[1:])}\n")
            
            # Copy file ảnh
            shutil.copyfile(old_img_path, new_img_path)
            file_count += 1
            total_files_added += 1
        except Exception as e:
            print(f"    LOI khi copy file {label_file}: {e}")
            
    print(f"  Da them {file_count} file moi vao '{split}' cua 'merged_dataset'.")
    
print(f"\n*** HOAN THANH! ***")
print(f"Da them thanh cong tong cong {total_files_added} file (anh + label) tu 'moree' vao 'merged_dataset'.")
print("Ban da san sang de train lai!")