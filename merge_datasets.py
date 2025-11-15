import os
import shutil
import yaml
# 1. Định nghĩa TẤT CẢ 17 class 
MASTER_CLASSES_LIST = [
    # Rau củ
    'banana',
    'tomato',
    'egg',
    # Đồ ăn
    'kitkat',
    'lays',
    'haohao',
    'indomie_goreng',
    # Đồ uống
    'coca',
    'th_true_milk',
    'chocolate_sachet',
    'coffee_sachet',
    'juice_sachet',
    # Gia dụng
    'close_up',
    'colgate',
    'pepsodent',
    'sensodyne',
    'omo_chai',
]

# 2. Liệt kê đường dẫn đến TẤT CẢ các dataset con
SOURCE_DATASETS = [
    'D:/supermarket/drinks/coca',
    'D:/supermarket/drinks/TH_TRUE_MILK',
    'D:/supermarket/drinks/tea',
    'D:/supermarket/Food/eggs',
    'D:/supermarket/Food/kitkat',
    'D:/supermarket/Household/Colgate',
    'D:/supermarket/Household/omo',
    'D:/supermarket/Instant_Noodles/haohao',
    'D:/supermarket/Instant_Noodles/idomi',
    'D:/supermarket/Snacks/lays',
    'D:/supermarket/vegetable/banana',
    'D:/supermarket/vegetable/tomato',
    
    # --- ĐÃ THÊM DATASET MỚI  ---
    'D:/supermarket/moree'
    # --------------------------------------
]

# 3. Định nghĩa thư mục ĐẦU RA
OUTPUT_DIR = 'D:/supermarket/merged_dataset'

# ======================================================================
# PHẦN CODE XỬ LÝ (Không cần sửa)
# ======================================================================

def load_yaml(path):
    """Tải file YAML một cách an toàn."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG khi đọc file YAML: {path}. Lỗi: {e}")
        return None

def create_output_dirs(base_dir):
    """Tạo các thư mục đầu ra (train/valid/test)."""
    print(f"Đang tạo cấu trúc thư mục tại: {base_dir}")
    # Xóa thư mục cũ nếu tồn tại để làm lại từ đầu
    if os.path.exists(base_dir):
        print("Phat hien merged_dataset cu. Dang xoa...")
        shutil.rmtree(base_dir)
        print("Da xoa.")
        
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

def get_image_path(label_dir, img_dir, label_file_name):
    """Tìm file ảnh tương ứng (jpg, png, jpeg) cho file label."""
    img_name_base = os.path.splitext(label_file_name)[0]
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        img_path = os.path.join(img_dir, img_name_base + ext)
        if os.path.exists(img_path):
            return img_path, img_name_base + ext
    
    for root, _, files in os.walk(img_dir):
        for file in files:
            if os.path.splitext(file)[0] == img_name_base:
                return os.path.join(root, file), file

    return None, None

def process_dataset(dataset_path, master_classes_map):
    """Xử lý, "dịch" ID và gộp một dataset con."""
    print(f"\n--- Đang xử lý: {dataset_path} ---")
    
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    data_config = load_yaml(yaml_path)
    
    if data_config is None:
        print("LỖI: Không thể đọc file data.yaml. Bỏ qua dataset này.")
        return

    local_classes = data_config.get('names')
    if not local_classes:
        print("LỖI: File data.yaml không có mục 'names'. Bỏ qua dataset này.")
        return

    # Tạo bản đồ "dịch" ID
    remap = {}
    for local_id, class_name in enumerate(local_classes):
        if class_name in master_classes_map:
            remap[local_id] = master_classes_map[class_name]
        else:
            print(f"  CẢNH BÁO: Class '{class_name}' không có trong MASTER_CLASSES_LIST. Sẽ bỏ qua class này.")
            
    if not remap:
        print("KHÔNG CÓ CLASS NÀO TRÙNG KHỚP. Bỏ qua dataset này.")
        return

    print(f"  Bản đồ dịch ID: {remap}")

    # Xử lý các split (train/valid/test)
    for split in ['train', 'valid', 'test']:
        if split not in data_config:
            print(f"  Không tìm thấy split '{split}' trong data.yaml. Bỏ qua.")
            continue
            
        print(f"  Đang xử lý split: {split}...")
        
        img_dir_rel = data_config[split].replace('../', '').replace('./', '')
        img_dir_abs = os.path.join(dataset_path, img_dir_rel)
        
        label_dir_rel = img_dir_rel.replace('images', 'labels')
        label_dir_abs = os.path.join(dataset_path, label_dir_rel)

        if not os.path.isdir(img_dir_abs):
             img_dir_abs = os.path.join(dataset_path, split, 'images')
        if not os.path.isdir(label_dir_abs):
             label_dir_abs = os.path.join(dataset_path, split, 'labels')

        if not os.path.isdir(img_dir_abs) or not os.path.isdir(label_dir_abs):
            print(f"  CẢNH BÁO: Không tìm thấy thư mục images/labels cho split '{split}'. Bỏ qua.")
            print(f"    (Đã thử tìm ở: {img_dir_abs} và {label_dir_abs})")
            continue

        file_count = 0

        for root, _, files in os.walk(label_dir_abs):
            for label_file in files:
                if not label_file.endswith('.txt'):
                    continue
                
                old_label_path = os.path.join(root, label_file)
                old_img_path, img_file_name = get_image_path(root, img_dir_abs, label_file)
                
                if not old_img_path or not img_file_name:
                    print(f"    CẢNH BÁO: Không tìm thấy ảnh cho label '{label_file}'. Bỏ qua file.")
                    continue

                dataset_name_prefix = os.path.basename(dataset_path)
                new_file_name_base = f"{dataset_name_prefix}_{os.path.splitext(img_file_name)[0]}"
                
                new_label_path = os.path.join(OUTPUT_DIR, split, 'labels', f"{new_file_name_base}.txt")
                new_img_path = os.path.join(OUTPUT_DIR, split, 'images', f"{new_file_name_base}{os.path.splitext(img_file_name)[1]}")

                try:
                    with open(old_label_path, 'r') as f_in, open(new_label_path, 'w') as f_out:
                        for line in f_in:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            
                            try:
                                local_id = int(parts[0])
                            except ValueError:
                                print(f"    LỖI: File label '{old_label_path}' có dòng lỗi: {line}")
                                continue
                            
                            if local_id in remap:
                                master_id = remap[local_id]
                                new_line = f"{master_id} {' '.join(parts[1:])}\n"
                                f_out.write(new_line)
                    
                    shutil.copyfile(old_img_path, new_img_path)
                    file_count += 1

                except Exception as e:
                    print(f"    LỖI khi xử lý file {label_file}. Lỗi: {e}")

        print(f"  Đã xử lý và gộp {file_count} file từ split '{split}'.")

    print(f"--- Hoàn thành xử lý: {dataset_path} ---")


def main():
    print("Bắt đầu quá trình gộp dataset...")
    
    master_classes_map = {name: i for i, name in enumerate(MASTER_CLASSES_LIST)}
    
    create_output_dirs(OUTPUT_DIR)
    
    for dataset_path in SOURCE_DATASETS:
        process_dataset(dataset_path, master_classes_map)
        
    master_yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    master_yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(MASTER_CLASSES_LIST),
        'names': MASTER_CLASSES_LIST
    }
    
    with open(master_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(master_yaml_content, f, allow_unicode=True, sort_keys=False)
        
    print(f"\n*** HOÀN THÀNH! ***")
    print(f"Dataset tổng đã được tạo tại: {OUTPUT_DIR}")
    print(f"File cấu hình tổng: {master_yaml_path}")
    print(f"Tổng số class: {len(MASTER_CLASSES_LIST)}")
    print("\nBạn có thể bắt đầu huấn luyện bằng file data.yaml này.")

if __name__ == '__main__':
    main()