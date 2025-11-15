import cv2
import numpy as np
import os
import random
import math

# --- CẤU HÌNH CHÍNH ---
IMG_SIZE = 64  # Kích thước ảnh (64x64). Giống chinese_model.
NUM_CLASSES = 10
CLASSES = [
    "circle", "rectangle", "triangle", "pentagon", "hexagon",
    "octagon", "star", "rhombus", "cross", "arrow"
]
OUTPUT_DIR = "shapes_dataset_v2"
IMAGES_PER_CLASS = 2000  # Tổng số ảnh cho mỗi lớp
TRAIN_SPLIT = 0.8  # 80% cho huấn luyện, 20% cho kiểm tra
# -----------------------

def create_dirs():
    """Tạo cấu trúc thư mục train/test cho các lớp."""
    for split in ['train', 'test']:
        split_path = os.path.join(OUTPUT_DIR, split)
        for class_name in CLASSES:
            class_path = os.path.join(split_path, class_name)
            os.makedirs(class_path, exist_ok=True)
    print(f"Đã tạo cấu trúc thư mục tại: {OUTPUT_DIR}")

def get_polygon_points(center_x, center_y, radius, n_sides, rotation=0):
    """Tính toán các đỉnh của một đa giác đều."""
    points = []
    rotation_rad = math.radians(rotation)
    for i in range(n_sides):
        angle = (2 * math.pi * i / n_sides) + rotation_rad
        x = int(center_x + radius * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))
        points.append([x, y])
    return np.array(points, dtype=np.int32)

def draw_shape(img, shape_name):
    """Vẽ một hình ngẫu nhiên lên ảnh."""
    
    h, w = img.shape[:2]
    # Màu trắng
    color = (255, 255, 255) 
    
    # Thêm ngẫu nhiên
    cx = random.randint(int(w*0.2), int(w*0.8)) # Vị trí tâm x
    cy = random.randint(int(h*0.2), int(h*0.8)) # Vị trí tâm y
    r_base = min(w, h) * random.uniform(0.15, 0.35) # Bán kính/kích thước
    rot = random.randint(0, 360) # Góc xoay
    
    # -1 = tô đặc, > 0 = độ dày viền
    thickness = random.choice([-1, 2, 3]) 

    if shape_name == "circle":
        r = int(r_base)
        cv2.circle(img, (cx, cy), r, color, thickness)
        
    elif shape_name == "rectangle":
        w_rect = int(r_base * random.uniform(1.0, 1.5))
        h_rect = int(r_base * random.uniform(0.5, 1.0))
        # Cần xoay hình chữ nhật (phức tạp hơn)
        # Cách đơn giản: dùng `cv2.boxPoints` và `cv2.fillPoly`
        box = cv2.boxPoints(((cx, cy), (w_rect, h_rect), rot))
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, color, thickness)

    elif shape_name == "triangle":
        points = get_polygon_points(cx, cy, int(r_base), 3, rot)
        cv2.drawContours(img, [points], 0, color, thickness)
        
    elif shape_name == "pentagon":
        points = get_polygon_points(cx, cy, int(r_base), 5, rot)
        cv2.drawContours(img, [points], 0, color, thickness)

    elif shape_name == "hexagon":
        points = get_polygon_points(cx, cy, int(r_base), 6, rot)
        cv2.drawContours(img, [points], 0, color, thickness)

    elif shape_name == "octagon":
        points = get_polygon_points(cx, cy, int(r_base), 8, rot)
        cv2.drawContours(img, [points], 0, color, thickness)

    elif shape_name == "rhombus": # Hình thoi
        r_w = int(r_base * random.uniform(0.7, 1.0))
        r_h = int(r_base * random.uniform(1.2, 1.5))
        points = np.array([
            [cx, cy - r_h],
            [cx + r_w, cy],
            [cx, cy + r_h],
            [cx - r_w, cy]
        ], dtype=np.int32)
        # Chúng ta cũng có thể xoay hình thoi, nhưng để đơn giản, ta bỏ qua
        cv2.drawContours(img, [points], 0, color, thickness)

    elif shape_name == "star": # 5 cánh
        r_outer = int(r_base)
        r_inner = int(r_base * 0.4)
        points = []
        for i in range(10):
            r = r_outer if i % 2 == 0 else r_inner
            angle = (2 * math.pi * i / 10) + math.radians(rot) - (math.pi / 10)
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        cv2.drawContours(img, [points], 0, color, thickness)
        
    elif shape_name == "cross": # Chữ thập
        l = int(r_base * 1.2) # Dài
        s = int(r_base * 0.4) # Ngắn
        points = np.array([
            [cx - s, cy - l], [cx + s, cy - l], [cx + s, cy - s],
            [cx + l, cy - s], [cx + l, cy + s], [cx + s, cy + s],
            [cx + s, cy + l], [cx - s, cy + l], [cx - s, cy + s],
            [cx - l, cy + s], [cx - l, cy - s], [cx - s, cy - s]
        ], dtype=np.int32)
        # Xoay (tùy chọn)
        M = cv2.getRotationMatrix2D((cx, cy), rot, 1)
        points = cv2.transform(np.array([points]), M)[0]
        cv2.drawContours(img, [points], 0, color, thickness)

    elif shape_name == "arrow": # Mũi tên (chỉ lên)
        h = int(r_base * 1.2)
        w = int(r_base * 0.8)
        shaft_w = int(w * 0.4)
        points = np.array([
            [cx, cy - h],           # Đỉnh
            [cx + w, cy],           # Cạnh phải
            [cx + shaft_w, cy],     # Cạnh trong phải
            [cx + shaft_w, cy + h], # Đuôi phải
            [cx - shaft_w, cy + h], # Đuôi trái
            [cx - shaft_w, cy],     # Cạnh trong trái
            [cx - w, cy]            # Cạnh trái
        ], dtype=np.int32)
        M = cv2.getRotationMatrix2D((cx, cy), rot, 1) # Xoay
        points = cv2.transform(np.array([points]), M)[0]
        cv2.drawContours(img, [points], 0, color, thickness)

    return img

def generate_dataset():
    """Hàm chính để tạo và lưu ảnh."""
    create_dirs()
    
    num_train = int(IMAGES_PER_CLASS * TRAIN_SPLIT)
    
    for class_name in CLASSES:
        print(f"Đang tạo ảnh cho lớp: {class_name}...")
        for i in range(IMAGES_PER_CLASS):
            try:
                # Tạo ảnh nền đen (3 kênh BGR)
                # Lý do dùng 3 kênh: Giống với ảnh đầu vào mà app.py đang đọc (cv2.imdecode)
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                
                # Vẽ hình
                img = draw_shape(img, class_name)
                
                # Quyết định lưu vào train hay test
                if i < num_train:
                    split = 'train'
                    img_num = i
                else:
                    split = 'test'
                    img_num = i - num_train
                    
                # Tạo tên file và lưu
                filename = f"{img_num:05d}.png" # Ví dụ: 00001.png
                output_path = os.path.join(OUTPUT_DIR, split, class_name, filename)
                cv2.imwrite(output_path, img)
            except Exception as e:
                print(f"Lỗi khi tạo ảnh {i} cho lớp {class_name}: {e}")
                import traceback
                traceback.print_exc()
                
    print("Hoàn tất! Dataset đã được tạo.")

if __name__ == "__main__":
    generate_dataset()