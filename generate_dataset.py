import cv2
import numpy as np
import os
import random
import math

# --- CẤU HÌNH CHÍNH ---
IMG_SIZE = 64
NUM_CLASSES = 10
CLASSES = [
    "circle", "rectangle", "triangle", "pentagon", "hexagon",
    "octagon", "star", "rhombus", "cross", "arrow"
]
OUTPUT_DIR = "shapes_dataset_v3"
IMAGES_PER_CLASS = 2000
TRAIN_SPLIT = 0.8

def create_dirs():
    """Tạo cấu trúc thư mục train/test cho các lớp."""
    for split in ['train', 'test']:
        split_path = os.path.join(OUTPUT_DIR, split)
        for class_name in CLASSES:
            class_path = os.path.join(split_path, class_name)
            os.makedirs(class_path, exist_ok=True)
    print(f"Đã tạo cấu trúc thư mục tại: {OUTPUT_DIR}")

def add_noise(img):
    """Thêm nhiễu Gaussian để tăng tính đa dạng."""
    if random.random() < 0.3:  # 30% cơ hội có nhiễu
        noise = np.random.normal(0, random.uniform(5, 15), img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img

def add_blur(img):
    """Thêm blur ngẫu nhiên."""
    if random.random() < 0.2:  # 20% cơ hội blur
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img

def add_brightness_contrast(img):
    """Điều chỉnh độ sáng và độ tương phản."""
    if random.random() < 0.3:
        alpha = random.uniform(0.7, 1.3)  # Contrast
        beta = random.uniform(-20, 20)    # Brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

def elastic_transform(img, alpha=20, sigma=5):
    """Biến dạng đàn hồi (elastic distortion)."""
    if random.random() < 0.2:  # 20% cơ hội
        random_state = np.random.RandomState(None)
        shape = img.shape[:2]
        
        dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
        y = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)
        
        img = cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return img

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
    """Vẽ một hình ngẫu nhiên lên ảnh với nhiều biến thể hơn."""
    
    h, w = img.shape[:2]
    
    # Màu trắng với intensity ngẫu nhiên
    intensity = random.randint(200, 255)
    color = (intensity, intensity, intensity)
    
    # Vị trí và kích thước ngẫu nhiên (tăng phạm vi)
    cx = random.randint(int(w*0.25), int(w*0.75))
    cy = random.randint(int(h*0.25), int(h*0.75))
    r_base = min(w, h) * random.uniform(0.2, 0.4)  # Tăng kích thước
    rot = random.randint(0, 360)
    
    # Thickness ngẫu nhiên với nhiều lựa chọn hơn
    thickness = random.choice([-1, -1, 2, 3, 4])  # Tăng tỷ lệ filled shapes

    if shape_name == "circle":
        r = int(r_base)
        cv2.circle(img, (cx, cy), r, color, thickness)
        
    elif shape_name == "rectangle":
        w_rect = int(r_base * random.uniform(0.8, 1.8))
        h_rect = int(r_base * random.uniform(0.5, 1.2))
        box = cv2.boxPoints(((cx, cy), (w_rect, h_rect), rot))
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, color, thickness)

    elif shape_name == "triangle":
        # Thêm variation cho triangle (equilateral, isosceles, scalene)
        r_var = r_base * random.uniform(0.9, 1.1)
        points = get_polygon_points(cx, cy, int(r_var), 3, rot)
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

    elif shape_name == "rhombus":
        r_w = int(r_base * random.uniform(0.6, 1.0))
        r_h = int(r_base * random.uniform(1.0, 1.6))
        points = np.array([
            [cx, cy - r_h],
            [cx + r_w, cy],
            [cx, cy + r_h],
            [cx - r_w, cy]
        ], dtype=np.int32)
        # Thêm rotation cho rhombus
        M = cv2.getRotationMatrix2D((cx, cy), rot, 1)
        points = cv2.transform(np.array([points]), M)[0]
        cv2.drawContours(img, [points], 0, color, thickness)

    elif shape_name == "star":
        r_outer = int(r_base)
        r_inner = int(r_base * random.uniform(0.35, 0.5))  # Variation
        points = []
        for i in range(10):
            r = r_outer if i % 2 == 0 else r_inner
            angle = (2 * math.pi * i / 10) + math.radians(rot) - (math.pi / 10)
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        cv2.drawContours(img, [points], 0, color, thickness)
        
    elif shape_name == "cross":
        l = int(r_base * 1.2)
        s = int(r_base * random.uniform(0.3, 0.5))  # Variation
        points = np.array([
            [cx - s, cy - l], [cx + s, cy - l], [cx + s, cy - s],
            [cx + l, cy - s], [cx + l, cy + s], [cx + s, cy + s],
            [cx + s, cy + l], [cx - s, cy + l], [cx - s, cy + s],
            [cx - l, cy + s], [cx - l, cy - s], [cx - s, cy - s]
        ], dtype=np.int32)
        M = cv2.getRotationMatrix2D((cx, cy), rot, 1)
        points = cv2.transform(np.array([points]), M)[0]
        cv2.drawContours(img, [points], 0, color, thickness)

    elif shape_name == "arrow":
        h = int(r_base * 1.2)
        w = int(r_base * random.uniform(0.7, 0.9))
        shaft_w = int(w * random.uniform(0.3, 0.5))
        points = np.array([
            [cx, cy - h],
            [cx + w, cy],
            [cx + shaft_w, cy],
            [cx + shaft_w, cy + h],
            [cx - shaft_w, cy + h],
            [cx - shaft_w, cy],
            [cx - w, cy]
        ], dtype=np.int32)
        M = cv2.getRotationMatrix2D((cx, cy), rot, 1)
        points = cv2.transform(np.array([points]), M)[0]
        cv2.drawContours(img, [points], 0, color, thickness)

    return img

def generate_dataset():
    """Hàm chính để tạo và lưu ảnh với augmentation."""
    create_dirs()
    
    num_train = int(IMAGES_PER_CLASS * TRAIN_SPLIT)
    
    for class_name in CLASSES:
        print(f"Đang tạo ảnh cho lớp: {class_name}...")
        for i in range(IMAGES_PER_CLASS):
            try:
                # Tạo ảnh nền đen
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                
                # Vẽ hình
                img = draw_shape(img, class_name)
                
                # Áp dụng augmentation (chỉ cho train set)
                if i < num_train:
                    img = add_noise(img)
                    img = add_blur(img)
                    img = add_brightness_contrast(img)
                    img = elastic_transform(img)
                    split = 'train'
                    img_num = i
                else:
                    split = 'test'
                    img_num = i - num_train
                    
                # Lưu ảnh
                filename = f"{img_num:05d}.png"
                output_path = os.path.join(OUTPUT_DIR, split, class_name, filename)
                cv2.imwrite(output_path, img)
                
            except Exception as e:
                print(f"Lỗi khi tạo ảnh {i} cho lớp {class_name}: {e}")
                import traceback
                traceback.print_exc()
                
    print("Hoàn tất! Dataset đã được tạo.")

if __name__ == "__main__":
    generate_dataset()