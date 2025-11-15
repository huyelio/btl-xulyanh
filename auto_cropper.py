import cv2
import numpy as np
import os

# --- CẤU HÌNH ---
IMAGE_PATH = 'test_img/z7226810320655_5dc866768b59192812574cf2f9221fff.jpg'  # Ảnh lớn chứa các chữ số
OUTPUT_DIR = 'finetune_data/temp_crops/' # Nơi lưu các ảnh đã cắt
PADDING = 15                           # Thêm 15px đệm xung quanh

# Tạo thư mục output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Đọc và Tiền xử lý ảnh
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Lỗi: Không thể đọc ảnh từ {IMAGE_PATH}")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# QUAN TRỌNG: Dùng Adaptive Threshold và INVERT
# findContours cần vật thể TRẮNG trên nền ĐEN
thresh = cv2.adaptiveThreshold(
    blurred, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV,  # _INV = Invert
    11, 2
)

# Dùng Dilation để làm nét chữ mỏng dày lên
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=2)

# 2. Tìm Contours
contours, hierarchy = cv2.findContours(
    dilated, 
    cv2.RETR_EXTERNAL, # Chỉ lấy contour ngoài cùng
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"Tìm thấy {len(contours)} chữ số.")
count = 0

# 3. Lặp, Cắt và Lưu
for c in contours:
    # Bỏ qua các contour quá nhỏ (nhiễu)
    if cv2.contourArea(c) < 50: # (Bạn có thể cần chỉnh số này)
        continue
        
    # Lấy Bounding Box (hộp bao quanh)
    (x, y, w, h) = cv2.boundingRect(c)
    
    # Thêm PADDING (giống như "loose crop" bạn làm thủ công)
    x1 = max(0, x - PADDING)
    y1 = max(0, y - PADDING)
    x2 = min(img.shape[1], x + w + PADDING)
    y2 = min(img.shape[0], y + h + PADDING)
    
    # Cắt từ ảnh GỐC (có màu hoặc xám, tuỳ bạn)
    # Cắt từ ảnh GỐC (img) chứ không phải ảnh thresh
    cropped_digit = img[y1:y2, x1:x2]
    
    # Lưu file
    save_path = os.path.join(OUTPUT_DIR, f'crop_{count}.png')
    cv2.imwrite(save_path, cropped_digit)
    
    count += 1

print(f"Đã lưu {count} ảnh vào {OUTPUT_DIR}")