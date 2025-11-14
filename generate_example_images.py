"""
Script tạo ảnh minh họa preprocessing cho báo cáo - V3 với save steps
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras

# Import preprocessing functions
import sys
sys.path.append('src')
from preprocessing import preprocess_for_mnist, preprocess_for_shapes

print("="*70)
print("🎨 TẠO ẢNH MINH HỌA V3 - VỚI SAVE STEPS")
print("="*70)

# Tạo thư mục output
output_dir = 'example_progress'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/progress_images', exist_ok=True)

# ============================================================================
# 1. MNIST PREPROCESSING DEMO
# ============================================================================
print("\n📊 Bước 1: Tạo ảnh minh họa MNIST preprocessing...")

# Load MNIST samples
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Chọn 3 samples
mnist_samples = [
    (x_train[0], y_train[0], 'sample0'),
    (x_train[100], y_train[100], 'sample1'),
    (x_train[200], y_train[200], 'sample2')
]

for img, label, name in mnist_samples:
    # Gốc
    cv2.imwrite(f'{output_dir}/{name}_original.png', img)
    
    # Preprocessing với save_steps=True
    processed, display, progress = preprocess_for_mnist(
        img, 
        save_steps=True,
        output_dir=f'{output_dir}/progress_images/mnist_{name}'
    )
    
    print(f"✓ Đã lưu MNIST {name} (label={label}) với {len(progress)} bước")

# ============================================================================
# 2. SHAPES PREPROCESSING DEMO
# ============================================================================
print("\n📐 Bước 2: Tạo ảnh minh họa Shapes preprocessing...")

# Generate demo shapes
def generate_circle(img_size=64):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.circle(img, (img_size//2, img_size//2), 20, 255, -1)
    return img

def generate_rectangle(img_size=64):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.rectangle(img, (16, 16), (48, 48), 255, -1)
    return img

def generate_triangle(img_size=64):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    pts = np.array([[32, 12], [8, 52], [56, 52]], dtype=np.int32)
    cv2.fillPoly(img, [pts], 255)
    return img

shapes = [
    ('circle', generate_circle()),
    ('rectangle', generate_rectangle()),
    ('triangle', generate_triangle())
]

for name, img in shapes:
    # Gốc
    cv2.imwrite(f'{output_dir}/shape_{name}_original.png', img)
    
    # Preprocessing với save_steps=True
    processed, display, progress = preprocess_for_shapes(
        img,
        save_steps=True,
        output_dir=f'{output_dir}/progress_images/shape_{name}'
    )
    
    print(f"✓ Đã lưu shape: {name} với {len(progress)} bước")

# ============================================================================
# 3. TẠO COMPARISON FIGURES
# ============================================================================
print("\n📊 Bước 3: Tạo ảnh so sánh preprocessing...")

# MNIST comparison - chỉ key steps
fig, axes = plt.subplots(3, 6, figsize=(15, 8))
fig.suptitle('MNIST Preprocessing Pipeline (Key Steps)', fontsize=16, fontweight='bold')

key_steps = ['step01_grayscale', 'step02_clahe', 'step04_otsu_threshold', 
             'step05_inverted', 'step11_centered', 'step12_final_smoothed']
step_names = ['Grayscale', 'CLAHE', 'Otsu', 'Invert', 'Centered', 'Final']

for row, (img, label, name) in enumerate(mnist_samples):
    for col, (step, step_name) in enumerate(zip(key_steps, step_names)):
        img_path = f'{output_dir}/progress_images/mnist_{name}/{step}.png'
        
        if os.path.exists(img_path):
            img_show = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            axes[row, col].imshow(img_show, cmap='gray')
            if row == 0:
                axes[row, col].set_title(step_name, fontsize=9, fontweight='bold')
            if col == 0:
                axes[row, col].set_ylabel(f'Label: {label}', fontsize=9)
            axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/mnist_preprocessing_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Đã lưu MNIST comparison")

# Shapes comparison
fig, axes = plt.subplots(3, 6, figsize=(15, 8))
fig.suptitle('Shapes Preprocessing Pipeline (Key Steps)', fontsize=16, fontweight='bold')

for row, (name, img) in enumerate(shapes):
    for col, (step, step_name) in enumerate(zip(key_steps, step_names)):
        img_path = f'{output_dir}/progress_images/shape_{name}/{step}.png'
        
        if os.path.exists(img_path):
            img_show = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            axes[row, col].imshow(img_show, cmap='gray')
            if row == 0:
                axes[row, col].set_title(step_name, fontsize=9, fontweight='bold')
            if col == 0:
                axes[row, col].set_ylabel(name.capitalize(), fontsize=9)
            axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/shapes_preprocessing_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Đã lưu Shapes comparison")

# ============================================================================
# 4. TẠO PIPELINE FLOWCHART
# ============================================================================
print("\n🔄 Bước 4: Tạo flowchart preprocessing V3...")

fig, ax = plt.subplots(figsize=(10, 12))
ax.axis('off')

# Các bước preprocessing V3
steps_text = [
    "1. Grayscale Conversion",
    "2. CLAHE (Histogram Eq)",
    "3. Gaussian Blur (5x5)",
    "4. Otsu Threshold",
    "5. Corner-based Invert Detection",
    "6. Morphology Opening",
    "7. Morphology Closing",
    "8. Contour Detection",
    "9. Crop with Padding",
    "10. Resize (keep aspect ratio)",
    "11. Center on Canvas",
    "12. Final Smoothing (3x3)",
    "→ Output: WHITE on BLACK"
]

y_pos = 0.95
for i, step in enumerate(steps_text):
    # Box
    if i == len(steps_text) - 1:
        bbox = dict(boxstyle="round,pad=0.5", facecolor='lightgreen', edgecolor='darkgreen', linewidth=3)
        fontsize = 13
        weight = 'bold'
    else:
        bbox = dict(boxstyle="round,pad=0.5", facecolor='lightblue', edgecolor='black', linewidth=2)
        fontsize = 11
        weight = 'normal'
    
    ax.text(0.5, y_pos, step, ha='center', va='center', fontsize=fontsize, bbox=bbox, fontweight=weight)
    
    # Arrow
    if i < len(steps_text) - 1:
        ax.annotate('', xy=(0.5, y_pos-0.055), xytext=(0.5, y_pos-0.025),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    y_pos -= 0.07

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('V3 Ultra Robust Preprocessing Pipeline', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{output_dir}/preprocessing_flowchart_v3.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Đã lưu flowchart V3")

# ============================================================================
# 5. TẠO BEFORE/AFTER COMPARISON
# ============================================================================
print("\n🎨 Bước 5: Tạo Before/After comparison...")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Before vs After Preprocessing', fontsize=16, fontweight='bold')

# MNIST samples
for col, (img, label, name) in enumerate(mnist_samples):
    # Before
    axes[0, col].imshow(img, cmap='gray')
    axes[0, col].set_title(f'Original (Label: {label})', fontsize=10)
    axes[0, col].axis('off')
    
    # After
    final_path = f'{output_dir}/progress_images/mnist_{name}/step12_final_smoothed.png'
    if os.path.exists(final_path):
        final_img = cv2.imread(final_path, cv2.IMREAD_GRAYSCALE)
        axes[1, col].imshow(final_img, cmap='gray')
        axes[1, col].set_title('Preprocessed', fontsize=10)
        axes[1, col].axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/mnist_before_after.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Đã lưu MNIST before/after")

# ============================================================================
# 6. TỔNG KẾT
# ============================================================================
print("\n" + "="*70)
print("✅ HOÀN THÀNH!")
print("="*70)

# Đếm số files
png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
progress_dirs = [d for d in os.listdir(f'{output_dir}/progress_images') if os.path.isdir(f'{output_dir}/progress_images/{d}')]

print(f"\n📁 Đã tạo ảnh trong: {output_dir}/")
print(f"\n📸 Summary:")
print(f"  - {len(png_files)} ảnh tổng hợp trong {output_dir}/")
print(f"  - {len(progress_dirs)} thư mục progress images")
print(f"  - Mỗi thư mục có ~12 bước xử lý")

print("\n💡 Các ảnh quan trọng cho báo cáo:")
print("  1. mnist_preprocessing_comparison.png - So sánh pipeline MNIST")
print("  2. shapes_preprocessing_comparison.png - So sánh pipeline Shapes")
print("  3. preprocessing_flowchart_v3.png - Sơ đồ quy trình V3")
print("  4. mnist_before_after.png - So sánh trước/sau")
print("  5. example_progress/progress_images/* - Chi tiết từng bước")

print("\n🎯 Đặc điểm V3:")
print("  ✅ Corner-based background detection")
print("  ✅ Output chuẩn: WHITE on BLACK")
print("  ✅ Lưu MỌI bước xử lý")
print("  ✅ CLAHE thay vì equalizeHist")
print("  ✅ Otsu threshold tự động")
print("="*70)
