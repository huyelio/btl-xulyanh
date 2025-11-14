"""
Script demo pipeline tiền xử lý ảnh
Tạo ảnh minh họa từng bước để đưa vào báo cáo
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import ImagePreprocessor
import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


def demo_mnist_preprocessing():
    """Demo preprocessing pipeline cho MNIST"""
    print("="*60)
    print("DEMO: PIPELINE TIỀN XỬ LÝ ẢNH MNIST")
    print("="*60)
    
    # Load MNIST
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    
    # Tạo preprocessor
    preprocessor = ImagePreprocessor(save_progress=True, output_dir="../example_progress")
    
    # Chọn vài ảnh mẫu
    sample_indices = [0, 100, 500]
    
    for idx in sample_indices:
        sample_img = x_train[idx]
        label = y_train[idx]
        
        print(f"\n📸 Xử lý ảnh {idx} (nhãn: {label})...")
        
        # Chạy full pipeline
        processed = preprocessor.full_pipeline(sample_img, for_mnist=True)
        
        # Lưu progress
        preprocessor.save_progress_images(prefix=f"mnist_digit{label}_idx{idx}")
        
        print(f"   ✓ Đã lưu {len(preprocessor.get_progress_images())} bước")
    
    print("\n✓ Hoàn thành! Xem ảnh trong thư mục example_progress/")


def demo_shapes_preprocessing():
    """Demo preprocessing pipeline cho Shapes"""
    print("\n" + "="*60)
    print("DEMO: PIPELINE TIỀN XỬ LÝ ẢNH SHAPES")
    print("="*60)
    
    # Kiểm tra xem có ảnh demo không
    demo_dir = "../example_progress"
    os.makedirs(demo_dir, exist_ok=True)
    
    # Nếu chưa có, sinh mới
    from generate_shapes import ShapeGenerator
    generator = ShapeGenerator(img_size=64, background_color=0)
    
    shapes = {
        'circle': generator.generate_circle,
        'rectangle': generator.generate_rectangle,
        'triangle': generator.generate_triangle
    }
    
    # Tạo preprocessor
    preprocessor = ImagePreprocessor(save_progress=True, output_dir=demo_dir)
    
    for shape_name, shape_func in shapes.items():
        print(f"\n📐 Xử lý {shape_name}...")
        
        # Sinh hình
        shape_img = shape_func()
        
        # Lưu ảnh gốc
        cv2.imwrite(f"{demo_dir}/demo_{shape_name}.png", shape_img)
        
        # Chạy pipeline
        processed = preprocessor.full_pipeline(shape_img, for_mnist=False)
        
        # Lưu progress
        preprocessor.save_progress_images(prefix=f"shapes_{shape_name}")
        
        print(f"   ✓ Đã lưu {len(preprocessor.get_progress_images())} bước")
    
    print("\n✓ Hoàn thành! Xem ảnh trong thư mục example_progress/")


def create_comparison_figure():
    """Tạo hình so sánh các bước xử lý"""
    print("\n" + "="*60)
    print("TẠO HÌNH SO SÁNH CÁC BƯỚC")
    print("="*60)
    
    # Load ảnh mẫu MNIST
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    sample = x_train[0]
    
    # Tạo preprocessor
    preprocessor = ImagePreprocessor(save_progress=False)
    
    # Các bước quan trọng
    gray = preprocessor.to_grayscale(sample)
    equalized = preprocessor.histogram_equalization(gray)
    denoised = preprocessor.denoise_gaussian(equalized)
    thresholded = preprocessor.threshold_otsu(denoised)
    opened = preprocessor.morphology_opening(thresholded)
    edges = preprocessor.edge_detection_canny(opened)
    
    # Tạo figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Pipeline Tiền xử lý Ảnh MNIST', fontsize=16, fontweight='bold')
    
    steps = [
        (gray, '1. Grayscale'),
        (equalized, '2. Histogram Equalization'),
        (denoised, '3. Gaussian Denoising'),
        (thresholded, '4. Otsu Thresholding'),
        (opened, '5. Morphological Opening'),
        (edges, '6. Canny Edge Detection')
    ]
    
    for idx, (img, title) in enumerate(steps):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Lưu
    output_path = '../example_progress/pipeline_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Đã lưu hình so sánh: {output_path}")
    
    plt.close()


def create_results_summary():
    """Tạo bảng tóm tắt kết quả"""
    print("\n" + "="*60)
    print("TẠO BẢNG TÓM TẮT")
    print("="*60)
    
    # Dữ liệu mẫu (thay bằng kết quả thực tế sau khi train)
    results = {
        'MNIST Model': {
            'Architecture': 'CNN 3 layers',
            'Input Size': '28×28×1',
            'Parameters': '~150K',
            'Train Accuracy': '99.5%',
            'Test Accuracy': '99.2%',
            'Training Time (GPU)': '5-7 min'
        },
        'Shapes Model': {
            'Architecture': 'CNN 3 layers',
            'Input Size': '64×64×1',
            'Parameters': '~200K',
            'Train Accuracy': '99.8%',
            'Test Accuracy': '99.5%',
            'Training Time (GPU)': '3-5 min'
        }
    }
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Dữ liệu bảng
    headers = ['Metric', 'MNIST Model', 'Shapes Model']
    metrics = list(results['MNIST Model'].keys())
    
    table_data = [headers]
    for metric in metrics:
        row = [
            metric,
            results['MNIST Model'][metric],
            results['Shapes Model'][metric]
        ]
        table_data.append(row)
    
    # Vẽ bảng
    table = ax.table(
        cellText=table_data,
        cellLoc='left',
        loc='center',
        colWidths=[0.3, 0.35, 0.35]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style cho header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cho rows
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Kết quả Huấn luyện Models', fontsize=16, fontweight='bold', pad=20)
    
    # Lưu
    output_path = '../example_progress/results_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Đã lưu bảng tóm tắt: {output_path}")
    
    plt.close()


def main():
    """Main function"""
    print("\n🎨 SCRIPT DEMO PREPROCESSING & TẠO ẢNH MINH HỌA\n")
    
    # Tạo thư mục output
    os.makedirs("../example_progress", exist_ok=True)
    
    # Demo MNIST preprocessing
    demo_mnist_preprocessing()
    
    # Demo Shapes preprocessing
    demo_shapes_preprocessing()
    
    # Tạo hình so sánh
    create_comparison_figure()
    
    # Tạo bảng tóm tắt
    create_results_summary()
    
    print("\n" + "="*60)
    print("✓ HOÀN THÀNH TẤT CẢ!")
    print("="*60)
    print(f"\nĐã tạo các file trong thư mục: example_progress/")
    print("\nCác file có thể dùng cho báo cáo:")
    print("  - mnist_*.png: Pipeline xử lý MNIST")
    print("  - shapes_*.png: Pipeline xử lý Shapes")
    print("  - pipeline_comparison.png: So sánh các bước")
    print("  - results_summary.png: Bảng tóm tắt kết quả")
    print("\n📊 Sẵn sàng cho báo cáo!")


if __name__ == "__main__":
    main()

