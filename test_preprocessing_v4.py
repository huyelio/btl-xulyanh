"""
Script test NHANH Preprocessing V4 với ảnh viết tay thực tế
Chạy: python test_preprocessing_v4.py <đường_dẫn_ảnh>
"""

import cv2
import numpy as np
import sys
import os
from tensorflow import keras

# Add src to path
sys.path.append('src')
from preprocessing import preprocess_for_mnist

def test_preprocessing_v4(image_path):
    """
    Test preprocessing V4 với 1 ảnh
    
    Args:
        image_path: Đường dẫn đến ảnh viết tay
    """
    print("\n" + "="*70)
    print("🧪 TEST PREPROCESSING V4 - REAL HANDWRITING OPTIMIZED")
    print("="*70)
    
    # Kiểm tra file tồn tại
    if not os.path.exists(image_path):
        print(f"❌ Lỗi: Không tìm thấy file {image_path}")
        return
    
    # Load ảnh
    print(f"\n📥 Đang load ảnh: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Lỗi: Không thể đọc ảnh {image_path}")
        return
    
    print(f"✅ Đã load ảnh, kích thước: {image.shape}")
    
    # Preprocessing V4
    print("\n🔄 Đang chạy Preprocessing V4...")
    print("   → Bilateral filter (giảm nhiễu, giữ edge)")
    print("   → CLAHE mạnh hơn (tăng contrast)")
    print("   → Dilation (làm dày nét)")
    print("   → Padding lớn hơn (20%)")
    
    try:
        normalized, display_img, progress = preprocess_for_mnist(
            image,
            save_steps=True,  # Lưu từng bước
            output_dir="preprocessing_v4_test"
        )
        
        print(f"✅ Preprocessing thành công!")
        print(f"   → Đã lưu {len(progress)} bước vào: preprocessing_v4_test/")
        
        # Hiển thị ảnh cuối
        print("\n📊 Ảnh sau preprocessing:")
        print(f"   - Shape: {normalized.shape}")
        print(f"   - Min: {normalized.min():.3f}, Max: {normalized.max():.3f}")
        print(f"   - Mean: {normalized.mean():.3f}, Std: {normalized.std():.3f}")
        
        # Lưu ảnh cuối
        cv2.imwrite("preprocessing_v4_test/FINAL_28x28.png", display_img)
        print(f"\n✅ Đã lưu ảnh cuối: preprocessing_v4_test/FINAL_28x28.png")
        
        # Load model và predict (nếu có)
        model_path = 'models/mnist_model.h5'
        if os.path.exists(model_path):
            print(f"\n🤖 Đang load model: {model_path}")
            model = keras.models.load_model(model_path)
            
            # Predict
            print("🎯 Đang dự đoán...")
            prediction = model.predict(normalized, verbose=0)
            pred_label = np.argmax(prediction[0])
            confidence = prediction[0][pred_label] * 100
            
            print("\n" + "="*70)
            print("📈 KẾT QUẢ DỰ ĐOÁN")
            print("="*70)
            print(f"   🎯 Chữ số dự đoán: **{pred_label}**")
            print(f"   📊 Độ tin cậy: {confidence:.2f}%")
            print(f"   📊 Confidence score: {confidence/100:.4f}")
            
            # Top 3
            print(f"\n   📋 Top 3 dự đoán:")
            top3_idx = np.argsort(prediction[0])[-3:][::-1]
            for i, idx in enumerate(top3_idx, 1):
                prob = prediction[0][idx] * 100
                print(f"      {i}. Số {idx}: {prob:.2f}%")
            
            # Đánh giá
            if confidence >= 90:
                print(f"\n   ✅ Model RẤT TỰ TIN ({confidence:.1f}% >= 90%)")
            elif confidence >= 70:
                print(f"\n   ⚠️  Model TỰ TIN VỪA PHẢI ({confidence:.1f}% >= 70%)")
            else:
                print(f"\n   ❌ Model KHÔNG CHẮC CHẮN ({confidence:.1f}% < 70%)")
                print(f"      → Có thể ảnh chưa đủ rõ hoặc preprocessing cần cải thiện thêm")
        else:
            print(f"\n⚠️  Không tìm thấy model tại: {model_path}")
            print(f"   → Train model bằng cách chạy: python train_all.py")
        
    except Exception as e:
        print(f"\n❌ Lỗi khi preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH!")
    print("="*70)
    print(f"\n💡 Hướng dẫn:")
    print(f"   1. Mở thư mục: preprocessing_v4_test/")
    print(f"   2. Xem các ảnh step01, step02, ... đến FINAL_28x28.png")
    print(f"   3. Quan sát xem preprocessing có hoạt động tốt không")
    print(f"   4. Nếu kết quả chưa tốt:")
    print(f"      - Chụp lại ảnh với ánh sáng tốt hơn")
    print(f"      - Viết chữ rõ ràng hơn, nét không quá mỏng")
    print(f"      - Đảm bảo nền đơn giản (giấy trắng)")
    print("\n")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("🧪 TEST PREPROCESSING V4")
        print("="*70)
        print("\nCách sử dụng:")
        print("   python test_preprocessing_v4.py <đường_dẫn_ảnh>")
        print("\nVí dụ:")
        print("   python test_preprocessing_v4.py test_img/my_handwriting.jpg")
        print("   python test_preprocessing_v4.py \"C:/Users/YourName/Desktop/digit.png\"")
        print("\n💡 Tips:")
        print("   - Ảnh viết tay số trên giấy trắng")
        print("   - Chụp với ánh sáng đủ")
        print("   - Viết rõ ràng, không quá mỏng")
        print("\n")
        return
    
    image_path = sys.argv[1]
    test_preprocessing_v4(image_path)


if __name__ == "__main__":
    main()


