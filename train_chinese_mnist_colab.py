"""
Google Colab Script để Huấn luyện Model Chinese MNIST
Hướng dẫn sử dụng:
1. Tạo một notebook mới trên Google Colab
2. Copy toàn bộ code này vào một cell
3. Chạy cell và làm theo hướng dẫn
4. File chinese_model.h5 sẽ được tự động tải về máy
"""

# ========== CELL 1: SETUP VÀ TẢI DỮ LIỆU ==========
print("=" * 60)
print("BƯỚC 1: CÀI ĐẶT MÔI TRƯỜNG VÀ TẢI DỮ LIỆU")
print("=" * 60)

# Import các thư viện cần thiết
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os
import zipfile

print("\n✓ Đã import thư viện")
print(f"TensorFlow version: {tf.__version__}")

# Kiểm tra GPU
print(f"\nGPU available: {tf.config.list_physical_devices('GPU')}")

# ========== PHƯƠNG ÁN 1: SỬ DỤNG KAGGLE API ==========
print("\n" + "=" * 60)
print("CÀI ĐẶT KAGGLE API")
print("=" * 60)
print("\nHƯỚNG DẪN:")
print("1. Truy cập: https://www.kaggle.com/settings/account")
print("2. Scroll xuống phần 'API', click 'Create New Token'")
print("3. File kaggle.json sẽ được tải về")
print("4. Upload file kaggle.json vào Colab bằng Files panel (bên trái)")
print("\nSau khi upload, chạy tiếp để cài đặt...")

# Upload kaggle.json
from google.colab import files
print("\n📤 Vui lòng upload file kaggle.json:")
uploaded = files.upload()

# Cài đặt Kaggle
!pip install -q kaggle

# Di chuyển kaggle.json vào đúng thư mục
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

print("\n✓ Đã cài đặt Kaggle API")

# Tải dataset Chinese MNIST
print("\n" + "=" * 60)
print("ĐANG TẢI DATASET CHINESE MNIST...")
print("=" * 60)

!kaggle datasets download -d gpreda/chinese-mnist

print("\n✓ Đã tải xong dataset")

# Giải nén file
print("\n📦 Đang giải nén...")
with zipfile.ZipFile('chinese-mnist.zip', 'r') as zip_ref:
    zip_ref.extractall('chinese_mnist_data')

print("✓ Đã giải nén xong")

# Kiểm tra files
print("\n📁 Files trong dataset:")
!ls -lh chinese_mnist_data/

# ========== BƯỚC 2: ĐỌC VÀ CHUẨN BỊ DỮ LIỆU ==========
print("\n" + "=" * 60)
print("BƯỚC 2: ĐỌC VÀ CHUẨN BỊ DỮ LIỆU")
print("=" * 60)

# Đọc CSV file
csv_path = 'chinese_mnist_data/chinese_mnist.csv'
print(f"\n📖 Đang đọc file: {csv_path}")
df = pd.read_csv(csv_path)

print(f"\n✓ Đã đọc xong. Shape: {df.shape}")
print(f"\nCác cột trong dataset:")
print(df.columns.tolist())
print(f"\nMẫu dữ liệu đầu tiên:")
print(df.head())

# Phân tích labels
print(f"\n📊 Phân bố labels:")
print(df['character'].value_counts().sort_index())
print(f"\nSố lượng labels khác nhau: {df['character'].nunique()}")

# Tách X (ảnh) và y (labels)
print("\n" + "=" * 60)
print("CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN")
print("=" * 60)

# Dataset Chinese MNIST có các cột: suite_id, sample_id, code, value, character, và các pixel
# Các pixel columns là từ 'pixel1' đến 'pixel4096' (64x64 = 4096)
pixel_columns = [col for col in df.columns if col.startswith('pixel')]
print(f"\n✓ Tìm thấy {len(pixel_columns)} pixel columns")

# Tách features (X) và labels (y)
X = df[pixel_columns].values
y = df['code'].values  # 'code' là nhãn số từ 1-15

print(f"\nShape của X: {X.shape}")
print(f"Shape của y: {y.shape}")
print(f"Giá trị y: từ {y.min()} đến {y.max()}")

# Chuyển labels về 0-indexed (từ 0-14 thay vì 1-15)
y = y - 1
print(f"✓ Đã chuyển labels về 0-indexed: từ {y.min()} đến {y.max()}")

# Reshape ảnh về (N, 64, 64, 1)
print("\n🔄 Reshape ảnh...")
X = X.reshape(-1, 64, 64, 1)
print(f"✓ Shape sau reshape: {X.shape}")

# Chuẩn hóa về [0, 1]
print("\n📐 Chuẩn hóa pixel values...")
X = X.astype('float32') / 255.0
print(f"✓ Pixel range: [{X.min():.3f}, {X.max():.3f}]")

# Chuyển labels sang categorical (one-hot encoding)
print("\n🔢 Chuyển labels sang categorical...")
y_categorical = keras.utils.to_categorical(y, num_classes=15)
print(f"✓ Shape của y_categorical: {y_categorical.shape}")

# Chia train/validation
print("\n✂️ Chia dữ liệu train/validation...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Validation set: {X_val.shape[0]} samples")

# ========== BƯỚC 3: XÂY DỰNG MODEL ==========
print("\n" + "=" * 60)
print("BƯỚC 3: XÂY DỰNG MODEL CNN")
print("=" * 60)

model = keras.Sequential([
    # Convolutional Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Convolutional Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Convolutional Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Convolutional Block 4
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    
    # Output layer: 15 classes
    layers.Dense(15, activation='softmax')
])

print("\n✓ Đã tạo model")
print("\n📋 Model Summary:")
model.summary()

# ========== BƯỚC 4: COMPILE MODEL ==========
print("\n" + "=" * 60)
print("BƯỚC 4: COMPILE MODEL")
print("=" * 60)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Đã compile model với:")
print("  - Optimizer: Adam")
print("  - Loss: Categorical Crossentropy")
print("  - Metrics: Accuracy")

# ========== BƯỚC 5: HUẤN LUYỆN ==========
print("\n" + "=" * 60)
print("BƯỚC 5: HUẤN LUYỆN MODEL")
print("=" * 60)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

print("\n🚀 Bắt đầu huấn luyện...")
print("Lưu ý: Quá trình có thể mất 10-20 phút tùy thuộc vào GPU\n")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n✅ ĐÃ HOÀN THÀNH HUẤN LUYỆN!")

# ========== BƯỚC 6: ĐÁNH GIÁ ==========
print("\n" + "=" * 60)
print("BƯỚC 6: ĐÁNH GIÁ MODEL")
print("=" * 60)

# Đánh giá trên validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"\n📊 Kết quả trên Validation Set:")
print(f"  - Loss: {val_loss:.4f}")
print(f"  - Accuracy: {val_accuracy*100:.2f}%")

# Vẽ biểu đồ training history
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# ========== BƯỚC 7: LƯU VÀ TẢI VỀ MODEL ==========
print("\n" + "=" * 60)
print("BƯỚC 7: LƯU VÀ TẢI MODEL VỀ MÁY")
print("=" * 60)

# Lưu model
model_filename = 'chinese_model.h5'
print(f"\n💾 Đang lưu model: {model_filename}")
model.save(model_filename)
print("✓ Đã lưu model")

# Kiểm tra file size
file_size = os.path.getsize(model_filename) / (1024 * 1024)  # Convert to MB
print(f"\n📦 Kích thước file: {file_size:.2f} MB")

# Tải về máy
print("\n📥 Tự động tải file về máy của bạn...")
files.download(model_filename)

print("\n" + "=" * 60)
print("✅ HOÀN TẤT!")
print("=" * 60)
print(f"""
TỔNG KẾT:
✓ Model đã được huấn luyện thành công
✓ Validation Accuracy: {val_accuracy*100:.2f}%
✓ File {model_filename} đã được tải về máy
✓ Bạn có thể sử dụng file này trong app.py

BƯỚC TIẾP THEO:
1. Di chuyển file {model_filename} vào thư mục: btl_final/models/
2. Chạy file app.py để sử dụng model

LABELS MAPPING:
0: 零 (zero)      5: 五 (five)       10: 十 (ten)
1: 一 (one)       6: 六 (six)        11: 百 (hundred)
2: 二 (two)       7: 七 (seven)      12: 千 (thousand)
3: 三 (three)     8: 八 (eight)      13: 万 (ten thousand)
4: 四 (four)      9: 九 (nine)       14: 亿 (hundred million)
""")

