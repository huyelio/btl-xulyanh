"""
Script đơn giản để huấn luyện cả 2 models (MNIST và Shapes) trên local
Chạy: python train_all.py
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import random

print("="*70)
print("🚀 HUẤN LUYỆN CNN CHO MNIST VÀ SHAPES")
print("="*70)

# ============================================================================
# 1. SETUP
# ============================================================================
print("\n📦 Bước 1: Kiểm tra môi trường...")

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU'))} GPU(s)")

if not tf.config.list_physical_devices('GPU'):
    print("⚠️  WARNING: Không tìm thấy GPU! Training sẽ chậm hơn.")
    response = input("Tiếp tục với CPU? (y/n): ")
    if response.lower() != 'y':
        exit()

# Tạo thư mục lưu models
save_dir = 'models'
os.makedirs(save_dir, exist_ok=True)
print(f"✓ Sẽ lưu models vào: {save_dir}")

# ============================================================================
# 2. HUẤN LUYỆN MNIST MODEL
# ============================================================================
print("\n" + "="*70)
print("📊 Bước 2: HUẤN LUYỆN MNIST MODEL")
print("="*70)

# Load data
print("\n📥 Loading MNIST dataset...")
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = keras.datasets.mnist.load_data()

# Preprocess
x_train_mnist = x_train_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test_mnist = x_test_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train_mnist = keras.utils.to_categorical(y_train_mnist, 10)
y_test_mnist = keras.utils.to_categorical(y_test_mnist, 10)

print(f"✓ Train: {x_train_mnist.shape}, Test: {x_test_mnist.shape}")

# Hiển thị mẫu
fig, axes = plt.subplots(1, 5, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(x_train_mnist[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {np.argmax(y_train_mnist[i])}")
    ax.axis('off')
plt.suptitle('MNIST Samples')
plt.tight_layout()
plt.savefig('example_progress/mnist_samples.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Đã lưu ảnh mẫu MNIST")

# Build model
print("\n🏗️  Building MNIST model...")
mnist_model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

mnist_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

mnist_model.summary()

# Train
print("\n🎯 Training MNIST model...")
history_mnist = mnist_model.fit(
    x_train_mnist, y_train_mnist,
    batch_size=128,
    epochs=20,
    validation_data=(x_test_mnist, y_test_mnist),
    verbose=1
)

# Evaluate
test_loss, test_acc = mnist_model.evaluate(x_test_mnist, y_test_mnist, verbose=0)
print(f"\n✓ MNIST Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Save
mnist_path = os.path.join(save_dir, 'mnist_model.h5')
mnist_model.save(mnist_path)
print(f"✓ Saved: {mnist_path}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history_mnist.history['accuracy'], label='Train')
ax1.plot(history_mnist.history['val_accuracy'], label='Val')
ax1.set_title('MNIST Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history_mnist.history['loss'], label='Train')
ax2.plot(history_mnist.history['val_loss'], label='Val')
ax2.set_title('MNIST Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('example_progress/mnist_training_history.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Đã lưu history plot")

# ============================================================================
# 3. HUẤN LUYỆN SHAPES MODEL
# ============================================================================
print("\n" + "="*70)
print("📐 Bước 3: HUẤN LUYỆN SHAPES MODEL")
print("="*70)

# Functions to generate shapes
def generate_circle(img_size=64):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    radius = random.randint(15, 25)
    cv2.circle(img, (img_size//2, img_size//2), radius, 255, -1)
    return img

def generate_rectangle(img_size=64):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    w, h = random.randint(20, 40), random.randint(20, 40)
    x1, y1 = (img_size-w)//2, (img_size-h)//2
    cv2.rectangle(img, (x1, y1), (x1+w, y1+h), 255, -1)
    return img

def generate_triangle(img_size=64):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    center = img_size // 2
    size = random.randint(20, 30)
    pts = np.array([
        [center, center-size],
        [center-size, center+size],
        [center+size, center+size]
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], 255)
    return img

# Generate dataset
print("\n🎨 Generating shapes dataset...")
shapes_funcs = [generate_circle, generate_rectangle, generate_triangle]
X_shapes, y_shapes = [], []

for label, func in enumerate(shapes_funcs):
    shape_name = 'circle' if label==0 else 'rectangle' if label==1 else 'triangle'
    print(f"  Generating {shape_name}...")
    for _ in range(800):
        img = func()
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
            img = cv2.warpAffine(img, M, (64, 64))
        X_shapes.append(img)
        y_shapes.append(label)

X_shapes = np.array(X_shapes).reshape(-1, 64, 64, 1).astype('float32') / 255.0
y_shapes = keras.utils.to_categorical(y_shapes, 3)

# Split
x_train_shapes, x_test_shapes, y_train_shapes, y_test_shapes = train_test_split(
    X_shapes, y_shapes, test_size=0.2, random_state=42
)

print(f"✓ Train: {x_train_shapes.shape}, Test: {x_test_shapes.shape}")

# Show samples
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
shape_names = ['Circle', 'Rectangle', 'Triangle']
for i in range(3):
    idx = np.where(np.argmax(y_train_shapes, axis=1) == i)[0][0]
    axes[i].imshow(x_train_shapes[idx].reshape(64, 64), cmap='gray')
    axes[i].set_title(shape_names[i])
    axes[i].axis('off')
plt.suptitle('Shapes Samples')
plt.tight_layout()
plt.savefig('example_progress/shapes_samples.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Đã lưu ảnh mẫu Shapes")

# Build model
print("\n🏗️  Building Shapes model...")
shapes_model = keras.Sequential([
    layers.Input(shape=(64, 64, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

shapes_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

shapes_model.summary()

# Train
print("\n🎯 Training Shapes model...")
history_shapes = shapes_model.fit(
    x_train_shapes, y_train_shapes,
    batch_size=32,
    epochs=15,
    validation_data=(x_test_shapes, y_test_shapes),
    verbose=1
)

# Evaluate
test_loss, test_acc = shapes_model.evaluate(x_test_shapes, y_test_shapes, verbose=0)
print(f"\n✓ Shapes Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Save
shapes_path = os.path.join(save_dir, 'shapes_model.h5')
shapes_model.save(shapes_path)
print(f"✓ Saved: {shapes_path}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history_shapes.history['accuracy'], label='Train')
ax1.plot(history_shapes.history['val_accuracy'], label='Val')
ax1.set_title('Shapes Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history_shapes.history['loss'], label='Train')
ax2.plot(history_shapes.history['val_loss'], label='Val')
ax2.set_title('Shapes Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('example_progress/shapes_training_history.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Đã lưu history plot")

# ============================================================================
# 4. TỔNG KẾT
# ============================================================================
print("\n" + "="*70)
print("✅ HOÀN THÀNH!")
print("="*70)

mnist_loss, mnist_acc = mnist_model.evaluate(x_test_mnist, y_test_mnist, verbose=0)
shapes_loss, shapes_acc = shapes_model.evaluate(x_test_shapes, y_test_shapes, verbose=0)

print(f"\n📊 KẾT QUẢ CUỐI CÙNG:")
print(f"\nMNIST Model:")
print(f"  - Test Accuracy: {mnist_acc:.4f} ({mnist_acc*100:.2f}%)")
print(f"  - Test Loss: {mnist_loss:.4f}")
print(f"  - Saved: {mnist_path}")

print(f"\nShapes Model:")
print(f"  - Test Accuracy: {shapes_acc:.4f} ({shapes_acc*100:.2f}%)")
print(f"  - Test Loss: {shapes_loss:.4f}")
print(f"  - Saved: {shapes_path}")

print(f"\n📁 Models location: {save_dir}/")
print(f"📊 Training plots saved in: example_progress/")

print("\n🎉 Chúc mừng! Bạn đã huấn luyện thành công 2 models!")
print("\n💡 Tiếp theo:")
print("   1. Chạy web app: streamlit run app.py")
print("   2. Test với ảnh của bạn!")
print("="*70)

