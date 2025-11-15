"""
Script huấn luyện CNN model cho MNIST
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from preprocessing import ImagePreprocessor


def create_mnist_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Tạo CNN model cho MNIST
    
    Architecture:
    - Conv2D (32 filters) + MaxPooling
    - Conv2D (64 filters) + MaxPooling
    - Conv2D (64 filters) + MaxPooling
    - Flatten + Dense(128) + Dense(10)
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Convolutional Block 1
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional Block 2
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional Block 3
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def load_and_preprocess_mnist():
    """Tải và tiền xử lý MNIST dataset"""
    print("Đang tải MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"Train shape: {x_train.shape}")
    print(f"Test shape: {x_test.shape}")
    
    # Reshape và chuẩn hóa
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"✓ Đã chuẩn bị dữ liệu")
    print(f"  - X_train: {x_train.shape}, Y_train: {y_train.shape}")
    print(f"  - X_test: {x_test.shape}, Y_test: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)


def plot_training_history(history, save_path='models/mnist_history.png'):
    """Vẽ biểu đồ quá trình huấn luyện"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ training history tại {save_path}")


def demonstrate_preprocessing():
    """Demo pipeline tiền xử lý với một vài ảnh MNIST"""
    print("\n" + "="*60)
    print("DEMO: Pipeline tiền xử lý ảnh MNIST")
    print("="*60)
    
    # Load một vài ảnh mẫu
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    
    # Tạo preprocessor với save_progress=True
    preprocessor = ImagePreprocessor(save_progress=True, output_dir="example_progress")
    
    # Lấy một ảnh mẫu (chữ số 7)
    sample_indices = [0, 100, 200]
    
    for idx in sample_indices:
        sample_img = x_train[idx]
        sample_label = y_train[idx]
        
        print(f"\nXử lý ảnh mẫu {idx} (nhãn: {sample_label})...")
        
        # Chạy full pipeline
        processed = preprocessor.full_pipeline(sample_img, for_mnist=True)
        
        # Lưu progress images
        preprocessor.save_progress_images(prefix=f"mnist_sample_{idx}")
        
        print(f"✓ Đã xử lý và lưu {len(preprocessor.get_progress_images())} bước")


def train_mnist_model(epochs=30, batch_size=128, save_dir='models', use_augmentation=True):
    """
    Huấn luyện MNIST model với Data Augmentation
    
    Args:
        epochs: Số epoch (mặc định 30 với augmentation)
        batch_size: Batch size
        save_dir: Thư mục lưu model
        use_augmentation: Sử dụng Data Augmentation để giải quyết Domain Gap
    """
    # Tạo thư mục nếu chưa có
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("HUẤN LUYỆN CNN MODEL CHO MNIST")
    print("="*60)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    
    # 🚀 DATA AUGMENTATION - Giải pháp cho Domain Gap!
    if use_augmentation:
        print("\n🎨 Tạo Data Augmentation Generator...")
        print("   → Giúp model quen với ảnh bị lệch, xoay, zoom...")
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        datagen = ImageDataGenerator(
            rotation_range=15,       # Ngẫu nhiên xoay +/- 15 độ
            width_shift_range=0.15,  # Ngẫu nhiên dịch ngang 15%
            height_shift_range=0.15, # Ngẫu nhiên dịch dọc 15%
            zoom_range=0.15,         # Ngẫu nhiên phóng to/thu nhỏ 15%
            shear_range=0.1,         # Ngẫu nhiên làm méo ảnh
            fill_mode='constant',    # Fill phần trống bằng 0 (màu đen)
            cval=0
        )
        
        # Fit datagen vào training data
        datagen.fit(x_train)
        print("✓ Data Augmentation ready!")
    
    # Tạo model
    print("\nTạo model...")
    model = create_mnist_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # In tóm tắt model
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(save_dir, 'mnist_model_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\nBắt đầu huấn luyện ({epochs} epochs, batch_size={batch_size})...")
    if use_augmentation:
        print("   → Sử dụng Data Augmentation - model sẽ khoan dung hơn với ảnh thực tế!")
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // batch_size,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
    
    # Evaluate
    print("\nĐánh giá model trên test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✓ Test accuracy: {test_acc:.4f}")
    print(f"✓ Test loss: {test_loss:.4f}")
    
    # Save final model
    model_path = os.path.join(save_dir, 'mnist_model.h5')
    model.save(model_path)
    print(f"\n✓ Đã lưu model tại {model_path}")
    
    # Plot history
    plot_training_history(history, save_path=os.path.join(save_dir, 'mnist_history.png'))
    
    # Test với một vài predictions
    print("\n" + "="*60)
    print("TEST: Dự đoán một vài ảnh mẫu")
    print("="*60)
    
    sample_indices = [0, 100, 200, 500, 1000]
    predictions = model.predict(x_test[sample_indices], verbose=0)
    
    for i, idx in enumerate(sample_indices):
        true_label = np.argmax(y_test[idx])
        pred_label = np.argmax(predictions[i])
        confidence = predictions[i][pred_label]
        
        status = "✓" if true_label == pred_label else "✗"
        print(f"{status} Ảnh {idx}: True={true_label}, Pred={pred_label}, Confidence={confidence:.4f}")
    
    return model, history


if __name__ == "__main__":
    # Demo preprocessing pipeline
    demonstrate_preprocessing()
    
    # Train model với Data Augmentation
    model, history = train_mnist_model(
        epochs=30,  # Tăng epochs vì model phải học bài toán khó hơn
        batch_size=128,
        save_dir='models',
        use_augmentation=True  # Bật Data Augmentation để giải quyết Domain Gap
    )
    
    print("\n" + "="*60)
    print("✓ HOÀN THÀNH HUẤN LUYỆN MNIST MODEL")
    print("✓ Model đã được train với Data Augmentation!")
    print("✓ Bây giờ model sẽ chính xác hơn với ảnh viết tay thực tế!")
    print("="*60)

