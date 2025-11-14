"""
Script huấn luyện CNN model cho nhận dạng hình học
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from preprocessing import ImagePreprocessor


def create_shapes_model(input_shape=(64, 64, 1), num_classes=3):
    """
    Tạo CNN model cho Shapes (circle, rectangle, triangle)
    
    Architecture:
    - Conv2D (32 filters) + MaxPooling
    - Conv2D (64 filters) + MaxPooling
    - Conv2D (128 filters) + MaxPooling
    - Flatten + Dense(128) + Dense(3)
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Convolutional Block 1
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional Block 2
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional Block 3
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def load_shapes_dataset(data_dir='data/shapes', img_size=64):
    """
    Tải shapes dataset từ thư mục
    
    Args:
        data_dir: Thư mục chứa dữ liệu (train/test/shape_name/*.png)
        img_size: Kích thước ảnh
    
    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    shapes = ['circle', 'rectangle', 'triangle']
    shape_to_label = {shape: i for i, shape in enumerate(shapes)}
    
    def load_images_from_folder(folder_path):
        images = []
        labels = []
        
        for shape in shapes:
            shape_folder = os.path.join(folder_path, shape)
            if not os.path.exists(shape_folder):
                print(f"⚠ Thư mục {shape_folder} không tồn tại!")
                continue
            
            label = shape_to_label[shape]
            files = [f for f in os.listdir(shape_folder) if f.endswith('.png')]
            
            for filename in files:
                img_path = os.path.join(shape_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize nếu cần
                    if img.shape != (img_size, img_size):
                        img = cv2.resize(img, (img_size, img_size))
                    
                    images.append(img)
                    labels.append(label)
            
            print(f"  Đã tải {len(files)} ảnh {shape}")
        
        return np.array(images), np.array(labels)
    
    print("Đang tải Shapes dataset...")
    
    # Load train set
    train_folder = os.path.join(data_dir, 'train')
    x_train, y_train = load_images_from_folder(train_folder)
    
    # Load test set
    test_folder = os.path.join(data_dir, 'test')
    x_test, y_test = load_images_from_folder(test_folder)
    
    # Reshape và chuẩn hóa
    x_train = x_train.reshape(-1, img_size, img_size, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, img_size, img_size, 1).astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, len(shapes))
    y_test = keras.utils.to_categorical(y_test, len(shapes))
    
    print(f"✓ Đã tải dataset")
    print(f"  - X_train: {x_train.shape}, Y_train: {y_train.shape}")
    print(f"  - X_test: {x_test.shape}, Y_test: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)


def plot_training_history(history, save_path='models/shapes_history.png'):
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
    """Demo pipeline tiền xử lý với các hình mẫu"""
    print("\n" + "="*60)
    print("DEMO: Pipeline tiền xử lý ảnh Shapes")
    print("="*60)
    
    # Kiểm tra xem có dữ liệu mẫu không
    demo_files = [
        "example_progress/demo_circle.png",
        "example_progress/demo_rectangle.png",
        "example_progress/demo_triangle.png"
    ]
    
    # Tạo preprocessor
    preprocessor = ImagePreprocessor(save_progress=True, output_dir="example_progress")
    
    for demo_file in demo_files:
        if os.path.exists(demo_file):
            shape_name = os.path.basename(demo_file).replace('demo_', '').replace('.png', '')
            print(f"\nXử lý {shape_name}...")
            
            img = cv2.imread(demo_file, cv2.IMREAD_GRAYSCALE)
            
            # Chạy full pipeline
            processed = preprocessor.full_pipeline(img, for_mnist=False)
            
            # Lưu progress images
            preprocessor.save_progress_images(prefix=f"shapes_{shape_name}")
            
            print(f"✓ Đã xử lý và lưu {len(preprocessor.get_progress_images())} bước")


def train_shapes_model(data_dir='data/shapes', epochs=20, batch_size=32, save_dir='models'):
    """
    Huấn luyện Shapes model
    
    Args:
        data_dir: Thư mục chứa dữ liệu
        epochs: Số epoch
        batch_size: Batch size
        save_dir: Thư mục lưu model
    """
    # Tạo thư mục nếu chưa có
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("HUẤN LUYỆN CNN MODEL CHO SHAPES")
    print("="*60)
    
    # Load data
    try:
        (x_train, y_train), (x_test, y_test) = load_shapes_dataset(data_dir)
    except Exception as e:
        print(f"✗ Lỗi khi tải dataset: {e}")
        print(f"⚠ Vui lòng chạy generate_shapes.py trước để tạo dataset!")
        return None, None
    
    # Tạo model
    print("\nTạo model...")
    model = create_shapes_model()
    
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
            patience=7,
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
            os.path.join(save_dir, 'shapes_model_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\nBắt đầu huấn luyện ({epochs} epochs, batch_size={batch_size})...")
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
    model_path = os.path.join(save_dir, 'shapes_model.h5')
    model.save(model_path)
    print(f"\n✓ Đã lưu model tại {model_path}")
    
    # Plot history
    plot_training_history(history, save_path=os.path.join(save_dir, 'shapes_history.png'))
    
    # Test với một vài predictions
    print("\n" + "="*60)
    print("TEST: Dự đoán một vài ảnh mẫu")
    print("="*60)
    
    shapes = ['circle', 'rectangle', 'triangle']
    sample_indices = [0, 50, 100, 150, 200]
    predictions = model.predict(x_test[sample_indices], verbose=0)
    
    for i, idx in enumerate(sample_indices):
        true_label = np.argmax(y_test[idx])
        pred_label = np.argmax(predictions[i])
        confidence = predictions[i][pred_label]
        
        status = "✓" if true_label == pred_label else "✗"
        print(f"{status} Ảnh {idx}: True={shapes[true_label]}, "
              f"Pred={shapes[pred_label]}, Confidence={confidence:.4f}")
    
    return model, history


if __name__ == "__main__":
    # Kiểm tra xem đã có dataset chưa
    if not os.path.exists('data/shapes/train'):
        print("⚠ Dataset chưa được tạo!")
        print("Đang tạo dataset...")
        from generate_shapes import generate_dataset
        generate_dataset(
            output_dir='data/shapes',
            num_samples_per_class=1000,
            img_size=64,
            add_augmentation=True
        )
    
    # Demo preprocessing pipeline
    demonstrate_preprocessing()
    
    # Train model
    model, history = train_shapes_model(
        data_dir='data/shapes',
        epochs=20,
        batch_size=32,
        save_dir='models'
    )
    
    if model is not None:
        print("\n" + "="*60)
        print("✓ HOÀN THÀNH HUẤN LUYỆN SHAPES MODEL")
        print("="*60)

