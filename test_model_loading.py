"""
Script test loading models để kiểm tra compatibility
"""
import os
import numpy as np
from tensorflow import keras

print("Kiểm tra models...")

# Check files exist
mnist_path = 'models/mnist_model.h5'
shapes_path = 'models/shapes_model.h5'

if os.path.exists(mnist_path):
    print(f"✓ {mnist_path} exists")
    try:
        mnist_model = keras.models.load_model(mnist_path)
        print(f"✓ MNIST model loaded successfully")
        print(f"  Input shape: {mnist_model.input_shape}")
        print(f"  Output shape: {mnist_model.output_shape}")
        
        # Test prediction
        test_input = np.random.rand(1, 28, 28, 1).astype('float32')
        pred = mnist_model.predict(test_input, verbose=0)
        print(f"  Test prediction shape: {pred.shape}")
        print(f"  ✓ MNIST model works!")
    except Exception as e:
        print(f"  ✗ Error loading MNIST: {e}")
else:
    print(f"✗ {mnist_path} NOT FOUND")

if os.path.exists(shapes_path):
    print(f"\n✓ {shapes_path} exists")
    try:
        shapes_model = keras.models.load_model(shapes_path)
        print(f"✓ Shapes model loaded successfully")
        print(f"  Input shape: {shapes_model.input_shape}")
        print(f"  Output shape: {shapes_model.output_shape}")
        
        # Test prediction
        test_input = np.random.rand(1, 64, 64, 1).astype('float32')
        pred = shapes_model.predict(test_input, verbose=0)
        print(f"  Test prediction shape: {pred.shape}")
        print(f"  ✓ Shapes model works!")
    except Exception as e:
        print(f"  ✗ Error loading Shapes: {e}")
else:
    print(f"\n✗ {shapes_path} NOT FOUND")

print("\n" + "="*50)
print("Nếu cả 2 models đều works → Vấn đề có thể là:")
print("1. Keras 3.x load model train từ Keras 2.x")
print("2. Model cần được train lại với TensorFlow 2.20")
print("="*50)

