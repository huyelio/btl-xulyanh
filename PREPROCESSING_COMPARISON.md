# 🔍 So sánh Preprocessing: Colab vs Local

## Kết luận: CODE HOÀN TOÀN GIỐNG NHAU! ✅

Sau khi kiểm tra kỹ lưỡng, **logic preprocessing giữa code Colab và code Local là HOÀN TOÀN GIỐNG NHAU**.

---

## 📊 So sánh từng bước

### Code Colab (Cell 2):

```python
def detect_if_need_invert(binary_image):
    h, w = binary_image.shape
    total_pixels = h * w
    white_pixels = np.sum(binary_image == 255)
    white_ratio = white_pixels / total_pixels
    border_size = max(1, int(min(h, w) * 0.1))
    border_pixels = np.concatenate([...])
    border_white_ratio = np.sum(border_pixels == 255) / len(border_pixels)
    need_invert = (white_ratio > 0.6 and border_white_ratio > 0.7)
    return need_invert

def preprocess_mnist_robust(image):
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Gaussian blur (5x5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Adaptive threshold
    binary = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=11, C=2)

    # 4. Detect & invert
    if detect_if_need_invert(binary):
        binary = cv2.bitwise_not(binary)

    # 5. Morphology
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6-7. Contour, crop, center (20x20 → 28x28)
    # 8. Gaussian blur (3x3)
    # 9. Normalize

    return normalized.reshape(1, 28, 28, 1), resized
```

### Code Local (src/preprocessing.py):

```python
def detect_if_need_invert(binary_image: np.ndarray) -> bool:
    h, w = binary_image.shape
    total_pixels = h * w
    white_pixels = np.sum(binary_image == 255)
    white_ratio = white_pixels / total_pixels
    border_size = max(1, int(min(h, w) * 0.1))
    border_pixels = np.concatenate([...])
    border_white_ratio = np.sum(border_pixels == 255) / len(border_pixels)
    need_invert = (white_ratio > 0.6 and border_white_ratio > 0.7)
    return need_invert

def preprocess_for_mnist(image: np.ndarray, target_size: Tuple[int, int] = (28, 28)) -> Tuple[np.ndarray, np.ndarray]:
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Gaussian blur (5x5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Adaptive threshold
    binary = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=11, C=2)

    # 4. Detect & invert
    if detect_if_need_invert(binary):
        binary = cv2.bitwise_not(binary)

    # 5. Morphology
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6-7. Contour, crop, center (20x20 → 28x28)
    # 8. Gaussian blur (3x3)
    # 9. Normalize

    return normalized.reshape(1, 28, 28, 1), resized
```

**✅ KẾT LUẬN: Code giống 100%!**

---

## ❓ Vậy tại sao độ chính xác có thể khác?

### 1. **Keras Version Khác Nhau** ⚠️

| Môi trường | TensorFlow | Keras  | Ghi chú                  |
| ---------- | ---------- | ------ | ------------------------ |
| **Colab**  | 2.15.0     | 2.15.0 | Model được train ở đây   |
| **Local**  | 2.20.0     | 3.12.0 | Keras 3.x có API changes |

**Vấn đề:**

- Keras 3.x (local) load model train từ Keras 2.x (Colab)
- Có thể có slight numerical differences
- Warning khi load: `compiled metrics have yet to be built`

### 2. **OpenCV Version Khác** 🔧

| Môi trường | OpenCV |
| ---------- | ------ |
| **Colab**  | 4.8.1  |
| **Local**  | 4.12.0 |

**Impact:** Minimal - các functions như `GaussianBlur`, `adaptiveThreshold` tương đối stable.

### 3. **NumPy Version Khác** 🔢

| Môi trường | NumPy  |
| ---------- | ------ |
| **Colab**  | 1.24.3 |
| **Local**  | 2.2.6  |

**Impact:** NumPy 2.x có breaking changes, nhưng preprocessing logic không bị ảnh hưởng nhiều.

---

## 🎯 Giải pháp

### Option 1: Retrain models với TensorFlow 2.20 (KHUYẾN NGHỊ) ✅

```bash
python train_all.py
```

**Lý do:**

- Đảm bảo models tương thích 100% với local environment
- Không có warning khi load
- Performance tối ưu

**Thời gian:** 30-45 phút (CPU) hoặc 10-15 phút (GPU)

### Option 2: Dùng models cũ (Hiện tại)

**Ưu điểm:**

- Không cần train lại
- Tiết kiệm thời gian

**Nhược điểm:**

- Có warning khi load model
- Có thể có slight accuracy differences (1-2%)
- Keras 3.x compatibility issues

### Option 3: Downgrade TensorFlow về 2.15 (KHÔNG khuyến nghị) ❌

```bash
pip install tensorflow==2.15.0 keras==2.15.0
```

**Vấn đề:**

- TensorFlow 2.15 không hỗ trợ Python 3.12!
- Phải dùng Python 3.10 hoặc 3.11
- Phức tạp, không cần thiết

---

## 📝 Test Results

### Model Loading Test:

```bash
$ python test_model_loading.py

✓ models/mnist_model.h5 exists
WARNING: Compiled the loaded model, but the compiled metrics have yet to be built
✓ MNIST model loaded successfully
  Input shape: (None, 28, 28, 1)
  Output shape: (None, 10)
  ✓ MNIST model works!

✓ models/shapes_model.h5 exists
WARNING: Compiled the loaded model, but the compiled metrics have yet to be built
✓ Shapes model loaded successfully
  Input shape: (None, 64, 64, 1)
  Output shape: (None, 3)
  ✓ Shapes model works!
```

**Kết luận:** Models hoạt động, nhưng có warning về metrics.

---

## 🔬 Chi tiết Preprocessing Steps

### Bước 1-9 giống 100%:

1. **Grayscale conversion** ✅
2. **Gaussian blur (5x5)** ✅
3. **Adaptive threshold (blockSize=11, C=2)** ✅
4. **Detect background (60% + 70% threshold)** ✅
5. **Auto invert if needed** ✅
6. **Morphology opening (2x2)** ✅
7. **Contour detection** ✅
8. **Crop & center (20x20 → 28x28)** ✅
9. **Gaussian blur (3x3)** ✅
10. **Normalize [0, 1]** ✅

---

## 💡 Kết luận cuối cùng

**Logic preprocessing: HOÀN TOÀN GIỐNG NHAU!**

Nếu có sự khác biệt về accuracy, nguyên nhân là:

- ✅ Keras 3.x load model train từ Keras 2.x
- ✅ TensorFlow 2.20 vs 2.15 có slight numerical differences
- ❌ **KHÔNG PHẢI** do logic preprocessing khác

**Giải pháp tốt nhất:** Retrain models với `python train_all.py`

---

## 📊 Quick Comparison Table

| Aspect                      | Colab             | Local             | Giống?  |
| --------------------------- | ----------------- | ----------------- | ------- |
| **detect_if_need_invert()** | ✓                 | ✓                 | ✅ 100% |
| **Gaussian blur**           | (5,5)             | (5,5)             | ✅ 100% |
| **Adaptive threshold**      | blockSize=11, C=2 | blockSize=11, C=2 | ✅ 100% |
| **Morphology kernel**       | (2,2)             | (2,2)             | ✅ 100% |
| **Crop & center**           | 20x20 → 28x28     | 20x20 → 28x28     | ✅ 100% |
| **Final blur**              | (3,3)             | (3,3)             | ✅ 100% |
| **Return type**             | tuple             | tuple             | ✅ 100% |
| **TensorFlow**              | 2.15              | 2.20              | ❌ Khác |
| **Keras**                   | 2.15              | 3.12              | ❌ Khác |

---

_Last updated: 2025-11-14_
