# 🛠️ Hướng dẫn Cài đặt và Sử dụng

Hướng dẫn chi tiết để chạy dự án CNN nhận dạng MNIST và Shapes trên máy local.

---

## 📋 Yêu cầu Hệ thống

### Tối thiểu:

- **Python**: 3.8 trở lên
- **RAM**: 4GB+
- **Disk**: 2GB trống
- **OS**: Windows/Linux/macOS

### Khuyến nghị:

- **Python**: 3.10
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU với CUDA (tùy chọn, giúp training nhanh hơn)

---

## 🚀 Cài đặt Nhanh (3 bước)

### Bước 1: Clone/Download dự án

```bash
# Nếu có git
git clone <repo-url>
cd btl_final

# Hoặc download ZIP và giải nén
```

### Bước 2: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

**Lưu ý Windows**: Nếu gặp lỗi, thử:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 3: Kiểm tra cài đặt

```bash
python -c "import tensorflow; print(tensorflow.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import streamlit; print(streamlit.__version__)"
```

Nếu không có lỗi → OK! ✅

---

## 📦 Cấu trúc Thư viện

File `requirements.txt` bao gồm:

```
tensorflow>=2.10.0
opencv-python>=4.7.0
streamlit>=1.28.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
pillow>=9.3.0
```

**Dung lượng download**: ~500MB (TensorFlow chiếm nhiều nhất)

---

## 🎯 Hướng dẫn Sử dụng

### Option 1: Chạy toàn bộ (Khuyến nghị)

**Bước 1: Training (một lần duy nhất)**

```bash
python train_all.py
```

Script này sẽ:

- ✅ Load MNIST dataset
- ✅ Generate Shapes dataset
- ✅ Train cả 2 models
- ✅ Save models vào `models/`
- ✅ Tạo training plots trong `example_progress/`

**Thời gian:**

- CPU: 30-45 phút
- GPU: 8-12 phút

**Output:**

```
models/
├── mnist_model.h5      (~2MB)
└── shapes_model.h5     (~3MB)

example_progress/
├── mnist_samples.png
├── mnist_training_history.png
├── shapes_samples.png
└── shapes_training_history.png
```

**Bước 2: Chạy Web App**

```bash
streamlit run app.py
```

Tự động mở browser tại: `http://localhost:8501`

---

### Option 2: Training riêng lẻ

Nếu bạn chỉ muốn train 1 model:

**MNIST only:**

```bash
python src/train_mnist.py
```

**Shapes only:**

```bash
# Generate data trước
python src/generate_shapes.py

# Rồi train
python src/train_shapes.py
```

---

## 🖥️ Sử dụng Web App

### 1. Khởi động app

```bash
streamlit run app.py
```

### 2. Giao diện

**Cột trái:**

- Chọn chế độ: "Chữ số (MNIST)" hoặc "Hình học (Shapes)"
- Upload ảnh (PNG, JPG, JPEG)

**Cột phải:**

- Nhấn "Nhận dạng"
- Xem kết quả:
  - Prediction chính
  - Confidence score
  - Top 3 predictions
  - Ảnh sau xử lý

### 3. Tips

**Ảnh tốt nhất:**

- ✅ Rõ nét, không bị mờ
- ✅ 1 chữ số hoặc 1 hình duy nhất
- ✅ Nền trắng hoặc nền đen đều OK
- ✅ Kích thước bất kỳ (app tự resize)

**Ảnh không tốt:**

- ❌ Nhiều chữ số/hình trong 1 ảnh
- ❌ Quá nhỏ (<20x20 pixels)
- ❌ Quá mờ hoặc nhiễu nhiều
- ❌ Chữ nghệ thuật, font fancy

---

## 📊 Demo với Preprocessing Pipeline

Để xem chi tiết các bước xử lý ảnh:

```bash
python src/demo_preprocessing.py
```

Output:

- Ảnh từng bước xử lý trong `example_progress/`
- So sánh trước/sau
- Summary table

---

## 🔧 Troubleshooting

### Lỗi 1: Không tìm thấy module

**Lỗi:**

```
ModuleNotFoundError: No module named 'tensorflow'
```

**Giải pháp:**

```bash
pip install tensorflow
# Hoặc
pip install -r requirements.txt
```

---

### Lỗi 2: GPU không được detect

**Lỗi:**

```
GPU available: []
```

**Không phải lỗi nghiêm trọng!** Training vẫn chạy được với CPU (chỉ chậm hơn).

**Nếu bạn có GPU NVIDIA và muốn dùng:**

1. Cài CUDA Toolkit
2. Cài cuDNN
3. Cài `tensorflow-gpu`

Nhưng không bắt buộc! CPU vẫn OK.

---

### Lỗi 3: Model chưa được load

**Lỗi trong app:**

```
❌ Model chưa được tải!
```

**Giải pháp:**

```bash
# Chạy training trước
python train_all.py

# Kiểm tra models có tồn tại
ls models/
# Phải có: mnist_model.h5, shapes_model.h5
```

---

### Lỗi 4: Port already in use

**Lỗi:**

```
Address already in use
```

**Giải pháp:**

```bash
# Dùng port khác
streamlit run app.py --server.port 8502

# Hoặc kill process cũ (Windows)
taskkill /F /IM streamlit.exe

# Linux/Mac
pkill -9 streamlit
```

---

### Lỗi 5: Out of Memory

**Lỗi khi training:**

```
ResourceExhaustedError: OOM when allocating tensor
```

**Giải pháp:**

1. **Giảm batch_size** trong `train_all.py`:

```python
# Tìm dòng:
batch_size=128,  # MNIST
# Đổi thành:
batch_size=64,   # hoặc 32

# Tương tự với Shapes
batch_size=32,   # Shapes
# Đổi thành:
batch_size=16,
```

2. **Close các app khác** đang chạy

3. **Restart máy** rồi thử lại

---

### Lỗi 6: Nhận dạng sai

**Nguyên nhân:**

- Ảnh không rõ
- Nhiều đối tượng trong ảnh
- Font chữ quá khác MNIST

**Giải pháp:**

1. Thử ảnh rõ hơn
2. Crop để chỉ còn 1 chữ số/hình
3. Test với ảnh sample trước (ảnh trong `test_img/`)

---

## 💡 Tips & Best Practices

### Training:

1. **Lần đầu chạy:**

   - Chạy `train_all.py` một lần
   - Đợi hoàn thành
   - Models sẽ được lưu

2. **Không cần train lại** trừ khi:

   - Muốn cải thiện accuracy
   - Thay đổi model architecture
   - Xóa mất models

3. **Monitor training:**
   - Xem accuracy mỗi epoch
   - Target: 99%+ cho cả 2 models
   - Nếu <95% → có vấn đề

### Web App:

1. **Khởi động:**

   - Đảm bảo models đã train
   - Chạy trong terminal riêng
   - Không đóng terminal khi đang dùng

2. **Upload ảnh:**

   - PNG hoặc JPG
   - Kích thước bất kỳ
   - Nền trắng/đen đều OK

3. **Đọc kết quả:**
   - Prediction chính: label dự đoán
   - Confidence: độ chắc chắn (cao = tốt)
   - Top 3: xem alternatives

### Performance:

1. **Nếu chậm:**

   - Close các app không cần
   - Giảm batch_size khi training
   - Dùng GPU nếu có

2. **Nếu hết RAM:**
   - Giảm batch_size
   - Train từng model riêng
   - Restart máy

---

## 📖 Chi tiết Kỹ thuật

### Preprocessing Pipeline

**MNIST (28x28):**

1. Grayscale conversion
2. Gaussian blur (5x5)
3. Adaptive threshold
4. Auto-detect & invert if needed
5. Morphology opening (2x2)
6. Contour detection
7. Crop & center (20x20 → 28x28)
8. Gaussian smooth (3x3)
9. Normalize [0, 1]

**Shapes (64x64):**

1. Grayscale conversion
2. Gaussian blur (5x5)
3. Adaptive threshold
4. Auto-detect & invert if needed
5. Morphology closing (3x3) + opening
6. Contour detection
7. Crop & center (56x56 → 64x64)
8. Normalize [0, 1]

### Model Architecture

**MNIST:**

```
Conv2D(32) → MaxPool → Dropout(0.25)
Conv2D(64) → MaxPool → Dropout(0.25)
Conv2D(64) → Dropout(0.25)
Dense(128) → Dropout(0.5)
Dense(10, softmax)
```

**Shapes:**

```
Conv2D(32) + BatchNorm → MaxPool → Dropout(0.25)
Conv2D(64) + BatchNorm → MaxPool → Dropout(0.25)
Conv2D(128) + BatchNorm → MaxPool → Dropout(0.25)
Dense(128) + BatchNorm → Dropout(0.5)
Dense(3, softmax)
```

---

## 🎓 Workflow Chuẩn

### Lần đầu setup:

```bash
# 1. Cài đặt
pip install -r requirements.txt

# 2. Training
python train_all.py
# Đợi ~30 phút (CPU) hoặc ~10 phút (GPU)

# 3. Verify models
ls models/
# Phải thấy: mnist_model.h5, shapes_model.h5

# 4. Run app
streamlit run app.py

# 5. Test trong browser!
```

### Lần sau:

```bash
# Chỉ cần chạy app (không cần train lại)
streamlit run app.py
```

---

## 📁 Quản lý Files

### Models (models/)

- `mnist_model.h5` - MNIST CNN weights
- `shapes_model.h5` - Shapes CNN weights
- **Dung lượng:** ~5MB total
- **Không commit** lên Git (đã có trong .gitignore)

### Data (data/)

- `shapes/` - Generated shapes (nếu dùng Option 2)
- MNIST tự động download

### Output (example_progress/)

- Training plots
- Sample images
- Demo preprocessing results

---

## 🚨 Lưu ý Quan trọng

1. **Internet cần cho lần đầu:**

   - Download MNIST dataset (~11MB)
   - Install packages (~500MB)

2. **Models file:**

   - Cần train trước khi dùng app
   - Chỉ train 1 lần
   - Backup nếu cần

3. **RAM usage:**

   - Training: 2-3GB
   - App: 500MB-1GB
   - Đóng Chrome nếu thiếu RAM

4. **Thời gian:**
   - Setup: 5-10 phút
   - Training: 30-45 phút (CPU)
   - Sử dụng app: Instant!

---

## ✅ Checklist Hoàn thành

Trước khi nộp bài/demo, kiểm tra:

- [ ] Đã cài đặt requirements.txt
- [ ] Đã train cả 2 models
- [ ] Models đạt >99% accuracy
- [ ] App chạy được không lỗi
- [ ] Test với ít nhất 5 ảnh khác nhau
- [ ] Có ảnh screenshots kết quả
- [ ] Có training history plots
- [ ] Đọc và hiểu code trong `src/preprocessing.py`

---

## 🎉 Hoàn thành!

Sau khi làm theo hướng dẫn này, bạn đã có:

- ✅ 2 CNN models hoạt động tốt
- ✅ Web app để demo
- ✅ Hiểu rõ preprocessing pipeline
- ✅ Sẵn sàng nộp bài/demo

**Chúc bạn thành công! 🎓**

---

## 📞 Cần thêm trợ giúp?

1. Đọc lại phần Troubleshooting
2. Kiểm tra logs/errors cụ thể
3. Xem code comments trong source
4. Google error message

---

_Last updated: 2025-11-14_  
_Version: 2.0_
