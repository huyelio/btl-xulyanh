# Hướng Dẫn Tích Hợp Model Chinese MNIST

## 📋 Tổng Quan

Hướng dẫn này giúp bạn huấn luyện model nhận diện chữ số Trung Quốc trên Google Colab và tích hợp vào ứng dụng.

---

## 🚀 BƯỚC 1: Huấn Luyện Model trên Google Colab

### 1.1. Chuẩn Bị Kaggle API

1. Truy cập: https://www.kaggle.com/settings/account
2. Scroll xuống phần **API**
3. Click **"Create New Token"**
4. File `kaggle.json` sẽ được tải về máy

### 1.2. Tạo Notebook trên Google Colab

1. Truy cập: https://colab.research.google.com/
2. Click **"New Notebook"**
3. Đổi tên notebook: `Train_Chinese_MNIST`

### 1.3. Copy Code vào Colab

1. Mở file `train_chinese_mnist_colab.py` trong project của bạn
2. **Copy toàn bộ nội dung** của file này
3. **Paste** vào một cell trong Colab notebook
4. Chọn **Runtime > Change runtime type > GPU** (T4 hoặc A100)

### 1.4. Chạy Training

1. Click nút **Play** (▶) bên trái cell
2. Khi có thông báo upload file, click **"Choose Files"** và upload `kaggle.json`
3. Đợi quá trình huấn luyện (10-20 phút tùy GPU)
4. File `chinese_model.h5` sẽ **tự động tải về** máy của bạn

### 1.5. Kết Quả Mong Đợi

```
✅ ĐÃ HOÀN THÀNH HUẤN LUYỆN!

📊 Kết quả trên Validation Set:
  - Loss: ~0.1-0.3
  - Accuracy: 85-95%

💾 Kích thước file: ~15-25 MB
```

---

## 📦 BƯỚC 2: Cài Đặt Model vào App

### 2.1. Di Chuyển File Model

1. Tìm file `chinese_model.h5` trong thư mục Downloads
2. Di chuyển vào thư mục project:

```
btl_final/
├── models/
│   ├── mnist_model_augmented.h5
│   ├── shapes_model.h5
│   └── chinese_model.h5  ← Đặt file vào đây
```

### 2.2. Cấu Trúc Project Sau Khi Hoàn Thành

```
btl_final/
├── app.py                          (Đã được cập nhật)
├── src/
│   ├── preprocessing.py            (Đã thêm preprocess_for_chinese)
│   └── ...
├── models/
│   ├── mnist_model_augmented.h5
│   ├── shapes_model.h5
│   └── chinese_model.h5            (Model mới)
├── train_chinese_mnist_colab.py    (Script cho Colab)
└── HUONG_DAN_CHINESE_MNIST.md      (File này)
```

---

## 🎮 BƯỚC 3: Chạy Ứng Dụng

### 3.1. Khởi Động App

Mở terminal và chạy:

```bash
cd D:\School\xuLyAnh\btl_final
streamlit run app.py
```

### 3.2. Sử Dụng

1. Ứng dụng sẽ mở trong browser
2. Chọn **"Chữ số Trung Quốc (Chinese)"** trong dropdown
3. Upload hoặc vẽ một chữ số Trung Quốc
4. Click **"🔍 Nhận dạng"**

### 3.3. Kết Quả Hiển Thị

```
Chữ số Trung Quốc: 三
Độ tin cậy: 95.8%

📊 Top 3 dự đoán:
三 ████████████████ 95.8%
二 ██████           12.3%
五 ███              8.9%
```

---

## 🔢 Mapping Chữ Số Trung Quốc

| Index | Ký Tự | Nghĩa           | Pinyin |
| ----- | ----- | --------------- | ------ |
| 0     | 零    | Zero            | líng   |
| 1     | 一    | One             | yī     |
| 2     | 二    | Two             | èr     |
| 3     | 三    | Three           | sān    |
| 4     | 四    | Four            | sì     |
| 5     | 五    | Five            | wǔ     |
| 6     | 六    | Six             | liù    |
| 7     | 七    | Seven           | qī     |
| 8     | 八    | Eight           | bā     |
| 9     | 九    | Nine            | jiǔ    |
| 10    | 十    | Ten             | shí    |
| 11    | 百    | Hundred         | bǎi    |
| 12    | 千    | Thousand        | qiān   |
| 13    | 万    | Ten thousand    | wàn    |
| 14    | 亿    | Hundred million | yì     |

---

## 🛠️ Troubleshooting

### Vấn Đề 1: Model Không Load

**Lỗi:**

```
❌ Model chinese chưa được tải! Vui lòng đảm bảo file models/chinese_model.h5 tồn tại.
```

**Giải pháp:**

- Kiểm tra file `chinese_model.h5` có tồn tại trong thư mục `models/`
- Đảm bảo tên file chính xác (không có khoảng trắng)
- Khởi động lại ứng dụng: `Ctrl+C` rồi chạy lại `streamlit run app.py`

### Vấn Đề 2: Kaggle API Lỗi trên Colab

**Lỗi:**

```
Unauthorized: invalid credentials
```

**Giải pháp:**

- Tải lại `kaggle.json` từ Kaggle
- Đảm bảo upload đúng file `kaggle.json` (không phải file khác)
- Kiểm tra file không bị corrupt

### Vấn Đề 3: Colab Hết RAM/Time

**Giải pháp:**

- Chọn GPU runtime: Runtime > Change runtime type > T4 GPU
- Giảm batch_size trong code: `batch_size=32` thay vì 64
- Giảm số epochs: `epochs=30` thay vì 50

### Vấn Đề 4: Accuracy Thấp (<80%)

**Giải pháp:**

1. Tăng số epochs: `epochs=100`
2. Thêm data augmentation:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
```

### Vấn Đề 5: File Model Quá Lớn

**Giải pháp:**

- Giảm số filters trong Conv2D layers
- Giảm số Dense units
- Sử dụng model compression sau khi train

---

## 📊 Thông Số Model

### Architecture

```
Layer (type)                Output Shape              Param #
=================================================================
conv2d (Conv2D)            (None, 62, 62, 32)        320
max_pooling2d (MaxPooling2D)(None, 31, 31, 32)       0
batch_normalization        (None, 31, 31, 32)        128

conv2d_1 (Conv2D)          (None, 29, 29, 64)        18496
max_pooling2d_1            (None, 14, 14, 64)        0
batch_normalization_1      (None, 14, 14, 64)        256

conv2d_2 (Conv2D)          (None, 12, 12, 128)       73856
max_pooling2d_2            (None, 6, 6, 128)         0
batch_normalization_2      (None, 6, 6, 128)         512

conv2d_3 (Conv2D)          (None, 4, 4, 256)         295168
max_pooling2d_3            (None, 2, 2, 256)         0
batch_normalization_3      (None, 2, 2, 256)         1024

flatten (Flatten)          (None, 1024)              0
dropout (Dropout)          (None, 1024)              0
dense (Dense)              (None, 512)               524800
dropout_1 (Dropout)        (None, 512)               0
dense_1 (Dense)            (None, 256)               131328
dropout_2 (Dropout)        (None, 256)               0
dense_2 (Dense)            (None, 15)                3855
=================================================================
Total params: 1,049,743
Trainable params: 1,048,783
Non-trainable params: 960
```

### Preprocessing Pipeline

1. **Grayscale** → Chuyển sang ảnh xám
2. **Gaussian Blur** → Giảm nhiễu (kernel 5x5)
3. **Otsu Threshold** → Tách foreground/background
4. **Invert** → Đảm bảo nền đen, chữ trắng
5. **Find Contours** → Tìm bounding box
6. **Crop + Padding** → Cắt với padding 15%
7. **Resize** → Giữ tỷ lệ, fit vào 56x56
8. **Center** → Đặt vào canvas 64x64
9. **Smooth** → Làm mượt cuối cùng (kernel 3x3)
10. **Normalize** → Chia cho 255.0

---

## 📚 Dataset Information

- **Tên:** Chinese MNIST
- **Nguồn:** Kaggle (gpreda/chinese-mnist)
- **Kích thước:** ~15,000 ảnh
- **Ảnh:** 64x64 grayscale
- **Số lớp:** 15 (零-亿)
- **Format:** CSV (flattened pixels)

---

## ✅ Checklist Hoàn Thành

- [ ] Tải `kaggle.json` từ Kaggle
- [ ] Tạo notebook trên Google Colab
- [ ] Chọn GPU runtime
- [ ] Copy code từ `train_chinese_mnist_colab.py`
- [ ] Chạy training và đợi hoàn thành
- [ ] Tải file `chinese_model.h5` về máy
- [ ] Di chuyển file vào `models/`
- [ ] Khởi động app và test
- [ ] Kiểm tra 3 modes hoạt động đúng

---

## 🎯 Tips & Best Practices

1. **Training:**

   - Luôn dùng GPU trên Colab (nhanh hơn 10-20x)
   - Theo dõi validation accuracy để tránh overfitting
   - Early stopping sẽ tự động dừng khi model không cải thiện

2. **Testing:**

   - Test với nhiều loại ảnh: viết tay, in ấn, background khác nhau
   - Sử dụng checkbox "Hiển thị từng bước xử lý" để debug

3. **Optimization:**
   - Nếu model chậm, giảm kích thước layers
   - Nếu accuracy thấp, tăng epochs hoặc thêm augmentation

---

## 🔗 Resources

- [Chinese MNIST Dataset](https://www.kaggle.com/datasets/gpreda/chinese-mnist)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Google Colab](https://colab.research.google.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📝 Ghi Chú

- Model này được thiết kế tương thích hoàn toàn với codebase hiện có
- Preprocessing pipeline tương tự MNIST nhưng output 64x64
- Chinese labels được hardcode trong app.py (có thể customize)

---

**Chúc bạn thành công! 🎉**

Nếu có vấn đề, hãy kiểm tra lại từng bước trong hướng dẫn này.
