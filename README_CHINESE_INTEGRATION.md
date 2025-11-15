# 🇨🇳 Tích Hợp Model Chữ Số Trung Quốc - Quick Start

## 📦 Files Đã Được Tạo/Cập Nhật

### ✅ Files Mới
1. **`train_chinese_mnist_colab.py`** - Script để chạy trên Google Colab
2. **`HUONG_DAN_CHINESE_MNIST.md`** - Hướng dẫn chi tiết đầy đủ

### ✅ Files Đã Cập Nhật
1. **`app.py`** - Thêm mode Chinese, load model, hiển thị kết quả
2. **`src/preprocessing.py`** - Thêm hàm `preprocess_for_chinese()`

---

## 🚀 Quick Start (3 Bước)

### BƯỚC 1: Huấn Luyện Model trên Colab (10-20 phút)

1. Truy cập: https://colab.research.google.com/
2. Tạo notebook mới, chọn **GPU runtime**
3. Copy toàn bộ code từ `train_chinese_mnist_colab.py` vào 1 cell
4. Chạy cell và làm theo hướng dẫn (upload kaggle.json)
5. Đợi training xong → file `chinese_model.h5` tự động tải về

### BƯỚC 2: Cài Đặt Model

Di chuyển file `chinese_model.h5` vào:
```
btl_final/models/chinese_model.h5
```

### BƯỚC 3: Chạy App

```bash
streamlit run app.py
```

Chọn **"Chữ số Trung Quốc (Chinese)"** và test!

---

## 📋 Chi Tiết Thay Đổi

### 1. app.py

**Thêm:**
- Import `preprocess_for_chinese`
- Load model `chinese_model.h5`
- Radio button có thêm option "Chữ số Trung Quốc (Chinese)"
- Logic xử lý cho Chinese mode
- Danh sách `CHINESE_LABELS` (15 ký tự)
- Hiển thị kết quả Chinese với top 3 predictions

**Dòng code quan trọng:**
```python
# Line 17: Import preprocessing function
from preprocessing import preprocess_for_mnist, preprocess_for_shapes, preprocess_for_chinese

# Line 56-63: Load chinese model
chinese_path = 'models/chinese_model.h5'
if os.path.exists(chinese_path):
    models['chinese'] = keras.models.load_model(chinese_path)

# Line 76: Chinese labels
CHINESE_LABELS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']

# Line 84: Radio button với option mới
mode = st.radio("Chế độ:", ["Chữ số (MNIST)", "Hình học (Shapes)", "Chữ số Trung Quốc (Chinese)"])

# Line 144-157: Logic xử lý Chinese
else:  # Chinese Numerals
    processed, display_img, progress = preprocess_for_chinese(
        image,
        save_steps=show_pipeline,
        output_dir="example_progress/progress_images"
    )
    prediction = models['chinese'].predict(processed, verbose=0)
    result = np.argmax(prediction)
    confidence = prediction[0][result]
    result_text = f"Chữ số Trung Quốc: **{CHINESE_LABELS[result]}**"
```

### 2. src/preprocessing.py

**Thêm:**
- Hàm `preprocess_for_chinese()` (dòng 499-617)
- Pipeline 10 bước cho ảnh 64x64:
  1. Grayscale
  2. Gaussian Blur (5x5)
  3. Otsu Threshold
  4. Invert (nền đen, chữ trắng)
  5. Find Contours + Bounding Box
  6. Crop với padding 15%
  7. Resize giữ tỷ lệ (fit vào 56x56)
  8. Center vào canvas 64x64
  9. Smooth (Gaussian 3x3)
  10. Normalize [0, 1]

**Output:** `(1, 64, 64, 1)` - Tương tự MNIST nhưng kích thước khác

---

## 🔢 Chinese Labels Mapping

| Index | Ký Tự | Nghĩa Tiếng Việt | Pinyin |
|-------|-------|------------------|---------|
| 0 | 零 | Không | líng |
| 1 | 一 | Một | yī |
| 2 | 二 | Hai | èr |
| 3 | 三 | Ba | sān |
| 4 | 四 | Bốn | sì |
| 5 | 五 | Năm | wǔ |
| 6 | 六 | Sáu | liù |
| 7 | 七 | Bảy | qī |
| 8 | 八 | Tám | bā |
| 9 | 九 | Chín | jiǔ |
| 10 | 十 | Mười | shí |
| 11 | 百 | Trăm | bǎi |
| 12 | 千 | Nghìn | qiān |
| 13 | 万 | Vạn (10,000) | wàn |
| 14 | 亿 | Ức (100,000,000) | yì |

---

## 📊 Model Architecture

```
Input: 64x64x1 grayscale image
↓
4x Conv2D + MaxPooling + BatchNorm blocks (32→64→128→256 filters)
↓
Flatten
↓
Dense(512) → Dropout(0.5)
↓
Dense(256) → Dropout(0.3)
↓
Dense(15, softmax) → 15 classes output
```

**Total params:** ~1M  
**Expected accuracy:** 85-95%  
**Training time:** 10-20 minutes on Colab T4 GPU

---

## 🎯 Testing

### Test Cases

1. **Chữ viết tay:**
   - Vẽ chữ 三 (ba) → Nên nhận dạng đúng
   
2. **Ảnh in ấn:**
   - Upload ảnh chữ 八 (tám) in rõ ràng → Độ tin cậy cao (>90%)

3. **Background khác nhau:**
   - Nền trắng: ✅ Auto invert
   - Nền đen: ✅ Giữ nguyên
   - Nền màu: ✅ Chuyển grayscale rồi xử lý

4. **Pipeline visualization:**
   - Bật checkbox "📊 Hiển thị từng bước xử lý"
   - Xem 10 bước preprocessing

---

## ⚠️ Lưu Ý

1. **File model phải có tên chính xác:** `chinese_model.h5`
2. **Đặt đúng vị trí:** `models/chinese_model.h5`
3. **Kaggle API:** Cần có `kaggle.json` để tải dataset
4. **GPU trên Colab:** Bắt buộc để train nhanh (10-20 phút vs 2-3 giờ)
5. **Restart app:** Nếu thêm model mới, cần restart Streamlit app

---

## 🐛 Common Issues

### Issue: "Model chinese chưa được tải"
**Fix:** Kiểm tra file `models/chinese_model.h5` tồn tại và restart app

### Issue: Kaggle Unauthorized
**Fix:** Tải lại `kaggle.json` từ Kaggle và upload lại

### Issue: Colab out of memory
**Fix:** Chọn T4 GPU runtime, giảm batch_size xuống 32

---

## 📚 Đọc Thêm

Xem **`HUONG_DAN_CHINESE_MNIST.md`** để biết:
- Hướng dẫn từng bước chi tiết
- Troubleshooting đầy đủ
- Architecture details
- Dataset information
- Optimization tips

---

**Hoàn thành! 🎉** Giờ bạn có 3 models trong 1 app:
- 🔢 MNIST (0-9, 28x28)
- 📐 Shapes (3 hình học, 64x64)
- 🇨🇳 Chinese (15 ký tự, 64x64)

