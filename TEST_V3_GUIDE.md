# 🧪 Hướng dẫn Test Preprocessing V3

## ✅ Setup đã hoàn thành!

App đang chạy tại: **http://localhost:8501** 🚀

---

## 🎯 Cách Test

### 1. Mở App
```
Truy cập: http://localhost:8501
```

### 2. Test với ảnh có sẵn

#### Option A: Dùng ảnh trong test_img/
```
test_img/
├── download.png
├── download (1).png
├── download (2).png
├── images.png
└── Screenshot *.png  (nhiều ảnh)
```

#### Option B: Dùng ảnh đã generate
```
example_progress/
├── sample0_original.png       (MNIST digit 5)
├── sample1_original.png       (MNIST digit 5)
├── sample2_original.png       (MNIST digit 1)
├── shape_circle_original.png
├── shape_rectangle_original.png
└── shape_triangle_original.png
```

### 3. Test Workflow

**Bước 1:** Upload ảnh
- Click "Browse files" hoặc drag & drop
- Chọn ảnh từ `test_img/` hoặc `example_progress/`

**Bước 2:** Chọn mode
- "Chữ số (MNIST)" - cho ảnh chữ số
- "Hình học (Shapes)" - cho ảnh hình tròn/chữ nhật/tam giác

**Bước 3:** (Optional) Check "Hiển thị từng bước xử lý"
- ✅ Check: Xem grid 12 bước xử lý + ảnh được lưu
- ⬜ Uncheck: Chỉ xem kết quả cuối

**Bước 4:** Click "🔍 Nhận dạng"

---

## 📊 Kết quả Mong đợi

### Với ảnh **NỀN ĐEN, CHỮ TRẮNG:**
✅ Ảnh sau xử lý: Rõ ràng, chữ trắng trên nền đen  
✅ Độ tin cậy: >90%  
✅ Top 3: Đúng thứ tự

### Với ảnh **NỀN TRẮNG, CHỮ ĐEN:**
✅ Ảnh sau xử lý: Được invert thành trắng trên đen  
✅ Độ tin cậy: >90%  
✅ Top 3: Đúng thứ tự

### Với checkbox "Hiển thị từng bước":
✅ Grid hiển thị 12 bước (4 hàng x 3 cột)  
✅ Message: "Đã lưu 12 ảnh vào: example_progress/progress_images/"  
✅ Có thể vào thư mục check từng ảnh

---

## 🐛 Troubleshooting

### Lỗi: "Model chưa được tải"
**Nguyên nhân:** Thiếu file models/mnist_model.h5 hoặc models/shapes_model.h5

**Giải pháp:**
```bash
# Check files
ls models/

# Nếu thiếu, train lại:
python train_all.py
```

### Lỗi: Ảnh bị mờ/sai
**Nguyên nhân:** App đang dùng code cũ chưa reload

**Giải pháp:**
```bash
# Restart app
taskkill /F /IM streamlit.exe
venv\Scripts\activate
streamlit run app.py
```

### Lỗi: Không hiển thị pipeline
**Nguyên nhân:** Chưa check "Hiển thị từng bước xử lý"

**Giải pháp:** Check vào checkbox trước khi nhấn "Nhận dạng"

---

## 📸 Check Ảnh đã lưu

### Xem ảnh progress:
```
example_progress/progress_images/
├── mnist_sample0/
│   ├── step01_grayscale.png
│   ├── step02_clahe.png
│   ├── step03_gaussian_blur.png
│   ├── step04_otsu_threshold.png
│   ├── step05_inverted.png
│   ├── step06_morphology_open.png
│   ├── step07_morphology_close.png
│   ├── step08_contour.png
│   ├── step09_cropped.png
│   ├── step10_resized.png
│   ├── step11_centered.png
│   └── step12_final_smoothed.png
├── mnist_sample1/ (12 files)
├── mnist_sample2/ (12 files)
├── shape_circle/ (12 files)
├── shape_rectangle/ (12 files)
└── shape_triangle/ (12 files)
```

### Xem ảnh comparison:
```
example_progress/
├── mnist_preprocessing_comparison.png  ⭐
├── shapes_preprocessing_comparison.png ⭐
├── preprocessing_flowchart_v3.png      ⭐
└── mnist_before_after.png              ⭐
```

---

## ✨ V3 Features để test

### 1. Corner-based Detection
**Test case:** Upload ảnh nền trắng  
**Expected:** Tự động invert thành nền đen  
**Check:** Xem `step05_inverted.png` có khác `step04_otsu_threshold.png`

### 2. CLAHE Enhancement
**Test case:** Upload ảnh contrast thấp  
**Expected:** CLAHE tăng cường tốt  
**Check:** So sánh `step01_grayscale.png` vs `step02_clahe.png`

### 3. Morphology Cleaning
**Test case:** Upload ảnh có noise  
**Expected:** Opening/Closing loại nhiễu  
**Check:** So sánh `step05_inverted.png` vs `step07_morphology_close.png`

### 4. Smart Centering
**Test case:** Upload ảnh object không ở giữa  
**Expected:** Được crop và center  
**Check:** Xem `step08_contour.png` (bounding box) → `step11_centered.png` (centered)

### 5. Consistent Output
**Test case:** Upload nhiều ảnh khác nhau (nền đen/trắng/xám)  
**Expected:** Tất cả output đều WHITE on BLACK  
**Check:** `step12_final_smoothed.png` luôn là chữ trắng trên nền đen

---

## 📝 Test Checklist

- [ ] App chạy được tại localhost:8501
- [ ] Upload ảnh nền đen → nhận dạng đúng
- [ ] Upload ảnh nền trắng → nhận dạng đúng  
- [ ] Check "Hiển thị từng bước" → thấy 12 bước
- [ ] Ảnh được lưu vào progress_images/
- [ ] Top 3 predictions hiển thị đúng
- [ ] Confidence score >90%
- [ ] Ảnh sau xử lý rõ ràng (không bị mờ)
- [ ] Test với ít nhất 5 ảnh khác nhau
- [ ] Check các file comparison trong example_progress/

---

## 🎉 Khi test xong

### Files để đưa vào báo cáo:
```
✅ mnist_preprocessing_comparison.png    - Pipeline MNIST
✅ shapes_preprocessing_comparison.png   - Pipeline Shapes  
✅ preprocessing_flowchart_v3.png        - Sơ đồ V3
✅ mnist_before_after.png                - So sánh trước/sau
✅ progress_images/* folders             - Chi tiết từng bước

Optional:
📸 Screenshot app đang chạy
📸 Screenshot kết quả nhận dạng
📸 Screenshot grid các bước xử lý
```

### Demo points:
1. ✅ Xử lý mọi loại ảnh (nền đen/trắng)
2. ✅ Pipeline đầy đủ 12 bước
3. ✅ Áp dụng đủ kỹ thuật: CLAHE, Otsu, Morphology, Contour...
4. ✅ Output consistent: WHITE on BLACK
5. ✅ Lưu và visualize từng bước

---

## 💡 Tips

### Để ảnh đẹp cho báo cáo:
1. Test với nhiều loại ảnh: rõ, mờ, noise, nền khác nhau
2. Chụp màn hình grid 12 bước
3. Dùng comparison figures đã generate
4. Highlight các bước quan trọng (step 5 invert, step 11 center)

### Để hiểu pipeline:
1. Upload 1 ảnh
2. Check "Hiển thị từng bước"  
3. Nhận dạng
4. Xem từng bước trong grid
5. Vào progress_images/ xem ảnh HD

---

**App đang chạy! Bắt đầu test ngay! 🚀**

_http://localhost:8501_

