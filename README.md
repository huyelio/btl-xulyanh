## Giới thiệu dự án

Dự án là một ứng dụng web dùng **CNN** để nhận dạng:

- **Chữ số viết tay (MNIST)**
- **Hình học cơ bản (shapes: circle, triangle, star, …)**
- **Chữ số Trung Quốc (Chinese MNIST)**
- **Chữ cái in hoa A–Z**

Giao diện được xây dựng bằng **Streamlit**, cho phép:

- Tải nhiều ảnh từ máy tính và nhận dạng theo lô
- Vẽ trực tiếp trên **canvas** rồi nhận dạng
- Xem **từng bước tiền xử lý ảnh** (nếu bật tùy chọn)

Các mô hình CNN đã huấn luyện sẵn được lưu trong thư mục `models/`, bạn có thể **train lại** trên Google Colab bằng các notebook trong `src/`.

---

## Cấu trúc thư mục chính

- **`app.py`**: File chính chạy ứng dụng web Streamlit.
- **`requirements.txt`**: Danh sách thư viện Python cần cài đặt.
- **`src/`**:
  - `preprocessing.py`: Hàm tiền xử lý ảnh cho từng loại bài toán (MNIST, shapes, Chinese, alphabet).
  - `Train_MNIST.ipynb`: Notebook train mô hình chữ số (MNIST).
  - `train_shapes.ipynb`: Notebook train mô hình nhận dạng hình học.
  - `Train_Chinese_MNIST.ipynb`: Notebook train mô hình chữ số Trung Quốc.
  - `train_alphabet.ipynb`: Notebook train mô hình nhận dạng chữ cái A–Z.
- **`models/`**:
  - `mnist_model_augmented.h5` / `mnist_model.h5`
  - `shapes_model_v3_final.h5` (và các phiên bản shapes khác)
  - `chinese_model.h5`
  - `alphabet_model.h5`
- **`shapes_dataset_v3/`**: Dataset hình học đã chia `train/` và `test/` theo từng lớp (arrow, circle, triangle, …).
- **`example_progress/`**: Lưu hình ảnh minh họa các bước tiền xử lý (khi bật chế độ hiển thị pipeline).
- **`test_img/`**: Một số ảnh test nhanh cho từng bài toán.

---

## Yêu cầu hệ thống

- **Python**: khuyến nghị **Python 3.12**
- Đã cài **pip** và (khuyến khích) sử dụng **virtual environment**
- Máy có hỗ trợ TensorFlow CPU (GPU không bắt buộc, nhưng sẽ train nhanh hơn nếu có)

---

## Cài đặt môi trường (local)

1. **(Tùy chọn nhưng khuyến nghị) Tạo virtualenv**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. **Cài đặt các thư viện cần thiết**
   Trong thư mục gốc của dự án:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Chạy ứng dụng web Streamlit

Trong thư mục gốc của dự án, sau khi đã cài xong môi trường:

```bash
streamlit run app.py
```

Ứng dụng sẽ mở trên trình duyệt (thường là `http://localhost:8501`).

Tại giao diện chính, bạn có thể:

- Chọn **chế độ**: MNIST, Alphabet, Shapes, Chinese.
- **Upload ảnh** (1 hoặc nhiều file) để nhận dạng theo lô.
- **Vẽ trực tiếp trên canvas** rồi bấm nút “Nhận dạng”.
- Bật **“Hiển thị từng bước xử lý”** để xem pipeline tiền xử lý ảnh.

---

## Huấn luyện lại mô hình trên Google Colab

Các file train nằm trong thư mục `src/` đã được thiết kế để bạn **mở trực tiếp trên Google Colab**:

- `src/Train_MNIST.ipynb`
- `src/train_shapes.ipynb`
- `src/Train_Chinese_MNIST.ipynb`
- `src/train_alphabet.ipynb`

Bạn có thể dùng chúng để train lại mô hình, sau đó copy file `.h5` mới về thư mục `models/` để ứng dụng sử dụng.
