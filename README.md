## Dự án Nhận dạng chữ số viết tay và hình dạng đơn giản bằng mạng neural CNN

Dự án là một ứng dụng web dùng **CNN** có khả năng nhận dạng chữ số viết tay (0–9), các chữ cái và các hình dạng cơ bản(tam giác, vuông, tròn, chữ nhật…) từ ảnh đầu vào.
Áp dụng kết hợp kỹ thuật xử lý ảnh và mạng nơ-ron tích chập (CNN).

Xây dựng 4 mô hình CNN độc lập cho 4 nhiệm vụ:

- **Chữ số viết tay (MNIST)**
- **Hình học cơ bản (shapes: circle, triangle, star, …)**
- **Chữ số Trung Quốc (Chinese MNIST)**
- **Chữ cái in hoa A–Z**

# Danh Sách Thành Viên Nhóm

| STT |   Họ và Tên    |    MSSV    |
| :-: | :------------: | :--------: |
|  1  | Trần Quang Huy | B22DCCN397 |
|  2  |  Đỗ Đức Cảnh   | B22DCCN086 |
|  3  | Trần Quang Huy | B22DCCN398 |

Giao diện được xây dựng bằng **Streamlit**, cho phép:

- Tải nhiều ảnh từ máy tính và nhận dạng theo lô
- Vẽ trực tiếp trên **canvas** rồi nhận dạng
- Xem **từng bước tiền xử lý ảnh** (nếu bật tùy chọn)

Các mô hình CNN đã huấn luyện sẵn được lưu trong thư mục `models/`, bạn có thể **train lại** trên Google Colab bằng các notebook trong `src/`.

---

## Cấu trúc thư mục chính

```txt
project_root/
│
├── code/
│   ├── app.py                    # File chính chạy ứng dụng Streamlit
│   ├── requirements.txt          # Danh sách thư viện cần cài đặt
│   │
│   ├── src/                      # Mã nguồn & notebook train model
│   │   ├── preprocessing.py
│   │   ├── Train_MNIST.ipynb
│   │   ├── train_shapes.ipynb
│   │   ├── Train_Chinese_MNIST.ipynb
│   │   └── train_alphabet.ipynb
│   │
│   ├── models/                   # Các mô hình đã train
│   │   ├── mnist_model.h5 / mnist_model_augmented.h5
│   │   ├── shapes_model_v3_final.h5
│   │   ├── chinese_model.h5
│   │   └── alphabet_model.h5
│   │
│   ├── shapes_dataset_v3/        # Dataset hình học (train/test)
│   │
│   ├── example_progress/         # Lưu ảnh minh họa pipeline
│   │
│   └── test_img/                 # Ảnh test nhanh
│
└── báo cáo/
    ├── Báo cáo XLA Nhóm 15.pdf   # Báo cáo PDF
    └── Slide Xử lý ảnh Nhóm 15.pdf # Slide thuyết trình
```

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
   Trong thư mục code/:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Chạy ứng dụng web Streamlit

Trong thư mục code/, sau khi đã cài xong môi trường:

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

Các notebook trong `code/src/` đã được thiết kế để **chạy trực tiếp trên Google Colab**:

- `src/Train_MNIST.ipynb`
- `src/train_shapes.ipynb`
- `src/Train_Chinese_MNIST.ipynb`
- `src/train_alphabet.ipynb`

CSau khi train xong, tải file `.h5` mới về thư mục `code/models/` để cập nhật mô hình dùng bởi ứng dụng.
