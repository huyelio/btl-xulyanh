"""
Giao diện web Streamlit cho dự án nhận dạng chữ số và hình học - V3 ULTRA ROBUST
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
import sys

from streamlit_drawable_canvas import st_canvas

# Thêm thư mục src vào path
sys.path.append('src')

from preprocessing import preprocess_for_mnist, preprocess_for_shapes, preprocess_for_chinese

# Cấu hình trang
st.set_page_config(
    page_title="Nhận dạng CNN",
    page_icon="🔍",
    layout="wide"
)

# CSS tùy chỉnh
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700&display=swap');

    .main {background-color: #ffffff;}
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .result-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load các models đã huấn luyện"""
    models = {}
    
    # mnist_path = 'models/mnist_model.h5'
    mnist_path = 'models/mnist_model_augmented.h5'
    shapes_path = 'models/shapes_model_v2.h5'
    chinese_path = 'models/chinese_model.h5'
    
    if os.path.exists(mnist_path):
        models['mnist'] = keras.models.load_model(mnist_path)
    if os.path.exists(shapes_path):
        models['shapes'] = keras.models.load_model(shapes_path)
    if os.path.exists(chinese_path):
        models['chinese'] = keras.models.load_model(chinese_path)
    
    return models


def main():
    """Hàm chính của ứng dụng"""
    
    st.title("Nhận dạng Chữ số, Hình học và Chữ số Trung Quốc")
    st.markdown("*Xử lý hoàn hảo mọi loại ảnh - nền trắng, nền đen, màu sắc bất kỳ*")
    st.markdown("---")
    
    # Chinese labels mapping
    CHINESE_LABELS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']
    CHINESE_LABELS_VN = ['số 0', 'số 1', 'số 2', 'số 3', 'số 4', 'số 5', 'số 6', 'số 7', 'số 8', 'số 9', 
                         'số 10', 'trăm', 'nghìn', 'vạn (10,000)', 'ức (100 triệu)']
    
    models = load_models()

    # Biến dùng chung cho cả upload và canvas
    uploaded_files = []
    canvas_image = None

    # Hàm phụ xử lý 1 ảnh (dùng chung cho canvas và từng file trong batch)
    def run_inference_for_image(input_image, mode, show_pipeline, output_dir):
        if mode == "Chữ số (MNIST)":
            processed, display_img, progress = preprocess_for_mnist(
                input_image,
                save_steps=show_pipeline,
                output_dir=output_dir,
            )
            prediction = models["mnist"].predict(processed, verbose=0)
            result = np.argmax(prediction)
            confidence = prediction[0][result]
            result_text = f"Chữ số: **{result}**"

            top3_idx = np.argsort(prediction[0])[-3:][::-1]
            top3_probs = prediction[0][top3_idx]
            shapes_labels = None

        elif mode == "Hình học (Shapes)":
            processed, display_img, progress = preprocess_for_shapes(
                input_image,
                save_steps=show_pipeline,
                output_dir=output_dir,
            )
            prediction = models["shapes"].predict(processed, verbose=0)
            result = np.argmax(prediction)
            confidence = prediction[0][result]
            shapes_labels = [
                "Hình tròn",
                "Hình chữ nhật",
                "Hình tam giác",
                "Hình ngũ giác",
                "Hình lục giác",
                "Hình bát giác",
                "Hình ngôi sao",
                "Hình thoi",
                "Hình chữ thập",
                "Mũi tên",
            ]
            result_text = f"Hình: **{shapes_labels[result]}**"

            top3_idx = np.argsort(prediction[0])[-3:][::-1]
            top3_probs = prediction[0][top3_idx]

        else:  # Chinese Numerals
            processed, display_img, progress = preprocess_for_chinese(
                input_image,
                save_steps=show_pipeline,
                output_dir=output_dir,
            )
            prediction = models["chinese"].predict(processed, verbose=0)
            result = np.argmax(prediction)
            confidence = prediction[0][result]
            result_text = (
                f"Chữ số Trung Quốc: **{CHINESE_LABELS[result]}** - {CHINESE_LABELS_VN[result]}"
            )

            top3_idx = np.argsort(prediction[0])[-3:][::-1]
            top3_probs = prediction[0][top3_idx]
            shapes_labels = None

        return result_text, confidence, top3_idx, top3_probs, display_img, progress, shapes_labels
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Cài đặt")
        mode = st.radio("Chế độ:", ["Chữ số (MNIST)", "Hình học (Shapes)", "Chữ số Trung Quốc (Chinese)"])
        
        # Thêm option hiển thị pipeline
        show_pipeline = st.checkbox("Hiển thị từng bước xử lý", value=False)
        
        # --- Khu vực upload ảnh ---
        st.subheader("Tải ảnh")
        uploaded_files = st.file_uploader(
            "Chọn một hoặc nhiều ảnh",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.write(f"Đã chọn {len(uploaded_files)} ảnh.")
            # Hiển thị nhanh ảnh đầu tiên để preview
            first_file = uploaded_files[0]
            first_file_bytes = np.asarray(bytearray(first_file.read()), dtype=np.uint8)
            first_image = cv2.imdecode(first_file_bytes, cv2.IMREAD_COLOR)
            st.image(
                first_image,
                channels="BGR",
                use_container_width=True,
                caption=f"Ảnh mẫu: {first_file.name}",
            )

        # --- Khu vực vẽ trực tiếp trên canvas ---
        st.subheader("Vẽ trực tiếp trên canvas")

        # Điều khiển nét vẽ
        col_canvas_ctrl1, col_canvas_ctrl2 = st.columns(2)
        with col_canvas_ctrl1:
            stroke_width = st.slider("Độ dày nét", 3, 25, 10)
        with col_canvas_ctrl2:
            stroke_color = st.color_picker("Màu bút", "#000000")

        # Nút xóa canvas sử dụng session_state để đổi key
        if "canvas_key" not in st.session_state:
            st.session_state.canvas_key = 0
        if st.button("Xóa canvas"):
            st.session_state.canvas_key += 1

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # không tô nền, chỉ nét vẽ
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#FFFFFF",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )

        if canvas_result is not None and canvas_result.image_data is not None:
            canvas_rgba = canvas_result.image_data.astype("uint8")

            # Kiểm tra canvas có nội dung hay không (không chỉ toàn nền trắng)
            # Nếu tất cả kênh RGB đều xấp xỉ 255 thì coi như rỗng
            rgb = canvas_rgba[..., :3]
            if not np.all(rgb == 255):
                # Chuyển RGBA (hoặc RGB) về BGR cho OpenCV
                if canvas_rgba.shape[2] == 4:
                    canvas_rgb = cv2.cvtColor(canvas_rgba, cv2.COLOR_RGBA2RGB)
                else:
                    canvas_rgb = canvas_rgba[..., :3]
                canvas_image = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
                st.image(canvas_image, channels="BGR", use_container_width=True, caption="Ảnh từ canvas")
    
    with col2:
        st.subheader("Kết quả")
        
        recognize_clicked = st.button("Nhận dạng")

        if recognize_clicked:
            # Xác định model tương ứng với chế độ
            if mode == "Chữ số (MNIST)":
                model_key = "mnist"
            elif mode == "Hình học (Shapes)":
                model_key = "shapes"
            else:
                model_key = "chinese"

            if model_key not in models:
                st.error(
                    f"Model {model_key} chưa được tải! Vui lòng đảm bảo file models/{model_key}_model.h5 tồn tại."
                )
                return

            # Nếu có ảnh từ canvas -> ưu tiên xử lý đơn lẻ
            if canvas_image is not None:
                with st.spinner("Đang xử lý ảnh từ canvas..."):
                    try:
                        (
                            result_text,
                            confidence,
                            top3_idx,
                            top3_probs,
                            display_img,
                            progress,
                            shapes_labels,
                        ) = run_inference_for_image(
                            canvas_image,
                            mode,
                            show_pipeline,
                            "example_progress/progress_images",
                        )

                        st.markdown(
                            f'<div class="result-box">'
                            f'<h1 style="text-align:center; color:#2E7D32">{result_text}</h1>'
                            f'<p style="text-align:center; font-size:24px; color:#1976D2">Độ tin cậy: {confidence*100:.1f}%</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.image(
                                display_img,
                                use_container_width=True,
                                caption="Ảnh sau xử lý cuối cùng",
                                clamp=True,
                            )

                        with col_b:
                            st.markdown("Top 3 dự đoán:")
                            for idx_pred, prob in zip(top3_idx, top3_probs):
                                if mode == "Chữ số (MNIST)":
                                    label = str(idx_pred)
                                elif mode == "Hình học (Shapes)" and shapes_labels is not None:
                                    label = shapes_labels[idx_pred]
                                else:
                                    label = f"{CHINESE_LABELS[idx_pred]} ({CHINESE_LABELS_VN[idx_pred]})"

                                st.write(label)
                                st.progress(float(prob))
                                st.write(f"{prob*100:.1f}%")

                        if show_pipeline and progress:
                            st.markdown("---")
                            st.subheader("Các bước xử lý ảnh")

                            step_keys = sorted(
                                [k for k in progress.keys() if k.startswith("step")]
                            )
                            num_cols = 3
                            for i in range(0, len(step_keys), num_cols):
                                cols = st.columns(num_cols)
                                for j in range(num_cols):
                                    if i + j < len(step_keys):
                                        key = step_keys[i + j]
                                        step_img = progress[key]
                                        step_name = (
                                            key.replace("step", "Bước ")
                                            .replace("_", " ")
                                            .title()
                                        )
                                        with cols[j]:
                                            st.image(
                                                step_img,
                                                caption=step_name,
                                                use_container_width=True,
                                                clamp=True,
                                            )
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
                        import traceback
                        st.code(traceback.format_exc())

                return

            # Không có canvas -> xử lý lô ảnh upload
            if not uploaded_files:
                st.warning(
                    "Vui lòng vẽ trên canvas hoặc tải lên ít nhất một ảnh trước khi nhận dạng."
                )
                return

            with st.spinner("Đang xử lý lô ảnh..."):
                for idx_file, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Do file đã được read() khi preview, cần seek(0) để đọc lại
                        uploaded_file.seek(0)
                        file_bytes = np.asarray(
                            bytearray(uploaded_file.read()), dtype=np.uint8
                        )
                        input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                        # Chọn thư mục lưu bước xử lý cho từng file (nếu bật)
                        if show_pipeline:
                            base_name, _ = os.path.splitext(uploaded_file.name)
                            safe_name = base_name.replace(" ", "_")
                            out_dir = os.path.join(
                                "example_progress/progress_images", safe_name
                            )
                        else:
                            out_dir = "example_progress/progress_images"

                        (
                            result_text,
                            confidence,
                            top3_idx,
                            top3_probs,
                            display_img,
                            progress,
                            shapes_labels,
                        ) = run_inference_for_image(
                            input_image, mode, show_pipeline, out_dir
                        )

                        with st.expander(
                            f"Ảnh {idx_file + 1}: {uploaded_file.name}",
                            expanded=(len(uploaded_files) == 1),
                        ):
                            st.markdown(
                                f'<div class="result-box">'
                                f'<h1 style="text-align:center; color:#2E7D32">{result_text}</h1>'
                                f'<p style="text-align:center; font-size:24px; color:#1976D2">Độ tin cậy: {confidence*100:.1f}%</p>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            col_orig, col_proc = st.columns(2)
                            with col_orig:
                                st.image(
                                    input_image,
                                    channels="BGR",
                                    use_container_width=True,
                                    caption="Ảnh gốc",
                                )
                            with col_proc:
                                st.image(
                                    display_img,
                                    use_container_width=True,
                                    caption="Ảnh sau xử lý cuối cùng",
                                    clamp=True,
                                )

                            st.markdown("Top 3 dự đoán:")
                            for idx_pred, prob in zip(top3_idx, top3_probs):
                                if mode == "Chữ số (MNIST)":
                                    label = str(idx_pred)
                                elif mode == "Hình học (Shapes)" and shapes_labels is not None:
                                    label = shapes_labels[idx_pred]
                                else:
                                    label = f"{CHINESE_LABELS[idx_pred]} ({CHINESE_LABELS_VN[idx_pred]})"

                                st.write(label)
                                st.progress(float(prob))
                                st.write(f"{prob*100:.1f}%")

                            if show_pipeline and progress:
                                st.markdown("---")
                                st.subheader("Các bước xử lý ảnh")

                                step_keys = sorted(
                                    [k for k in progress.keys() if k.startswith("step")]
                                )
                                num_cols = 3
                                for i in range(0, len(step_keys), num_cols):
                                    cols = st.columns(num_cols)
                                    for j in range(num_cols):
                                        if i + j < len(step_keys):
                                            key = step_keys[i + j]
                                            step_img = progress[key]
                                            step_name = (
                                                key.replace("step", "Bước ")
                                                .replace("_", " ")
                                                .title()
                                            )
                                            with cols[j]:
                                                st.image(
                                                    step_img,
                                                    caption=step_name,
                                                    use_container_width=True,
                                                    clamp=True,
                                                )

                            st.markdown("---")
                    except Exception as e:
                        st.error(f"Lỗi với file {uploaded_file.name}: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    st.markdown("---")
    st.markdown("""
    
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
