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

# Thêm thư mục src vào path
sys.path.append('src')

from preprocessing import preprocess_for_mnist, preprocess_for_shapes

# Cấu hình trang
st.set_page_config(
    page_title="Nhận dạng CNN",
    page_icon="🔍",
    layout="wide"
)

# CSS tùy chỉnh
st.markdown("""
<style>
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
    
    mnist_path = 'models/mnist_model.h5'
    shapes_path = 'models/shapes_model.h5'
    
    if os.path.exists(mnist_path):
        models['mnist'] = keras.models.load_model(mnist_path)
    if os.path.exists(shapes_path):
        models['shapes'] = keras.models.load_model(shapes_path)
    
    return models


def main():
    """Hàm chính của ứng dụng"""
    
    st.title("🔍 Nhận dạng Chữ số và Hình học - V3 ULTRA ROBUST")
    st.markdown("*Xử lý hoàn hảo mọi loại ảnh - nền trắng, nền đen, màu sắc bất kỳ*")
    st.markdown("---")
    
    models = load_models()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ Cài đặt")
        mode = st.radio("Chế độ:", ["Chữ số (MNIST)", "Hình học (Shapes)"])
        
        # Thêm option hiển thị pipeline
        show_pipeline = st.checkbox("📊 Hiển thị từng bước xử lý", value=False)
        
        st.subheader("📤 Tải ảnh")
        uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(image, channels="BGR", use_container_width=True, caption="Ảnh gốc")
    
    with col2:
        st.subheader("🎯 Kết quả")
        
        if uploaded_file and st.button("🔍 Nhận dạng"):
            model_key = 'mnist' if mode == "Chữ số (MNIST)" else 'shapes'
            
            if model_key not in models:
                st.error("❌ Model chưa được tải!")
            else:
                with st.spinner("Đang xử lý..."):
                    try:
                        if mode == "Chữ số (MNIST)":
                            processed, display_img, progress = preprocess_for_mnist(
                                image, 
                                save_steps=show_pipeline,
                                output_dir="example_progress/progress_images"
                            )
                            prediction = models['mnist'].predict(processed, verbose=0)
                            result = np.argmax(prediction)
                            confidence = prediction[0][result]
                            result_text = f"Chữ số: **{result}**"
                            
                            # Top 3
                            top3_idx = np.argsort(prediction[0])[-3:][::-1]
                            top3_probs = prediction[0][top3_idx]
                            
                        else:
                            processed, display_img, progress = preprocess_for_shapes(
                                image,
                                save_steps=show_pipeline,
                                output_dir="example_progress/progress_images"
                            )
                            prediction = models['shapes'].predict(processed, verbose=0)
                            result = np.argmax(prediction)
                            confidence = prediction[0][result]
                            shapes = ['Hình tròn', 'Hình chữ nhật', 'Hình tam giác']
                            result_text = f"Hình: **{shapes[result]}**"
                            
                            # Top 3
                            top3_idx = np.argsort(prediction[0])[-3:][::-1]
                            top3_probs = prediction[0][top3_idx]
                        
                        # Hiển thị kết quả chính
                        st.markdown(
                            f'<div class="result-box">'
                            f'<h1 style="text-align:center; color:#2E7D32">{result_text}</h1>'
                            f'<p style="text-align:center; font-size:24px; color:#1976D2">Độ tin cậy: {confidence*100:.1f}%</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Hiển thị ảnh đã xử lý
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.image(display_img, use_container_width=True, caption="Ảnh sau xử lý cuối cùng", clamp=True)
                        
                        with col_b:
                            # Top 3 predictions
                            st.markdown("**📊 Top 3 dự đoán:**")
                            for idx, prob in zip(top3_idx, top3_probs):
                                if mode == "Chữ số (MNIST)":
                                    label = str(idx)
                                else:
                                    label = shapes[idx]
                                
                                # Progress bar cho mỗi prediction
                                st.write(f"**{label}**")
                                st.progress(float(prob))
                                st.write(f"{prob*100:.1f}%")
                        
                        # Hiển thị pipeline nếu được chọn
                        if show_pipeline and progress:
                            st.markdown("---")
                            st.subheader("📸 Các bước xử lý ảnh")
                            
                            # Hiển thị grid các bước
                            step_keys = sorted([k for k in progress.keys() if k.startswith('step')])
                            
                            # Hiển thị 3 ảnh/hàng
                            num_cols = 3
                            for i in range(0, len(step_keys), num_cols):
                                cols = st.columns(num_cols)
                                for j in range(num_cols):
                                    if i + j < len(step_keys):
                                        key = step_keys[i + j]
                                        step_img = progress[key]
                                        
                                        # Tên bước dễ đọc
                                        step_name = key.replace('step', 'Bước ').replace('_', ' ').title()
                                        
                                        with cols[j]:
                                            st.image(step_img, caption=step_name, use_container_width=True, clamp=True)
                            
                            st.info(f"✅ Đã lưu {len(step_keys)} ảnh vào: example_progress/progress_images/")
                    
                    except Exception as e:
                        st.error(f"❌ Lỗi: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666'>
        <p><strong>✨ V3 Ultra Robust Features:</strong></p>
        <p>🎯 Corner-based background detection</p>
        <p>🔄 Perfect normalization: WHITE on BLACK</p>
        <p>📐 CLAHE + Otsu + Morphology pipeline</p>
        <p>📸 Save all processing steps</p>
        <p>💻 Running Locally</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
