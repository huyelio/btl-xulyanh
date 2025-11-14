"""
Module tiền xử lý ảnh cho dự án nhận dạng chữ số và hình học
Bao gồm các kỹ thuật: histogram equalization, filtering, thresholding,
morphological operations, edge detection, connected components
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict
import os


class ImagePreprocessor:
    """Lớp xử lý tiền xử lý ảnh với đầy đủ các bước"""
    
    def __init__(self, save_progress: bool = False, output_dir: str = "example_progress"):
        """
        Args:
            save_progress: Có lưu ảnh từng bước không
            output_dir: Thư mục lưu ảnh progress
        """
        self.save_progress = save_progress
        self.output_dir = output_dir
        if save_progress and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.progress_images = {}
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Chuyển ảnh sang grayscale nếu cần"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        self.progress_images['01_grayscale'] = gray
        return gray
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Chuẩn hóa pixel về [0, 1]"""
        normalized = image.astype(np.float32) / 255.0
        # Lưu dạng uint8 để hiển thị
        self.progress_images['02_normalized'] = (normalized * 255).astype(np.uint8)
        return normalized
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Tăng cường ảnh bằng Histogram Equalization"""
        # Chuyển về uint8 nếu là float
        if image.dtype == np.float32 or image.dtype == np.float64:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        equalized = cv2.equalizeHist(img_uint8)
        self.progress_images['03_histogram_equalized'] = equalized
        return equalized
    
    def denoise_gaussian(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Loại bỏ nhiễu bằng Gaussian filter"""
        if image.dtype == np.float32 or image.dtype == np.float64:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        denoised = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)
        self.progress_images['04_gaussian_denoised'] = denoised
        return denoised
    
    def denoise_median(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Loại bỏ nhiễu bằng Median filter"""
        if image.dtype == np.float32 or image.dtype == np.float64:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        denoised = cv2.medianBlur(img_uint8, kernel_size)
        self.progress_images['05_median_denoised'] = denoised
        return denoised
    
    def threshold_otsu(self, image: np.ndarray) -> np.ndarray:
        """Phân ngưỡng bằng Otsu"""
        if image.dtype == np.float32 or image.dtype == np.float64:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        _, thresholded = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.progress_images['06_otsu_threshold'] = thresholded
        return thresholded
    
    def threshold_adaptive(self, image: np.ndarray, block_size: int = 11) -> np.ndarray:
        """Phân ngưỡng bằng Adaptive Threshold"""
        if image.dtype == np.float32 or image.dtype == np.float64:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        thresholded = cv2.adaptiveThreshold(
            img_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, 2
        )
        self.progress_images['07_adaptive_threshold'] = thresholded
        return thresholded
    
    def morphology_erosion(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Phép hình thái học: Erosion"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(image, kernel, iterations=1)
        self.progress_images['08_erosion'] = eroded
        return eroded
    
    def morphology_dilation(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Phép hình thái học: Dilation"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(image, kernel, iterations=1)
        self.progress_images['09_dilation'] = dilated
        return dilated
    
    def morphology_opening(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Phép hình thái học: Opening (erosion sau đó dilation)"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        self.progress_images['10_opening'] = opened
        return opened
    
    def morphology_closing(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Phép hình thái học: Closing (dilation sau đó erosion)"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        self.progress_images['11_closing'] = closed
        return closed
    
    def edge_detection_canny(self, image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
        """Tách biên bằng Canny Edge Detection"""
        if image.dtype == np.float32 or image.dtype == np.float64:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        edges = cv2.Canny(img_uint8, low, high)
        self.progress_images['12_canny_edges'] = edges
        return edges
    
    def connected_components(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """Trích xuất thành phần kết nối"""
        num_labels, labels = cv2.connectedComponents(image)
        
        # Tạo ảnh màu để hiển thị các components
        colored = np.zeros((*image.shape, 3), dtype=np.uint8)
        for label in range(1, num_labels):
            mask = labels == label
            color = np.random.randint(0, 255, 3).tolist()
            colored[mask] = color
        
        self.progress_images['13_connected_components'] = colored
        return num_labels, labels
    
    def convex_hull(self, image: np.ndarray) -> np.ndarray:
        """Tìm bao lồi (Convex Hull)"""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Tạo ảnh mới để vẽ convex hull
        hull_image = np.zeros_like(image)
        for contour in contours:
            hull = cv2.convexHull(contour)
            cv2.drawContours(hull_image, [hull], 0, 255, -1)
        
        self.progress_images['14_convex_hull'] = hull_image
        return hull_image
    
    def full_pipeline(self, image: np.ndarray, for_mnist: bool = True) -> np.ndarray:
        """
        Pipeline đầy đủ cho xử lý ảnh
        
        Args:
            image: Ảnh đầu vào
            for_mnist: True nếu xử lý cho MNIST, False cho shapes
        
        Returns:
            Ảnh đã xử lý
        """
        # Bước 1: Chuyển sang grayscale
        gray = self.to_grayscale(image)
        
        # Bước 2: Histogram equalization
        equalized = self.histogram_equalization(gray)
        
        # Bước 3: Denoise với Gaussian
        denoised = self.denoise_gaussian(equalized, kernel_size=5)
        
        # Bước 4: Threshold
        if for_mnist:
            thresholded = self.threshold_otsu(denoised)
        else:
            thresholded = self.threshold_adaptive(denoised, block_size=11)
        
        # Bước 5: Morphological operations để làm sạch
        opened = self.morphology_opening(thresholded, kernel_size=3)
        closed = self.morphology_closing(opened, kernel_size=3)
        
        # Bước 6: Edge detection (để minh họa)
        edges = self.edge_detection_canny(closed)
        
        # Bước 7: Connected components (để minh họa)
        num_labels, labels = self.connected_components(closed)
        
        # Chuẩn hóa về [0, 1] cho model
        processed = closed.astype(np.float32) / 255.0
        self.progress_images['15_final_processed'] = closed
        
        return processed
    
    def save_progress_images(self, prefix: str = "sample"):
        """Lưu các ảnh progress ra file"""
        if not self.save_progress:
            return
        
        for name, image in self.progress_images.items():
            filename = f"{prefix}_{name}.png"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, image)
        
        print(f"Đã lưu {len(self.progress_images)} ảnh progress vào {self.output_dir}/")
    
    def get_progress_images(self) -> Dict[str, np.ndarray]:
        """Trả về dict các ảnh progress"""
        return self.progress_images


def preprocess_for_mnist(image: np.ndarray, target_size: Tuple[int, int] = (28, 28), 
                         save_steps: bool = False, output_dir: str = "example_progress/progress_images") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Preprocessing ROBUST cho MNIST - Xử lý mọi loại ảnh về chuẩn: TRẮNG trên ĐEN
    
    MỤC TIÊU: Chữ số TRẮNG (255) trên nền ĐEN (0) - giống MNIST gốc
    
    Args:
        image: Ảnh đầu vào (bất kỳ màu, nền gì)
        target_size: Kích thước đích (28, 28)
        save_steps: Có lưu ảnh từng bước không
        output_dir: Thư mục lưu ảnh
    
    Returns:
        Tuple (normalized_image, display_image, progress_dict):
            - normalized_image: shape (1, 28, 28, 1) - để predict
            - display_image: shape (28, 28) - để hiển thị
            - progress_dict: dict chứa ảnh từng bước
    """
    if save_steps:
        os.makedirs(output_dir, exist_ok=True)
    
    progress = {}
    
    # BƯỚC 1: Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    progress['step01_grayscale'] = gray.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step01_grayscale.png', gray)
    
    # BƯỚC 2: Histogram Equalization (CLAHE - tốt hơn equalizeHist)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    progress['step02_clahe'] = enhanced.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step02_clahe.png', enhanced)
    
    # BƯỚC 3: Gaussian Blur để giảm nhiễu
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    progress['step03_gaussian_blur'] = blurred.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step03_gaussian_blur.png', blurred)
    
    # BƯỚC 4: Otsu Threshold - tự động tìm ngưỡng tối ưu
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    progress['step04_otsu_threshold'] = binary.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step04_otsu_threshold.png', binary)
    
    # BƯỚC 5: XÁC ĐỊNH NỀN - Phương pháp ROBUST
    # Đếm pixel ở 4 góc ảnh (nền thường ở góc)
    h, w = binary.shape
    corner_size = min(h, w) // 10  # 10% kích thước
    corners = [
        binary[0:corner_size, 0:corner_size],              # Top-left
        binary[0:corner_size, w-corner_size:w],            # Top-right
        binary[h-corner_size:h, 0:corner_size],            # Bottom-left
        binary[h-corner_size:h, w-corner_size:w]           # Bottom-right
    ]
    
    # Đếm pixel trắng ở góc
    corner_white_ratio = np.mean([np.sum(corner == 255) / corner.size for corner in corners])
    
    # Nếu >50% góc là trắng → nền trắng → CẦN INVERT để có nền đen
    need_invert = corner_white_ratio > 0.5
    
    if need_invert:
        binary = cv2.bitwise_not(binary)
        progress['step05_inverted'] = binary.copy()
        if save_steps: cv2.imwrite(f'{output_dir}/step05_inverted.png', binary)
    else:
        progress['step05_inverted'] = binary.copy()
        if save_steps: cv2.imwrite(f'{output_dir}/step05_no_invert_needed.png', binary)
    
    # BƯỚC 6: Morphology Opening - loại bỏ nhiễu nhỏ
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    progress['step06_morphology_open'] = opened.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step06_morphology_open.png', opened)
    
    # BƯỚC 7: Morphology Closing - lấp lỗ nhỏ
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    progress['step07_morphology_close'] = closed.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step07_morphology_close.png', closed)
    
    # BƯỚC 8: Tìm contour và crop
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Không tìm thấy → resize trực tiếp
        resized = cv2.resize(closed, target_size, interpolation=cv2.INTER_AREA)
        progress['step08_contour'] = closed.copy()
    else:
        # Tìm contour lớn nhất
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_cont, h_cont = cv2.boundingRect(largest_contour)
        
        # Vẽ contour để debug
        contour_img = cv2.cvtColor(closed.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(contour_img, (x, y), (x+w_cont, y+h_cont), (0, 255, 0), 2)
        progress['step08_contour'] = contour_img
        if save_steps: cv2.imwrite(f'{output_dir}/step08_contour.png', contour_img)
        
        # Crop với padding
        pad = max(2, int(min(w_cont, h_cont) * 0.15))  # 15% padding
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(closed.shape[1], x + w_cont + pad)
        y2 = min(closed.shape[0], y + h_cont + pad)
        
        digit = closed[y1:y2, x1:x2]
        progress['step09_cropped'] = digit.copy()
        if save_steps: cv2.imwrite(f'{output_dir}/step09_cropped.png', digit)
        
        # BƯỚC 9: Resize giữ tỷ lệ + CENTER (giống MNIST)
        # Fit vào 20x20, để border 4px mỗi bên
        dh, dw = digit.shape
        target_inner = 20
        scale = min(target_inner / dw, target_inner / dh)
        
        new_w = int(dw * scale)
        new_h = int(dh * scale)
        
        # Resize với INTER_AREA (tốt nhất cho downscale)
        resized_digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
        progress['step10_resized'] = resized_digit.copy()
        if save_steps: cv2.imwrite(f'{output_dir}/step10_resized.png', resized_digit)
        
        # BƯỚC 10: Center vào canvas 28x28
        canvas = np.zeros(target_size, dtype=np.uint8)
        top = (target_size[0] - new_h) // 2
        left = (target_size[1] - new_w) // 2
        canvas[top:top+new_h, left:left+new_w] = resized_digit
        
        resized = canvas
        progress['step11_centered'] = resized.copy()
        if save_steps: cv2.imwrite(f'{output_dir}/step11_centered.png', resized)
    
    # BƯỚC 11: Làm mịn ranh giới nhẹ (giống MNIST gốc)
    smoothed = cv2.GaussianBlur(resized, (3, 3), 0)
    progress['step12_final_smoothed'] = smoothed.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step12_final_smoothed.png', smoothed)
    
    # BƯỚC 12: Normalize về [0, 1]
    normalized = smoothed.astype(np.float32) / 255.0
    
    return normalized.reshape(1, 28, 28, 1), smoothed, progress


def preprocess_for_shapes(image: np.ndarray, target_size: Tuple[int, int] = (64, 64),
                          save_steps: bool = False, output_dir: str = "example_progress/progress_images") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Preprocessing ROBUST cho Shapes - Xử lý mọi loại ảnh về chuẩn: TRẮNG trên ĐEN
    
    MỤC TIÊU: Hình TRẮNG (255) trên nền ĐEN (0)
    
    Args:
        image: Ảnh đầu vào
        target_size: Kích thước đích (64, 64)
        save_steps: Có lưu ảnh từng bước không
        output_dir: Thư mục lưu ảnh
    
    Returns:
        Tuple (normalized_image, display_image, progress_dict)
    """
    if save_steps:
        os.makedirs(output_dir, exist_ok=True)
    
    progress = {}
    
    # BƯỚC 1: Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    progress['step01_grayscale'] = gray.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step01_grayscale.png', gray)
    
    # BƯỚC 2: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    progress['step02_clahe'] = enhanced.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step02_clahe.png', enhanced)
    
    # BƯỚC 3: Gaussian Blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    progress['step03_gaussian_blur'] = blurred.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step03_gaussian_blur.png', blurred)
    
    # BƯỚC 4: Otsu Threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    progress['step04_otsu_threshold'] = binary.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step04_otsu_threshold.png', binary)
    
    # BƯỚC 5: Xác định nền và invert nếu cần
    h, w = binary.shape
    corner_size = min(h, w) // 10
    corners = [
        binary[0:corner_size, 0:corner_size],
        binary[0:corner_size, w-corner_size:w],
        binary[h-corner_size:h, 0:corner_size],
        binary[h-corner_size:h, w-corner_size:w]
    ]
    corner_white_ratio = np.mean([np.sum(corner == 255) / corner.size for corner in corners])
    
    need_invert = corner_white_ratio > 0.5
    if need_invert:
        binary = cv2.bitwise_not(binary)
        progress['step05_inverted'] = binary.copy()
        if save_steps: cv2.imwrite(f'{output_dir}/step05_inverted.png', binary)
    else:
        progress['step05_inverted'] = binary.copy()
        if save_steps: cv2.imwrite(f'{output_dir}/step05_no_invert_needed.png', binary)
    
    # BƯỚC 6: Morphology Closing - lấp lỗ
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    progress['step06_morphology_close'] = closed.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step06_morphology_close.png', closed)
    
    # BƯỚC 7: Morphology Opening - loại nhiễu
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    progress['step07_morphology_open'] = opened.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step07_morphology_open.png', opened)
    
    # BƯỚC 8: Tìm contour
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_cont, h_cont = cv2.boundingRect(largest_contour)
        
        # Vẽ contour
        contour_img = cv2.cvtColor(opened.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(contour_img, (x, y), (x+w_cont, y+h_cont), (0, 255, 0), 2)
        progress['step08_contour'] = contour_img
        if save_steps: cv2.imwrite(f'{output_dir}/step08_contour.png', contour_img)
        
        # Crop
        pad = max(2, int(min(w_cont, h_cont) * 0.1))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(opened.shape[1], x + w_cont + pad)
        y2 = min(opened.shape[0], y + h_cont + pad)
        
        shape = opened[y1:y2, x1:x2]
        progress['step09_cropped'] = shape.copy()
        if save_steps: cv2.imwrite(f'{output_dir}/step09_cropped.png', shape)
        
        # Resize giữ tỷ lệ, fit vào 56x56
        sh, sw = shape.shape
        target_inner = 56
        scale = min(target_inner / sw, target_inner / sh)
        new_w = int(sw * scale)
        new_h = int(sh * scale)
        
        resized_shape = cv2.resize(shape, (new_w, new_h), interpolation=cv2.INTER_AREA)
        progress['step10_resized'] = resized_shape.copy()
        if save_steps: cv2.imwrite(f'{output_dir}/step10_resized.png', resized_shape)
        
        # Center vào 64x64
        canvas = np.zeros(target_size, dtype=np.uint8)
        top = (target_size[0] - new_h) // 2
        left = (target_size[1] - new_w) // 2
        canvas[top:top+new_h, left:left+new_w] = resized_shape
        
        resized = canvas
        progress['step11_centered'] = resized.copy()
        if save_steps: cv2.imwrite(f'{output_dir}/step11_centered.png', resized)
    else:
        resized = cv2.resize(opened, target_size, interpolation=cv2.INTER_AREA)
        progress['step11_centered'] = resized.copy()
        if save_steps: cv2.imwrite(f'{output_dir}/step11_centered.png', resized)
    
    # BƯỚC 12: Final - Normalize
    normalized = resized.astype(np.float32) / 255.0
    progress['step12_final'] = resized.copy()
    if save_steps: cv2.imwrite(f'{output_dir}/step12_final.png', resized)
    
    return normalized.reshape(1, 64, 64, 1), resized, progress
