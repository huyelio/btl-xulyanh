"""
Script sinh dữ liệu hình học cơ bản (circle, rectangle, triangle)
Sử dụng OpenCV để vẽ các hình
"""

import cv2
import numpy as np
import os
from typing import Tuple, List
import random


class ShapeGenerator:
    """Lớp sinh các hình học cơ bản"""
    
    def __init__(self, img_size: int = 64, background_color: int = 0):
        """
        Args:
            img_size: Kích thước ảnh (img_size x img_size)
            background_color: Màu nền (0: đen, 255: trắng)
        """
        self.img_size = img_size
        self.background_color = background_color
    
    def generate_circle(self, radius: int = None, center: Tuple[int, int] = None) -> np.ndarray:
        """Sinh hình tròn"""
        img = np.ones((self.img_size, self.img_size), dtype=np.uint8) * self.background_color
        
        if radius is None:
            radius = random.randint(15, min(self.img_size // 3, 25))
        
        if center is None:
            center = (self.img_size // 2, self.img_size // 2)
        
        cv2.circle(img, center, radius, 255, -1)
        return img
    
    def generate_rectangle(self, width: int = None, height: int = None) -> np.ndarray:
        """Sinh hình chữ nhật"""
        img = np.ones((self.img_size, self.img_size), dtype=np.uint8) * self.background_color
        
        if width is None:
            width = random.randint(20, 40)
        if height is None:
            height = random.randint(20, 40)
        
        x1 = (self.img_size - width) // 2
        y1 = (self.img_size - height) // 2
        x2 = x1 + width
        y2 = y1 + height
        
        cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
        return img
    
    def generate_triangle(self) -> np.ndarray:
        """Sinh hình tam giác"""
        img = np.ones((self.img_size, self.img_size), dtype=np.uint8) * self.background_color
        
        # Tạo 3 đỉnh của tam giác
        center_x = self.img_size // 2
        center_y = self.img_size // 2
        size = random.randint(20, 30)
        
        # Đỉnh trên
        pt1 = (center_x, center_y - size)
        # Đỉnh dưới trái
        pt2 = (center_x - size, center_y + size)
        # Đỉnh dưới phải
        pt3 = (center_x + size, center_y + size)
        
        points = np.array([pt1, pt2, pt3], dtype=np.int32)
        cv2.fillPoly(img, [points], 255)
        
        return img
    
    def add_noise(self, image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Thêm nhiễu Gaussian vào ảnh"""
        noise = np.random.normal(0, noise_level * 255, image.shape)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
    def rotate_image(self, image: np.ndarray, angle: float = None) -> np.ndarray:
        """Xoay ảnh một góc ngẫu nhiên"""
        if angle is None:
            angle = random.uniform(-30, 30)
        
        center = (self.img_size // 2, self.img_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (self.img_size, self.img_size))
        return rotated


def generate_dataset(output_dir: str = "data/shapes", 
                    num_samples_per_class: int = 1000,
                    img_size: int = 64,
                    add_augmentation: bool = True):
    """
    Sinh dataset đầy đủ cho shapes
    
    Args:
        output_dir: Thư mục lưu dữ liệu
        num_samples_per_class: Số lượng mẫu mỗi lớp
        img_size: Kích thước ảnh
        add_augmentation: Có thêm augmentation không
    """
    shapes = ['circle', 'rectangle', 'triangle']
    generator = ShapeGenerator(img_size=img_size, background_color=0)
    
    # Tạo thư mục
    for split in ['train', 'test']:
        for shape in shapes:
            shape_dir = os.path.join(output_dir, split, shape)
            os.makedirs(shape_dir, exist_ok=True)
    
    print(f"Đang sinh dataset shapes tại {output_dir}...")
    
    for shape in shapes:
        print(f"\nSinh {shape}...")
        
        # Tính số mẫu cho train/test
        num_train = int(num_samples_per_class * 0.8)
        num_test = num_samples_per_class - num_train
        
        # Sinh train set
        for i in range(num_train):
            if shape == 'circle':
                img = generator.generate_circle()
            elif shape == 'rectangle':
                img = generator.generate_rectangle()
            else:  # triangle
                img = generator.generate_triangle()
            
            # Thêm augmentation
            if add_augmentation and random.random() > 0.5:
                img = generator.rotate_image(img)
            if add_augmentation and random.random() > 0.7:
                img = generator.add_noise(img, noise_level=0.05)
            
            filename = os.path.join(output_dir, 'train', shape, f'{shape}_{i:04d}.png')
            cv2.imwrite(filename, img)
            
            if (i + 1) % 200 == 0:
                print(f"  Train: {i + 1}/{num_train}")
        
        # Sinh test set
        for i in range(num_test):
            if shape == 'circle':
                img = generator.generate_circle()
            elif shape == 'rectangle':
                img = generator.generate_rectangle()
            else:  # triangle
                img = generator.generate_triangle()
            
            # Ít augmentation hơn cho test set
            if add_augmentation and random.random() > 0.7:
                img = generator.rotate_image(img, angle=random.uniform(-15, 15))
            
            filename = os.path.join(output_dir, 'test', shape, f'{shape}_{i:04d}.png')
            cv2.imwrite(filename, img)
        
        print(f"  Test: {num_test}/{num_test}")
        print(f"✓ Hoàn thành {shape}: {num_train} train + {num_test} test")
    
    print(f"\n✓ Đã sinh xong dataset tại {output_dir}/")
    print(f"  - Train: {num_train * len(shapes)} ảnh")
    print(f"  - Test: {num_test * len(shapes)} ảnh")


if __name__ == "__main__":
    # Sinh dataset với 1000 mẫu mỗi lớp
    generate_dataset(
        output_dir="data/shapes",
        num_samples_per_class=1000,
        img_size=64,
        add_augmentation=True
    )
    
    # Sinh một vài mẫu để demo
    print("\nSinh mẫu demo...")
    generator = ShapeGenerator(img_size=64)
    os.makedirs("example_progress", exist_ok=True)
    
    cv2.imwrite("example_progress/demo_circle.png", generator.generate_circle())
    cv2.imwrite("example_progress/demo_rectangle.png", generator.generate_rectangle())
    cv2.imwrite("example_progress/demo_triangle.png", generator.generate_triangle())
    
    print("✓ Đã lưu các mẫu demo vào example_progress/")

