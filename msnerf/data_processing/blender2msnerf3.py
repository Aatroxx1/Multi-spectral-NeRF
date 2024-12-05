import os
import cv2
import numpy as np

# 定义输入和输出路径
input_path = r"D:\files\PHD\myNeRF\nerfstudio\data\sim_plant\images"
output_base_path = r"D:\files\PHD\myNeRF\nerfstudio\data\sim_plant\images"

# 图片序号范围
image_indices = range(1, 151)

# 遍历所有图片
for idx in image_indices:
    # 读取合成图像
    input_filename = f"{idx:04d}.png"
    input_file_path = os.path.join(input_path, input_filename)
    img = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)

    # 创建 9 个波段的目录
    for band_index in range(1, 10):
        band_output_path = os.path.join(output_base_path, f"part{band_index}")
        os.makedirs(band_output_path, exist_ok=True)

    # 遍历光谱滤光阵列 3x3
    for row_offset in range(3):
        for col_offset in range(3):
            # 计算波段索引（从 1 开始）
            band_index = row_offset * 3 + col_offset + 1

            # 提取对应波段的像素
            band_image = img[row_offset::3, col_offset::3]

            # 保存波段图像
            output_filename = f"{idx:04d}.png"
            output_file_path = os.path.join(output_base_path, f"part{band_index}", output_filename)
            cv2.imwrite(output_file_path, band_image)

    print(f"Processed and split: {input_file_path}")

print("All images split into bands successfully!")