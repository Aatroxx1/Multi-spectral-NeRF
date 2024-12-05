import os
import cv2
import numpy as np

# 定义输入和输出路径
input_base_path = r"D:\files\PHD\myNeRF\nerfstudio\data\sim_plant\render"
output_path = r"D:\files\PHD\myNeRF\nerfstudio\data\sim_plant\images"

# 确保输出文件夹存在
os.makedirs(output_path, exist_ok=True)

# 文件夹名称（part1 到 part9，外加 part10 透明度通道）
folders = [f"part{i}" for i in range(1, 10)]
alpha_folder = "part10"

# 图片序号范围
image_indices = range(1, 151)

# 遍历所有图片序号
for idx in image_indices:
    # 创建一个空的光谱滤光阵列图像 (1080p, 每个像素一个波段值)
    output_image = np.zeros((1080, 1920), dtype=np.uint8)

    # 加载透明度图像 (part10)
    alpha_filename = f"Image{idx:04d}.png"
    alpha_file_path = os.path.join(input_base_path, alpha_folder, alpha_filename)

    # 读取透明度图像并检查
    alpha_img = cv2.imread(alpha_file_path, cv2.IMREAD_UNCHANGED)
    if alpha_img is None:
        raise FileNotFoundError(f"Alpha image not found: {alpha_file_path}")


    alpha_mask = (alpha_img[..., 0] > 0).astype(np.uint8)  # 生成透明度掩码（透明度为 0 的区域将被屏蔽）

    # 遍历 9 个文件夹，将数据填充到光谱滤光阵列中
    for folder_index, folder_name in enumerate(folders):
        # 获取当前文件夹路径
        folder_path = os.path.join(input_base_path, folder_name)

        # 生成当前图片的文件名（如 Image0001.png）
        filename = f"Image{idx:04d}.png"
        input_file_path = os.path.join(folder_path, filename)

        # 读取图片并转换为灰度图（8 位）
        img = cv2.imread(input_file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {input_file_path}")

        # 检查是否为 32 位图片
        if img.dtype != np.float32 and img.dtype != np.uint16 and img.dtype != np.uint8:
            raise ValueError(f"Unexpected image format in {input_file_path}, expected 32-bit float.")

        img = img[..., 0]
        # 计算当前光谱滤光阵列位置（3x3 的索引）
        row_offset = folder_index // 3
        col_offset = folder_index % 3

        # 将对应的像素填充到输出图像中
        output_image[row_offset::3, col_offset::3] = img[row_offset::3, col_offset::3]

    # 应用透明度掩码，将透明度为 0 的区域设置为 0
    output_image[alpha_mask == 0] = 0

    # 保存生成的滤光阵列图像
    output_filename = f"{idx:04d}.png"
    output_file_path = os.path.join(output_path, output_filename)
    cv2.imwrite(output_file_path, output_image)

    print(f"Processed and saved: {output_file_path}")

print("All images processed successfully!")