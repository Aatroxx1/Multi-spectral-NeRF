import json
import os

import numpy as np

path = r"D:\files\PHD\myNeRF\nerfstudio\data\sim_plant\render"

input_file = path + r"\camera_transforms_dynamic.txt"
output_file = path + r"\transforms.json"


# 设置图像宽度和高度 (根据实际情况调整)
image_width = 1920
image_height = 1080

# 假设相机的水平视场角，或者根据实际数据计算
f = 50 / 36 * image_width  # 例如：根据相机参数计算出来的视场角

# 将矩阵存储到这个列表
camera_data = {
    "fl_x": f,
    "fl_y": f,
    "cx": image_width / 2,
    "cy": image_height / 2,
    "h": 1080,
    "w": 1920,
    "frames": []
}

# 读取文件并处理矩阵
with open(input_file, 'r') as f:
    lines = f.readlines()

current_matrix = []
frame_count = 1

for line in lines:
    line = line.strip()

    # 如果是空行，则跳过
    if not line:
        continue

    # 如果当前矩阵读取了4行，则处理成一个4x4矩阵
    if len(current_matrix) == 4:
        # 转换为numpy数组
        transform_matrix = np.array(current_matrix, dtype=float)

        # 将矩阵转换为列表形式，以便JSON序列化
        transform_list = transform_matrix.tolist()

        # 假设图像文件名格式为frame_xxxx.png
        frame_data = {
            "file_path": f"./images/{frame_count :04d}.png",
            "transform_matrix": transform_list
        }

        # 添加到camera_data
        camera_data["frames"].append(frame_data)

        # 重置matrix，准备下一个
        current_matrix = []
        frame_count += 1

    # 读取矩阵的每一行
    current_matrix.append([float(x) for x in line.split()])

# 处理最后一个矩阵（如果存在未处理的矩阵）
if len(current_matrix) == 4:
    transform_matrix = np.array(current_matrix, dtype=float)
    transform_list = transform_matrix.tolist()
    frame_data = {
        "file_path": f"./images/{frame_count:04d}.png",
        "transform_matrix": transform_list
    }
    camera_data["frames"].append(frame_data)

# 将数据保存为JSON文件
with open(output_file, 'w') as json_file:
    json.dump(camera_data, json_file, indent=4)

print(f"Camera transforms successfully converted to {output_file}")

# for filename in os.listdir(path):
#     # 检查文件名是否是四位数字
#     if filename.endswith(".png") and len(filename) == 8:  # 假设文件扩展名是 .txt
#         base_name = filename[:4]  # 提取文件名前四位数字
#         if base_name.isdigit():  # 确保前四位确实是数字
#             new_number = int(base_name) + (batch - 1) * 1000  # 增加 1000
#             new_filename = f"{new_number:04d}.png"  # 保证新的文件名依旧是四位数字
#             old_file_path = os.path.join(path, filename)
#             new_file_path = os.path.join(path, new_filename)
#             os.rename(old_file_path, new_file_path)  # 重命名文件
#             print(f"Renamed: {filename} -> {new_filename}")
