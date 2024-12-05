from os import listdir
from PIL import Image
import numpy as np
import os

# 数据文件夹路径
data_dir = r"E:\SeMs-NeRF\20230822\2_v1_frames"
band2 = os.path.join(data_dir, "part2")
band10 = os.path.join(data_dir, "part10")
save_path = r"E:\SeMs-NeRF\ndvi"

# 获取图像数量
img_num = len(listdir(band2))

# 遍历每张图像，计算 NDVI
for i in range(img_num):
    # 读取band2和band10的图像
    img2_path = os.path.join(band2, f"frame_{(i + 1):04}_part2.png")
    img10_path = os.path.join(band10, f"frame_{(i + 1):04}_part10.png")

    img2 = Image.open(img2_path)
    img10 = Image.open(img10_path)

    # 转换为 numpy 数组
    img2_np = np.array(img2).astype(float)
    img10_np = np.array(img10).astype(float)

    # 避免除以0的情况，添加一个小的常数
    epsilon = 1e-5

    # 计算 NDVI
    ndvi = (img10_np - img2_np) / (img10_np + img2_np + epsilon)

    # 将NDVI值限制在 [-1, 1] 范围
    ndvi = np.clip(ndvi, -1, 1)

    # 将NDVI数组转换回图像并保存
    ndvi_img = Image.fromarray(((ndvi + 1) / 2 * 255).astype(np.uint8))  # 将NDVI从[-1,1]映射到[0,255]
    ndvi_img.save(os.path.join(save_path, f"ndvi_{(i + 1):04}.png"))

    print(f"NDVI for frame {i + 1} saved.")