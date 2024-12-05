import os
from os import listdir

import numpy as np
import pandas as pd
from PIL import Image
import random


def false_rgb(data_dir, save_path):
    # 数据文件夹路径
    band4 = os.path.join(data_dir, "part20")  # Red 波段
    band3 = os.path.join(data_dir, "part6")  # Green 波段
    band10 = os.path.join(data_dir, "part2")  # NIR 波段，映射为蓝色

    # 获取图像数量
    img_num = len(listdir(band4))

    # random_numbers = random.sample(range(img_num), 30)

    # 遍历每张图像，生成彩色图像
    for i in range(img_num):
        # 读取band2, band3, band10的图像
        img4_path = os.path.join(band4, f"frame_{(i + 1):04}_part20.png")  # Red
        img3_path = os.path.join(band3, f"frame_{(i + 1):04}_part6.png")  # Green
        img10_path = os.path.join(band10, f"frame_{(i + 1):04}_part2.png")  # NIR -> Blue

        img4 = Image.open(img4_path)  # Red
        img3 = Image.open(img3_path)  # Green
        img10 = Image.open(img10_path)  # NIR -> Blue

        # 转换为 numpy 数组
        img4_np = np.array(img4).astype(float)
        img3_np = np.array(img3).astype(float)
        img10_np = np.array(img10).astype(float)

        # 归一化到 [0, 255]，避免图像过暗或过亮
        def normalize(arr):
            arr_min = arr.min()
            arr_max = arr.max()
            return ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)

        img4_norm = normalize(img4_np)  # Red 通道
        img3_norm = normalize(img3_np)  # Green 通道
        img10_norm = normalize(img10_np)  # Blue 通道 (NIR 映射到 Blue)

        # 合并到一个 RGB 图像
        rgb_image = np.stack([img4_norm, img3_norm, img10_norm], axis=-1)  # [H, W, 3]，对应 Red, Green, Blue

        # 将 RGB 数组转换为图像
        rgb_img = Image.fromarray(rgb_image)

        # 保存彩色图像
        rgb_img.save(os.path.join(save_path, f"frame_{(i + 1):04}.png"))

    print(f"Color image for {data_dir} saved.")


if __name__ == '__main__':
    # cvs_path = r"E:\SeMs-NeRF\编号.csv"
    # input_dir = r"E:\SeMs-NeRF"
    # output_dir = r"E:\SeMs-NeRF\伪彩"
    # df = pd.read_csv(cvs_path)
    # # 遍历每一行
    # for index, row in df.iterrows():
    #     group_dir = row['old']
    #     output_group_dir = row['new']
    #     group_path = os.path.join(input_dir, group_dir+"_frames")
    #     output_group_path = os.path.join(output_dir, output_group_dir)
    #     os.makedirs(output_group_path, exist_ok=True)
    #     false_rgb(group_path, output_group_path)
    false_rgb(r"D:\files\PHD\myNeRF\nerfstudio\data\05_0\images", r"D:\files\PHD\myNeRF\nerfstudio\data\05_0\images_for_seg")