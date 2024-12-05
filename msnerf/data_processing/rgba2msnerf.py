import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

# 指定图像文件夹路径
input_folder = r'D:\files\PHD\myNeRF\nerfstudio\data\tomato1080\images'

sim_times = 5
variations = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0.3, 0.4, 0.3],
    [0.8, 0.1, 0.1]
]


def process_image(filename):
    img_path = os.path.join(input_folder, filename)
    img = Image.open(img_path).convert('RGBA')  # 确保图像是 RGBA 格式

    # 将图像转换为 numpy 数组
    img_array = np.array(img)

    # 分离 RGBA 通道
    r, g, b, a = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2], img_array[:, :, 3]

    for i in range(sim_times):
        gray = variations[i][0] * r + variations[i][1] * g + variations[i][2] * b
        gray[a == 0] = 255
        base_gray_img = gray.astype(np.uint8)
        fluctuated_gray = np.clip(base_gray_img, 0, 255)
        fluctuated_gray_img = Image.fromarray(fluctuated_gray, mode='L')

        output_path = os.path.join(input_folder, "part" + str(i + 1), filename[:-4] + '_part' + str(i + 1) + '.png')
        fluctuated_gray_img.save(output_path)


# 使用多线程加速图像处理
def main():
    with ThreadPoolExecutor() as executor:
        for filename in os.listdir(input_folder):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                executor.submit(process_image, filename)


if __name__ == "__main__":
    for i in range(sim_times):
        folder_path = os.path.join(input_folder, "part" + str(i + 1))
        os.makedirs(folder_path, exist_ok=True)
    main()

print("所有图像转换完成。")
