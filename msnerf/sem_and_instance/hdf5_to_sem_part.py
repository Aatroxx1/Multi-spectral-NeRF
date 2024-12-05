import numpy as np
from PIL import Image
import os
import concurrent.futures
import logging
from hdf5_reader import ReadAllHDF5

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_psn_semantic(folder, seg_symbol, max_workers=8):
    """
    预处理语义分割数据，将其转换为8位灰度图像并保存。

    数值映射规则:
    - 0 -> 0 (背景)
    - 1 -> 128 (前景)
    - 255 -> 255 (保持不变)

    参数:
    - folder (str): 数据所在的文件夹路径
    - seg_symbol (str): 分割符号，用于定位HDF5文件
    - max_workers (int): 线程池的最大工作线程数
    """
    hdf5_path = os.path.join(folder, seg_symbol, "hdf5")

    # 创建保存目录结构
    output_base_dir = os.path.join(folder, "images")
    output_dir = os.path.join(output_base_dir, "part4")  # 添加part4子文件夹
    os.makedirs(output_dir, exist_ok=True)

    HDF5_Sem = ReadAllHDF5(hdf5_path)
    semantic_data = HDF5_Sem.data

    def process_and_save_image(image_name, image_data):
        """
        处理单个图像数据并保存为8位灰度PNG。

        参数:
        - image_name (str): 图像的名称
        - image_data (np.ndarray): 图像的语义数据
        """
        try:
            # 构建保存路径，添加_part4后缀
            img_save_path = os.path.join(output_dir, f'{image_name}_part4.png')
            logging.info(f"Processing image: {image_name}")

            # 创建灰度图像，进行值映射
            # 0 -> 0
            # 1 -> 128
            # 255 -> 255 (保持不变)
            gray_image = np.where(image_data == 255, 255,
                                np.where(image_data > 0, 128, 0)).astype(np.uint8)

            # 创建PIL图像并保存为灰度图
            img = Image.fromarray(gray_image, 'L')
            img.save(img_save_path)
            logging.info(f'Saved image: {img_save_path}')

        except Exception as e:
            logging.error(f"Failed to process {image_name}: {e}")

    # 使用线程池并行处理和保存图像
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for image_name, image_data in semantic_data.items():
            futures.append(executor.submit(process_and_save_image, image_name, image_data))

        # 等待所有任务完成，并处理可能的异常
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Generated an exception: {exc}")

    logging.info(f"所有图像已成功处理和保存到: {output_dir}")

if __name__ == "__main__":
    folder_path = r'data/Semantic'
    seg_symbol = "YOLO11_Seg"
    preprocess_psn_semantic(folder_path, seg_symbol)